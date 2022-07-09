import json
import random
import torch
import numpy as np
import argparse
import soundfile as sf
# import musdb
# musdb package is stem based, can not read hq
# import museval
import norbert
from pathlib import Path
import scipy.signal
import resampy
from asteroid.models import XUMX
from asteroid.complex_nn import torch_complex_from_magphase
from asteroid.metrics import get_metrics
import os
import warnings
import sys
import pandas as pd
import torchaudio
from tqdm import tqdm
import MS_21Dataloader

os.environ['KMP_DUPLICATE_LIB_OK']='True'
COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar"]


def load_model(model_name, device="cpu"):
    print("Loading model from: {}".format(model_name), file=sys.stderr)
    model = XUMX.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, model.sources


def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2), rate, nperseg=n_fft, noverlap=n_fft - n_hopsize, boundary=True
    )
    return audio


def separate(
    audio,
    x_umx_target,
    instruments,
    niter=1,
    softmask=False,
    alpha=1.0,
    residual_model=False,
    device="cpu",
):
    """
    Performing the separation on audio input

    Parameters
    ----------
    audio: np.ndarray [shape=(nb_samples, nb_channels, nb_timesteps)]
        mixture audio

    x_umx_target: asteroid.models
        X-UMX model used for separating

    instruments: list
        The list of instruments, e.g., ["bass", "drums", "vocals"]

    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.

    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False

    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0

    residual_model: boolean
        computes a residual target, for custom separation scenarios
        when not all targets are available, defaults to False

    device: str
        set torch device. Defaults to `cpu`.

    Returns
    -------
    estimates: `dict` [`str`, `np.ndarray`]
        dictionary with all estimates obtained by the separation model.
    """

    # convert numpy audio to torch
    audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)
    audio_len = audio_torch.shape[-1]

    source_names = []
    V = []

    masked_tf_rep, _ = x_umx_target(audio_torch)
    # shape: (Sources, frames, batch, channels, fbin)

    for j, target in enumerate(instruments):
        Vj = masked_tf_rep[j, Ellipsis].cpu().detach().numpy()
        if softmask:
            # only exponentiate the model if we use softmask
            Vj = Vj ** alpha
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        V.append(Vj[:, 0, Ellipsis])  # remove sample dim
        source_names += [target]

    V = np.transpose(np.array(V), (1, 3, 2, 0))

    # convert to complex numpy type
    tmp = x_umx_target.encoder(audio_torch)
    X = torch_complex_from_magphase(tmp[0].permute(1, 2, 3, 0), tmp[1])
    X = X.detach().cpu().numpy()
    X = X[0].transpose(2, 1, 0)

    if residual_model or len(instruments) == 1:
        V = norbert.residual_model(V, X, alpha if softmask else 1)
        source_names += ["residual"] if len(instruments) > 1 else ["accompaniment"]

    Y = norbert.wiener(V, X.astype(np.complex128), niter, use_softmask=softmask)

    estimates = []
    for j, name in enumerate(source_names):
        audio_hat = istft(
            Y[..., j].T,
            rate=x_umx_target.sample_rate,
            n_fft=x_umx_target.in_chan,
            n_hopsize=x_umx_target.n_hop,
        )
        pad_len = abs(audio_hat.shape[1] - audio_len)
        if pad_len !=0:
            audio_pad = np.pad(audio_hat.squeeze(), (0,pad_len),'linear_ramp', end_values=(0,0))
        estimates.append(audio_pad.T)

    return estimates


def inference_args(parser, remaining_args):
    inf_parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parser],
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    inf_parser.add_argument(
        "--softmask",
        dest="softmask",
        action="store_true",
        help=(
            "if enabled, will initialize separation with softmask."
            "otherwise, will use mixture phase with spectrogram"
        ),
    )

    inf_parser.add_argument(
        "--niter", type=int, default=1, help="number of iterations for refining results."
    )
    
    inf_parser.add_argument(
        "--num_workers", type=int, default=6, help="number of workers."
    )

    inf_parser.add_argument(
        "--alpha", type=float, default=1.0, help="exponent in case of softmask separation"
    )

    inf_parser.add_argument("--samplerate", type=int, default=44100, help="model samplerate")

    inf_parser.add_argument(
        "--residual_model", action="store_true", help="create a model for the residual"
    )
    return inf_parser.parse_args()


def eval_main(parser, args):
    no_cuda=args.no_cuda
    test_dataset = MS_21Dataloader.MS_21Dataset(
        split='test',
        subset=None,
        sources=args.sources,
        targets=args.sources,
        sample_rate=args.samplerate,
        segment=args.duration,
        root=args.train_dir,
    )
    
    # Randomly choose the indexes of sentences to save.
    save_idx = random.sample(range(len(test_dataset)),len(test_dataset))
    series_list = []

    model_path = os.path.join(args.outdir, "best_model.pth")

    model_path = os.path.abspath(model_path)

    if not (os.path.exists(model_path)):
        outdir = os.path.abspath("./results_using_pre-trained_ms21")
        model_path = "r-sawata/XUMX_ms21_music_separation"
    else:
        outdir = os.path.join(
            os.path.abspath(args.outdir),
            "EvaluateResults_musdb18_testdata",
        )
    Path(outdir).mkdir(exist_ok=True, parents=True)
    print("Evaluated results will be saved in:\n {}".format(outdir), file=sys.stderr)

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model, instruments = load_model(model_path, device)

    # test_dataset = musdb.DB(root=root, subsets="test", is_wav=True)
    # results = museval.EvalStore()
    Path(outdir).mkdir(exist_ok=True, parents=True)
    txtout = os.path.join(outdir, "results.txt")
    fp = open(txtout, "w")
    for idx in tqdm(range(len(test_dataset))):
        # Forward the network on the mixture.
        audio, ground_truths = test_dataset[idx]
        audio = audio.T.numpy()
        ground_truths = ground_truths.numpy().squeeze()
        # if 'ms21' in root:
        #     input_file = os.path.join(root, 'test', track, track+'_MIX.wav')
        #     # handling an input audio path
        #     info = sf.info(input_file)
        #     start = int(start * info.samplerate)
        #     # check if dur is none
        #     if duration > 0:
        #         # stop in soundfile is calc in samples, not seconds
        #         stop = start + int(duration * info.samplerate)
        #     else:
        #         # set to None for reading complete file
        #         stop = None

        #     audio, rate = sf.read(input_file, always_2d=True, start=start, stop=stop)
        #     source_names = ['bass', 'percussion', 'vocal', 'other']
        #     ground_truths = []
        #     for source_name in source_names:
        #         sc, sr = sf.read(os.path.join(root, 'test', track, track+'_STEMS', 'MUSDB',track+'_STEM_MUSDB_'+source_name+'.wav'), always_2d=False, start=start, stop=stop)
        #         ground_truths.append(sc)
            
        # else:
        #     input_file = os.path.join(root, "test", track, "mixture.wav")
        #     # handling an input audio path
        #     info = sf.info(input_file)
        #     start = int(start * info.samplerate)
        #     # check if dur is none
        #     if duration > 0:
        #         # stop in soundfile is calc in samples, not seconds
        #         stop = start + int(duration * info.samplerate)
        #     else:
        #         # set to None for reading complete file
        #         stop = None

        #     audio, rate = sf.read(input_file, always_2d=True, start=start, stop=stop)
        #     source_names = ['bass', 'drums', 'vocals', 'other']
        #     ground_truths = []
        #     for source_name in source_names:
        #         sc, sr = sf.read(os.path.join(root, 'test', track, track+'_STEMS', 'MUSDB',track+'_STEM_MUSDB_'+source_name+'.wav'), always_2d=False, start=start, stop=stop)
        #         ground_truths.append(sc)



        if audio.shape[1] > 2:
            warnings.warn("Channel count > 2! " "Only the first two channels will be processed!")
            audio = audio[:, :2]

        # if rate != samplerate:
        #     # resample to model samplerate if needed
        #     audio = resampy.resample(audio, rate, samplerate, axis=0)

        # if audio.shape[1] == 1:
        #     # if we have mono, let's duplicate it
        #     # as the input of OpenUnmix is always stereo
        #     audio = np.repeat(audio, 2, axis=1)
        if audio.shape[1] == 2:
            # if we have mono, let's duplicate it
            # as the input of OpenUnmix is always stereo
            audio = audio.sum(axis=1) / 2
            audio = np.expand_dims(audio, axis=1)

        estimates = separate(
            audio,
            model,
            instruments,
            niter=args.niter,
            alpha=args.alpha,
            softmask=args.softmask,
            residual_model=args.residual_model,
            device=device,
        )


        utt_metrics = get_metrics(audio.T, ground_truths, np.stack(estimates), sample_rate=16000, metrics_list=COMPUTE_METRICS, average=False)

        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(outdir, "ex_{}/".format(idx + 1))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(
                local_save_dir + "mixture.wav",
                audio,
                args.samplerate
            )
            # Loop over the sources and estimates
            for src_idx, src in enumerate(ground_truths):
                sf.write(
                    local_save_dir + "s{}.wav".format(src_idx),
                    src,
                    args.samplerate
                )
            for src_idx, est_src in enumerate(estimates):
                est_src *= np.max(np.abs(audio)) / np.max(np.abs(est_src))
                sf.write(
                    local_save_dir + "s{}_estimate.wav".format(src_idx),
                    est_src,
                    args.samplerate
                )

        # Write local metrics to the example folder.
        with open(local_save_dir + "metrics.json", "w") as f:
            json.dump({k:v.tolist() for k,v in utt_metrics.items()}, f, indent=0)


    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(outdir, "all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in COMPUTE_METRICS:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()

    print("Overall metrics :")
    print(final_results)

    with open(os.path.join(outdir, "final_metrics.json"), "w") as f:
        json.dump({k:v.tolist() for k,v in final_results.items()}, f, indent=0)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="OSU Inference", add_help=False)

    parser.add_argument("--train_dir", type=str, default='E:/ms21_DB', help="The path to the MUSDB18 dataset")

    parser.add_argument(
        "--outdir",
        type=str,
        default="./results_using_pre-trained",
        help="Results path where " "best_model.pth" " is stored",
    )

    parser.add_argument("--start", type=float, default=0.0, help="Audio chunk start in seconds")

    parser.add_argument(
        "--duration",
        type=float,
        default=-1.0,
        help="Audio chunk duration in seconds, negative values load full track",
    )

    parser.add_argument(
        "--sources",        
        type=list,
        default=["bass", "percussion", "vocal", "other"],
        help="Target Source Types",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA inference"
    )

    args, _ = parser.parse_known_args()
    args = inference_args(parser, args)

    model = os.path.join(args.outdir, "best_model.pth")
    eval_main(parser, args)
