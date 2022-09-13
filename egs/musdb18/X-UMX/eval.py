import json
import random
import torch
import numpy as np
import argparse
import soundfile as sf
import musdb
# musdb package is stem based, can not read hq
import museval
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


def new_sdr(references, estimates):
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    # assert references.dim() == 4
    # assert estimates.dim() == 4
    delta = 1e-7  # avoid numerical errors
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - estimates), axis=(1, 2))
    num += delta
    den += delta
    scores = 10 * np.log10(num / den)
    return scores

    
def eval_track(references, estimates, win, hop, compute_sdr=True):
    # references = references.transpose(1, 2).double()
    # estimates = estimates.transpose(1, 2).double()

    new_scores = new_sdr(references, estimates)
    # new_scores = None

    if not compute_sdr:
        return None, new_scores
    else:
        # references = references.numpy()
        # estimates = estimates.numpy()
        scores = museval.metrics.bss_eval(
            references, estimates,
            compute_permutation=False,
            window=win,
            hop=hop,
            framewise_filters=False,
            bsseval_sources_version=False)[:-1]
        return scores, new_scores

def estimate_and_evaluate(track):
    # assume mix as estimates
    estimates = {
        'vocals': track.audio,
        'accompaniment': track.audio
    }

    # Evaluate using museval
    scores = museval.eval_mus_track(
        track, estimates, output_dir="path/to/json"
    )

    # print nicely formatted and aggregated scores
    print(scores)

# def separate(
#     audio,
#     x_umx_target,
#     instruments,
#     niter=1,
#     softmask=False,
#     alpha=1.0,
#     residual_model=False,
#     device="cpu",
# ):
#     """
#     Performing the separation on audio input

#     Parameters
#     ----------
#     audio: np.ndarray [shape=(nb_samples, nb_channels, nb_timesteps)]
#         mixture audio

#     x_umx_target: asteroid.models
#         X-UMX model used for separating

#     instruments: list
#         The list of instruments, e.g., ["bass", "drums", "vocals"]

#     niter: int
#          Number of EM steps for refining initial estimates in a
#          post-processing stage, defaults to 1.

#     softmask: boolean
#         if activated, then the initial estimates for the sources will
#         be obtained through a ratio mask of the mixture STFT, and not
#         by using the default behavior of reconstructing waveforms
#         by using the mixture phase, defaults to False

#     alpha: float
#         changes the exponent to use for building ratio masks, defaults to 1.0

#     residual_model: boolean
#         computes a residual target, for custom separation scenarios
#         when not all targets are available, defaults to False

#     device: str
#         set torch device. Defaults to `cpu`.

#     Returns
#     -------
#     estimates: `dict` [`str`, `np.ndarray`]
#         dictionary with all estimates obtained by the separation model.
#     """

#     # convert numpy audio to torch
#     audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)
#     audio_len = audio_torch.shape[-1]

#     source_names = []
#     V = []

#     masked_tf_rep, _ = x_umx_target(audio_torch)
#     # shape: (Sources, frames, batch, channels, fbin)

#     for j, target in enumerate(instruments):
#         Vj = masked_tf_rep[j, Ellipsis].cpu().detach().numpy()
#         if softmask:
#             # only exponentiate the model if we use softmask
#             Vj = Vj ** alpha
#         # output is nb_frames, nb_samples, nb_channels, nb_bins
#         V.append(Vj[:, 0, Ellipsis])  # remove sample dim
#         source_names += [target]

#     V = np.transpose(np.array(V), (1, 3, 2, 0))

#     # convert to complex numpy type
#     tmp = x_umx_target.encoder(audio_torch)
#     X = torch_complex_from_magphase(tmp[0].permute(1, 2, 3, 0), tmp[1])
#     X = X.detach().cpu().numpy()
#     X = X[0].transpose(2, 1, 0)

#     if residual_model or len(instruments) == 1:
#         V = norbert.residual_model(V, X, alpha if softmask else 1)
#         source_names += ["residual"] if len(instruments) > 1 else ["accompaniment"]

#     Y = norbert.wiener(V, X.astype(np.complex128), niter, use_softmask=softmask)

#     estimates = []
#     for j, name in enumerate(source_names):
#         audio_hat = istft(
#             Y[..., j].T,
#             rate=x_umx_target.sample_rate,
#             n_fft=x_umx_target.in_chan,
#             n_hopsize=x_umx_target.n_hop,
#         )
#         pad_len = abs(audio_hat.shape[1] - audio_len)
#         if pad_len !=0:
#             audio_pad = np.pad(audio_hat.squeeze(), (0,pad_len),'linear_ramp', end_values=(0,0))
#         estimates.append(audio_pad.T)

#     return estimates


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

    source_names = []
    V = []

    masked_tf_rep, _ = x_umx_target(audio_torch)
    # shape: (Sources, frames, batch, channels, fbin)

    # check instrument names:
    

    for j, target in enumerate(instruments):
        Vj = masked_tf_rep[j, Ellipsis].cpu().detach().numpy()
        if softmask:
            # only exponentiate the model if we use softmask
            Vj = Vj**alpha
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

    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = istft(
            Y[..., j].T,
            rate=x_umx_target.sample_rate,
            n_fft=x_umx_target.in_chan,
            n_hopsize=x_umx_target.n_hop,
        )
        estimates[name] = audio_hat.T

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

    inf_parser.add_argument("--samplerate", type=int, default=16000, help="model samplerate")

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
    save_idx = random.sample(range(len(test_dataset)),5)

    series_list = []

    if args.pretrained_type == 'ms21':
        model_path = os.path.join(args.outdir, 'best_model.pth')
    else:
        model_path = os.path.join(args.outdir, "pretrained_xumx_musdb18HQ.pth")
    
    model_path = os.path.abspath(model_path)

    if not (os.path.exists(model_path)):
        outdir = os.path.abspath("./results_using_pretrained_xumx_musdb18HQ")
        model_path = "r-sawata/XUMX_ms21_music_separation"
    else:
        outdir = os.path.join(
            os.path.abspath(args.outdir),
            f"EvaluateResults_{args.pretrained_type}_testdata",
        )
    Path(outdir).mkdir(exist_ok=True, parents=True)
    print("Evaluated results will be saved in:\n {}".format(outdir), file=sys.stderr)

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model, instruments = load_model(model_path, device)
    
    ms21_sources = []
    for s in instruments:
        if s == 'drums':
            s = 'percussion'
        elif s == 'vocals':
            s = 'vocal'
        ms21_sources.append(s)
    # test_dataset = musdb.DB(root=root, subsets="test", is_wav=True)
    # results = museval.EvalStore()
    Path(outdir).mkdir(exist_ok=True, parents=True)
    # txtout = os.path.join(outdir, "results.txt")
    # fp = open(txtout, "w")
    tracks = {}
    for idx in tqdm(range(len(test_dataset))):
        # Forward the network on the mixture.
        audio, ground_truths, track_name = test_dataset[idx]
        print(track_name)
        audio = audio.T.numpy()
        
        ground_truths = ground_truths.permute(0,2,1)
        ground_truths = ground_truths.numpy()



        if audio.shape[1] > 2:
            warnings.warn("Channel count > 2! " "Only the first two channels will be processed!")
            audio = audio[:, :2]

        # if rate != samplerate:
        #     # resample to model samplerate if needed
        #     audio = resampy.resample(audio, rate, samplerate, axis=0)

        if audio.shape[1] == 1:
            # if we have mono, let's duplicate it
            # as the input of OpenUnmix is always stereo
            audio = np.repeat(audio, 2, axis=1)

        # model._return_time_signals = True
        estimates = separate(
            audio,  
            model,
            ms21_sources,
            niter=args.niter,
            alpha=args.alpha,
            softmask=args.softmask,
            residual_model=args.residual_model,
            device=device,
        )
        estimates_eval_np = np.zeros((len(args.sources), audio.shape[0], audio.shape[1]))

        # gt_eval_np = np.zeros((len(args.sources), audio.shape[0]))
        # gt_eval_np = ground_truths.sum(axis = 1)
        
        for i, sc_name in enumerate(args.sources):
            
            estimates_eval_np[i,:estimates[sc_name].shape[0]] = estimates[sc_name] # summing to mono for evaluation

        del estimates
        # get_metrics only accept mono for each source
        scores, n_sdr = eval_track(ground_truths, estimates_eval_np, win=30*44100, hop=15*44100, compute_sdr=True)
        # Global SDR
        print(n_sdr)
        # Frame wise median SDR
        tracks[track_name] = {}
        for idx, target in enumerate(model.sources):
            tracks[track_name][target] = {'nsdr': float(n_sdr[idx])}
        if scores is not None:
            (sdr, isr, sir, sar) = scores
            for idx, target in enumerate(model.sources):
                    values = {
                        "SDR": np.nanmedian(sdr[idx].tolist()),
                        "SIR": np.nanmedian(sir[idx].tolist()),
                        "ISR": np.nanmedian(isr[idx].tolist()),
                        "SAR": np.nanmedian(sar[idx].tolist())
                    }
                    tracks[track_name][target].update(values)
        
        # utt_metrics = get_metrics(audio.sum(axis=1), gt_eval_np, estimates_eval_np.sum(axis = 1), sample_rate=44100, metrics_list=COMPUTE_METRICS, average=False)

        # series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        # track.name should be retrieved
        if idx in save_idx:
            local_save_dir = os.path.join(outdir, "{}/".format(track_name))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(
                local_save_dir + "mixture.wav",
                audio,
                args.samplerate
            )
            # Loop over the sources and estimates
            for src_idx, src in enumerate(ground_truths):
                sf.write(
                    local_save_dir + "{}.wav".format(ms21_sources[src_idx]),
                    src,
                    args.samplerate
                )
            for src_idx, est_src in enumerate(estimates_eval_np):
                est_src *= np.max(np.abs(audio)) / np.max(np.abs(est_src))
                sf.write(
                    local_save_dir + "{}_estimate.wav".format(ms21_sources[src_idx]),
                    est_src,
                    args.samplerate
                )

        # Write local metrics to the example folder.
        with open(local_save_dir + "metrics_{}.json".format(track_name), "w") as f:
            json.dump({k:v for k,v in tracks[track_name].items()}, f, indent=0)


    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame.from_dict(tracks, orient='index')
    all_metrics_df.to_csv(os.path.join(outdir, "all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    metric_names = next(iter(tracks.values()))[model.sources[0]]
    for metric_name in metric_names:
        avg = 0
        avg_of_medians = 0
        for source in model.sources:
            medians = [
                np.nanmedian(tracks[track][source][metric_name])
                for track in tracks.keys()]
            mean = np.mean(medians)
            median = np.median(medians)
            final_results[metric_name.lower() + "_" + source] = mean
            final_results[metric_name.lower() + "_med" + "_" + source] = median
            avg += mean / len(model.sources)
            avg_of_medians += median / len(model.sources)
        final_results[metric_name.lower()] = avg
        final_results[metric_name.lower() + "_med"] = avg_of_medians

    print("Overall metrics :")
    print(final_results)

    with open(os.path.join(outdir, "final_metrics.json"), "w") as f:
        json.dump({k:v for k,v in final_results.items()}, f, indent=0)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="OSU Inference", add_help=False)

    parser.add_argument("--train_dir", type=str, default='E:/ms21hq_DB_temp', help="The path to the MUSDB18 dataset")

    parser.add_argument(
        "--outdir",
        type=str,
        default="./results_using_pre-trained",
        help="Results path where " "best_model.pth" " is stored",
    )

    parser.add_argument("--pretrained_type", type=str, default='musdb18', help="The default pretrained model trained on MUSDB18 dataset")

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


    eval_main(parser, args)
