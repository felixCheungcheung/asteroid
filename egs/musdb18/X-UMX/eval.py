import json
import random
import torch
import numpy as np
import argparse
import soundfile as sf
# import musdb
# musdb package is stem based, can not read hq
import museval
import norbert
from pathlib import Path
import scipy.signal
# import resampy
from asteroid.models import XUMX
from asteroid.complex_nn import torch_complex_from_magphase
# from asteroid.metrics import get_metrics
import os
import warnings
import sys
import pandas as pd
import itertools
from tqdm import tqdm
import MS_21Dataloader
# from memory_profiler import profile
from scipy.signal import stft, istft

os.environ['KMP_DUPLICATE_LIB_OK']='True'
COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar"]

def batch_to_device(batch, device):
    new_batch = []
    for b in batch:
        new_dict = {}
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                new_dict[k] = v.to(device=device)
            else:
                new_dict[k] = v
        new_batch.append(new_dict)
    return new_batch

def load_model(model_name, device="cpu"):
    print("Loading model from: {}".format(model_name), file=sys.stderr)
    model = XUMX.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, model.sources


def istft_local(X, rate=44100, n_fft=4096, n_hopsize=1024):
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

    estimates_eval_np = np.zeros_like(references)

    for i, sc_name in enumerate(args.sources):
        
        estimates_eval_np[i,:estimates[sc_name].shape[0]] = estimates[sc_name] # summing to mono for evaluation

    new_scores = new_sdr(references, estimates_eval_np)
    # new_scores = None

    if not compute_sdr:
        return None, new_scores
    else:
        # references = references.numpy()
        # estimates = estimates.numpy()
        scores = museval.metrics.bss_eval(
            references, estimates_eval_np,
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


def invert(M, eps):
    """"inverting matrices M (matrices are the two last dimensions).
    This is assuming that these are 2x2 matrices, using the explicit
    inversion formula available in that case."""
    invDet = 1.0/(eps + M[..., 0, 0]*M[..., 1, 1] - M[..., 0, 1]*M[..., 1, 0])
    invM = np.zeros(M.shape, dtype='complex')
    invM[..., 0, 0] = invDet*M[..., 1, 1]
    invM[..., 1, 0] = -invDet*M[..., 1, 0]
    invM[..., 0, 1] = -invDet*M[..., 0, 1]
    invM[..., 1, 1] = invDet*M[..., 0, 0]
    return invM

# Reference: https://github.com/sigsep/sigsep-mus-oracle/blob/master/MWF.py
def MWF(audio, ground_truths, eval_dir=None):
    """Multichannel Wiener Filter:
    processing all channels jointly with the ideal multichannel filter
    based on the local gaussian model, assuming time invariant spatial
    covariance matrix."""

    # to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # parameters for STFT
    nfft = 2048

    # compute STFT of Mixture
    N = audio.shape[0]  # remember number of samples for future use
    X = stft(audio.T, nperseg=nfft)[-1]
    (I, F, T) = X.shape

    # Allocate variables P: PSD, R: Spatial Covarianc Matrices
    P = {}
    R = {}
    for idx, name in enumerate(args.sources):

        # compute STFT of target source
        Yj = stft(ground_truths[idx].T, nperseg=nfft)[-1]

        # Learn Power Spectral Density and spatial covariance matrix
        # -----------------------------------------------------------

        # 1/ compute observed covariance for source
        Rjj = np.zeros((F, T, I, I), dtype='complex')
        for (i1, i2) in itertools.product(range(I), range(I)):
            Rjj[..., i1, i2] = Yj[i1, ...] * np.conj(Yj[i2, ...])

        # 2/ compute first naive estimate of the source spectrogram as the
        #    average of spectrogram over channels
        P[name] = np.mean(np.abs(Yj)**2, axis=0)

        # 3/ take the spatial covariance matrix as the average of
        #    the observed Rjj weighted Rjj by 1/Pj. This is because the
        #    covariance is modeled as Pj Rj
        R[name] = np.mean(Rjj / (eps+P[name][..., None, None]), axis=1)

        # add some regularization to this estimate: normalize and add small
        # identify matrix, so we are sure it behaves well numerically.
        R[name] = R[name] * I / np.trace(R[name]) + eps * np.tile(
            np.eye(I, dtype='complex64')[None, ...], (F, 1, 1)
        )

        # 4/ Now refine the power spectral density estimate. This is to better
        #    estimate the PSD in case the source has some correlations between
        #    channels.

        #    invert Rj
        Rj_inv = invert(R[name], eps)

        #    now compute the PSD
        P[name] = 0
        for (i1, i2) in itertools.product(range(I), range(I)):
            P[name] += 1./I*np.real(
                Rj_inv[:, i1, i2][:, None]*Rjj[..., i2, i1]
            )

    # All parameters are estimated. compute the mix covariance matrix as
    # the sum of the sources covariances.
    Cxx = 0
    for name, source in track.sources.items():
        Cxx += P[name][..., None, None]*R[name][:, None, ...]

    # we need its inverse for computing the Wiener filter
    invCxx = invert(Cxx, eps)

    # now separate sources
    estimates = {}
    accompaniment_source = 0
    for idx, name in enumerate(args.sources):
        # computes multichannel Wiener gain as Pj Rj invCxx
        G = np.zeros(invCxx.shape, dtype='complex64')
        SR = P[name][..., None, None]*R[name][:, None, ...]
        for (i1, i2, i3) in itertools.product(range(I), range(I), range(I)):
            G[..., i1, i2] += SR[..., i1, i3]*invCxx[..., i3, i2]
        SR = 0  # free memory

        # separates by (matrix-)multiplying this gain with the mix.
        Yj = 0
        for i in range(I):
            Yj += G[..., i]*X[i, ..., None]
        Yj = np.rollaxis(Yj, -1)  # gets channels back in first position

        # inverte to time domain
        target_estimate = istft(Yj)[1].T[:N, :]

        # set this as the source estimate
        estimates[name] = target_estimate

        # accumulate to the accompaniment if this is not vocals
    #     if name != 'vocals':
    #         accompaniment_source += target_estimate

    # estimates['accompaniment'] = accompaniment_source

    # if eval_dir is not None:
    #     museval.eval_mus_track(
    #         track,
    #         estimates,
    #         output_dir=eval_dir,
    #     )

    return estimates

# Reference: https://github.com/sigsep/sigsep-mus-oracle/blob/master/IRM.py
def IRM(audio, ground_truths, alpha=2, eval_dir=None):
    """Ideal Ratio Mask:
    processing all channels inpependently with the ideal ratio mask.
    this is the ratio of spectrograms, where alpha is the exponent to take for
    spectrograms. usual values are 1 (magnitude) and 2 (power)"""

    # STFT parameters
    nfft = 4096

    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # compute STFT of Mixture
    N = audio.shape[0]  # remember number of samples for future use
    X = stft(audio.T, nperseg=nfft)[-1]
    (I, F, T) = X.shape

    # Compute sources spectrograms
    P = {}
    # compute model as the sum of spectrograms
    model = eps
    for idx, src_name in enumerate(args.sources):
        # compute spectrogram of target source:
        # magnitude of STFT to the power alpha
        P[src_name] = np.abs(stft(ground_truths[idx].T, nperseg=nfft)[-1])**alpha
        model += P[src_name]

    # now performs separation
    estimates = {}
    accompaniment_source = 0
    for idx, src_name in enumerate(args.sources):
        # compute soft mask as the ratio between source spectrogram and total
        Mask = np.divide(np.abs(P[src_name]), model)

        # multiply the mix by the mask
        Yj = np.multiply(X, Mask)

        # invert to time domain
        target_estimate = istft(Yj)[1].T[:N, :]

        # set this as the source estimate
        estimates[src_name] = target_estimate

        # accumulate to the accompaniment if this is not vocals
        # if name != 'vocals':
        #     accompaniment_source += target_estimate

    # estimates['accompaniment'] = accompaniment_source

    # if eval_dir is not None:
    #     museval.eval_mus_track(
    #         track,
    #         estimates,
    #         output_dir=eval_dir,
    #     )

    return estimates

# @ profile
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

    del masked_tf_rep

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
    del V
    del X
    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = istft_local(
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
        "--num_workers", type=int, default=1, help="number of workers."
    )

    inf_parser.add_argument(
        "--alpha", type=float, default=1.0, help="exponent in case of softmask separation"
    )

    inf_parser.add_argument("--samplerate", type=int, default=16000, help="model samplerate")

    inf_parser.add_argument(
        "--residual_model", action="store_true", help="create a model for the residual"
    )
    return inf_parser.parse_args()

def read_estimate(local_save_dir, sources):
    estimates = {}
    for src in sources:
        audio, rate = sf.read(os.path.join(local_save_dir,src+'.wav'), always_2d=True) # ideal
        estimates[src] = audio
    return estimates


def eval_main(parser, args):
    
    no_cuda=args.no_cuda
    test_dataset = MS_21Dataloader.MS_21Dataset(
        split='test',
        subset=None,
        sources=args.sources,
        targets=args.sources,
        sample_rate=args.samplerate,
        samples_per_track=1,
        segment=args.duration,
        root=args.train_dir,
    )
    
    
    eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    # Randomly choose the indexes of sentences to save.
    save_idx = random.sample(range(len(test_dataset)),len(test_dataset))
    # save_idx = []

    # series_list = []

    if args.pretrained_type == 'ms21':
        model_path = os.path.join(args.outdir, 'best_model.pth')
    else:
        model_path = os.path.join(args.outdir, "pretrained_xumx_musdb18HQ.pth")
    
    model_path = os.path.abspath(model_path)
    print(model_path)
    if not (os.path.exists(model_path)):
        outdir = os.path.abspath("./results_using_pretrained_xumx_musdb18HQ")
        model_path = "r-sawata/XUMX_ms21_music_separation"
    else:
        outdir = os.path.join(
            os.path.abspath(args.outdir),
            f"EvaluateResults_ms21_backVox_new_testset",
        ) # change to "EvaluateResults_musdb18hq_testset" when doing evaluation on musdb18 test set
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
    # tracks = {}
    for idx, batch in tqdm(enumerate(eval_loader)):
        # Forward the network on the mixture.
        
        audio, ground_truths, track_name = batch
        track_name = track_name[0]
        print(track_name)

        local_save_dir = os.path.join(outdir, "{}/".format(track_name))
        if args.oracle:
            metric_dir = os.path.join(local_save_dir, "ideal_metrics_{}.json".format(track_name))
            if os.path.exists(metric_dir):
                print("Found ideal_metric.json, skipping ", track_name)
                continue
            else:
                print("Not found, ", metric_dir)
        else:
            metric_dir = os.path.join(local_save_dir, "metrics_{}.json".format(track_name))
            if os.path.exists(metric_dir):
                print("Found metric.json, skipping ", track_name)
                continue
            else:
                print("Not found, ", metric_dir)

        
        

        # audio, ground_truths, track_name = test_dataset[idx]
        
        
        
        audio = audio.T.numpy().squeeze()
        
        ground_truths = ground_truths.squeeze().permute(0,2,1)
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
        if args.oracle:
            estimates = IRM(audio, ground_truths)
        else:
            estimates = separate(
                audio,  
                model,
                instruments, # change to instruments when evaluation on musdb18 test set
                niter=args.niter,
                alpha=args.alpha,
                softmask=args.softmask,
                residual_model=args.residual_model,
                device=device,
            )
        
        
        # del estimates
        # get_metrics only accept mono for each source
        scores, n_sdr = eval_track(ground_truths, estimates, win=30*44100, hop=15*44100, compute_sdr=True)
        # Global SDR
        print(n_sdr)
        # Frame wise median SDR
        tracks = {}
        for idx, target in enumerate(model.sources):
            tracks[target] = {'nsdr': float(n_sdr[idx])}
        if scores is not None:
            (sdr, isr, sir, sar) = scores
            for idx, target in enumerate(model.sources):
                    values = {
                        "SDR": np.nanmedian(sdr[idx].tolist()),
                        "SIR": np.nanmedian(sir[idx].tolist()),
                        "ISR": np.nanmedian(isr[idx].tolist()),
                        "SAR": np.nanmedian(sar[idx].tolist())
                    }
                    tracks[target].update(values)
        
        
        # Save some examples in a folder. Wav files and metrics as text.
        
        os.makedirs(local_save_dir, exist_ok=True)
        # Write local metrics to the example folder.
        with open(metric_dir, "w") as f:
            json.dump({k:v for k,v in tracks.items()}, f, indent=0)
            
        if (idx in save_idx) and not args.oracle:
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
            for src_idx, est_src in estimates.items():
                est_src *= np.max(np.abs(audio)) / np.max(np.abs(est_src))
                sf.write(
                    local_save_dir + "{}_estimate.wav".format(src_idx),
                    est_src,
                    args.samplerate
                )

        


    write_final_res(outdir,args.oracle)

def write_final_res(output_dir, oracle):
    tracks = {}
    for track in os.listdir(output_dir):
        if not os.path.isdir(os.path.join(output_dir,track)):
            continue
        tracks[track] = {}
        if oracle:
            with open(os.path.join(output_dir,track, "ideal_metrics_{}.json".format(track)), "r") as f:
                tracks[track]  = json.load(f)
        else:
            with open(os.path.join(output_dir,track, "metrics_{}.json".format(track)), "r") as f:
                tracks[track]  = json.load(f)

    # Print and save summary metrics
    final_results = {}
    sources = ['bass', 'drums', 'vocals', 'other']
    metric_names = ["nsdr", "SDR", "SIR", "ISR", "SAR"]
    # metric_names = next(iter(tracks[track].values()))[sources[0]]
    for metric_name in metric_names:
        avg = 0
        avg_of_medians = 0
        for source in sources:
            medians = [
                tracks[track][source][metric_name]
                for track in tracks.keys()]
            mean = np.mean(medians)
            median = np.median(medians)
            final_results[metric_name.lower() + "_" + source] = mean
            final_results[metric_name.lower() + "_med" + "_" + source] = median
            avg += mean / len(sources)
            avg_of_medians += median / len(sources)
        final_results[metric_name.lower()] = avg
        final_results[metric_name.lower() + "_med"] = avg_of_medians

    print("Overall metrics :")
    print(final_results)
    if oracle:
        with open(os.path.join(output_dir, "ideal_final_metrics.json"), "w") as f:
            json.dump({k:v for k,v in final_results.items()}, f, indent=0)
    else:
        with open(os.path.join(output_dir, "final_metrics.json"), "w") as f:
            json.dump({k:v for k,v in final_results.items()}, f, indent=0)
        
if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="OSU Inference", add_help=False)

    parser.add_argument("--train_dir", type=str, default='E:/musdb-XL-audio', help="The path to the MUSDB18 dataset")

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
        # default=["bass", "drums", "vocals", "other"], # use this when evaluate on musdb18 test set
        default=["bass", "drums", "vocals", "other"], # use this when evaluate on MS21 test set
        help="Target Source Types",
    )
    
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA inference"
    )
    parser.add_argument(
        "--oracle", action="store_true", default=False, help="calculate oracle IRM for MS21 test set"
    )
    args, _ = parser.parse_known_args()
    args = inference_args(parser, args)

    
    eval_main(parser, args)
