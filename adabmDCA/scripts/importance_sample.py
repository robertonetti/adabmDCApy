import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch

from adabmDCA.fasta import (
    get_tokens,
    write_fasta,
)
from adabmDCA.resampling import compute_mixing_time, compute_seqID
from adabmDCA.io import load_params
from adabmDCA.fasta import import_from_fasta
from adabmDCA.dataset import DatasetDCA
from adabmDCA.utils import init_chains, resample_sequences, get_device, get_dtype
from adabmDCA.sampling import get_sampler
from adabmDCA.functional import one_hot
from adabmDCA.statmech import compute_energy
from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points
from adabmDCA.parser import add_args_importance_sample


# import command-line input arguments
def create_parser():
    # Important arguments
    parser = argparse.ArgumentParser(description='Samples from a DCA model.')
    parser = add_args_importance_sample(parser)
    
    return parser


def main():       
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Create output folder
    filename = Path(args.output)
    folder = filename.parent / Path(filename.name)
    # Create the folder where to save the samples
    folder.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "".join(["*"] * 10) + f" Sampling from DCA model " + "".join(["*"] * 10) + "\n")
    # Set the device
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    tokens = get_tokens(args.alphabet)
    
    # Check that the data file exists
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data file {args.data} not found.")
    
    # Check that the parameters file exists
    if not Path(args.path_params).exists():
        raise FileNotFoundError(f"Parameters file {args.path_params} not found.")
        
    # Import parameters
    print(f"Loading parameters from {args.path_params}...")
    params = load_params(fname=args.path_params, tokens=tokens, device=device, dtype=dtype)
    L, q = params["bias"].shape
    
    # Select the sampler
    sampler = get_sampler("gibbs_importance")
    
    # Import data
    print(f"Loading data from {args.data}...")
    dataset = DatasetDCA(
        path_data=args.data,
        path_weights=args.weights,
        alphabet=tokens,
        clustering_th=args.clustering_seqid,
        no_reweighting=args.no_reweighting,
        device=device,
        dtype=dtype,
    )
    data = one_hot(dataset.data, num_classes=len(tokens)).to(dtype)
    weights = dataset.weights
    
    energies = compute_energy(data, params=params).cpu().numpy()
    print(f"Average energy of the data: {energies.mean():.3f} +/- {energies.std():.3f}")

    if args.pseudocount is None:
        args.pseudocount = 1. / weights.sum()
    print(f"Using pseudocount: {args.pseudocount}...")
    
    
    # Sample from random initialization
    print(f"Sampling for {args.nsweeps} sweeps...")
    
    _, target_seq = import_from_fasta(args.target_path, tokens=tokens, filter_sequences=True)
    target_seq = one_hot(torch.tensor(target_seq), num_classes=len(tokens)).to(device=device, dtype=dtype).squeeze(0)

    if args.target_path2 is not None:
        _, target_seq2 = import_from_fasta(args.target_path2, tokens=tokens, filter_sequences=True)
        target_seq2 = one_hot(torch.tensor(target_seq2), num_classes=len(tokens)).to(device=device, dtype=dtype).squeeze(0)

    # Initialize the chains at random
    samples = init_chains(
        num_chains=args.ngen,
        L=L,
        q=q,
        device=device,
        dtype=dtype,
    )
    
    # Compute single and two-site frequencies of the data
    fi = get_freq_single_point(data=data, weights=weights, pseudo_count=args.pseudocount)
    fij = get_freq_two_points(data=data, weights=weights, pseudo_count=args.pseudocount)
    
    # Dictionary to store Pearson coefficient and slope along the sampling
    results_sampling = {
        "nsweeps" : [],
        "pearson" : [],
        "slope" : [],
    }
    
    step = 10
    # Sample for (args.nmix * mixing_time) sweeps starting from random initialization
    for i in range(args.nsweeps // step):
        # pbar.update(1)
        if args.target_path2 is not None:
            samples = sampler(chains=samples, params=params, target_seq=target_seq, gamma=args.gamma, distance=args.distance, target_seq2=target_seq2, gamma2=args.gamma2, distance2=args.distance2, nsweeps=step,  beta=args.beta)
            seqID_target = compute_seqID(samples, target_seq)  # (N,)
            seqID_target2 = compute_seqID(samples, target_seq2)  # (N,)
            print(f"\nAfter {step * (i+1)} sweeps \naverage distance from target1: {(L - seqID_target).mean().item():.3f} +/- {(L - seqID_target).std().item():.3f}")
            print(f"average distance from target2: {(L - seqID_target2).mean().item():.3f} +/- {(L - seqID_target2).std().item():.3f}")
        else:
            samples = sampler(chains=samples, params=params, target_seq=target_seq, gamma=args.gamma, nsweeps=step, distance=args.distance, beta=args.beta)
            seqID_target = compute_seqID(samples, target_seq)  # (N,)
            print(f"\nAfter {step * (i+1)} sweeps, average distance from target: {(L - seqID_target).mean().item():.3f} +/- {(L - seqID_target).std().item():.3f}")


        energies = compute_energy(samples, params=params).cpu().numpy()
        print(f"Average energy: {energies.mean():.3f} +/- {energies.std():.3f}")
        distance_right = (L - seqID_target) == args.distance
        print("percentage of sequences at the right distance from target1: ", 100. * distance_right.sum().item() / args.ngen)
        if args.target_path2 is not None:
            distance_right2 = (L - seqID_target2) == args.distance2
            print("percentage of sequences at the right distance from target2: ", 100. * distance_right2.sum().item() / args.ngen)
            distance_right_both = distance_right & distance_right2
            print("percentage of sequences at the right distance from both targets: ", 100. * distance_right_both.sum().item() / args.ngen)
        results_sampling["nsweeps"].append(i)

    
    # Compute the energy of the samples
    print("Computing the energy of the samples...")
    energies = compute_energy(samples, params=params).cpu().numpy()
    
    print("Saving the samples...")
    headers = [f"sequence {i+1} | DCAenergy: {energies[i]:.3f}" for i in range(args.ngen)]
    write_fasta(
        fname=folder / Path(f"{args.label}_samples.fasta"),
        headers=headers,
        sequences=samples.argmax(-1).cpu().numpy(),
        numeric_input=True,
        remove_gaps=False,
        tokens=tokens,
    )
    
    print("Writing sampling log...")
    df_mix_log = pd.DataFrame.from_dict(results_mix)    
    df_mix_log.to_csv(
        folder / Path(f"{args.label}_mix.log"),
        index=False
    )
    df_samp_log = pd.DataFrame.from_dict(results_sampling)    
    df_samp_log.to_csv(
        folder / Path(f"{args.label}_sampling.log"),
        index=False
    )
    
    print(f"Done, results saved in {str(folder)}")
    
    
if __name__ == "__main__":
    main()