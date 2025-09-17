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

import numpy as np

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

    with open(folder / "log.txt", "w") as f:
    
        print("\n" + "".join(["*"] * 10) + f" Sampling from DCA model " + "".join(["*"] * 10) + "\n")
        f.write("\n" + "".join(["*"] * 10) + f" Sampling from DCA model " + "".join(["*"] * 10) + "\n")
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
        f.write(f"Loading parameters from {args.path_params}...\n")
        params = load_params(fname=args.path_params, tokens=tokens, device=device, dtype=dtype)
        L, q = params["bias"].shape
        
        # Select the sampler
        sampler = get_sampler("gibbs_importance")
        
        # Import data
        print(f"Loading data from {args.data}...")
        f.write(f"Loading data from {args.data}...\n")
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
        f.write(f"Average energy of the data: {energies.mean():.3f} +/- {energies.std():.3f}\n")

        if args.pseudocount is None:
            args.pseudocount = 1. / weights.sum()
        print(f"Using pseudocount: {args.pseudocount:.3f}...")
        f.write(f"Using pseudocount: {args.pseudocount:.3f}...\n")
        f.flush()

        # Sample from random initialization
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
        
        gamma = 0
        gamma2 = 0
        # STEP 1: find a gamma such that at least 10% of the sequences are at ZERO distance from target
        print("\nSTEP 0: Finding gamma such that at least 10% of sequences are at zero distance from target1...\n")   
        f.write("\nSTEP 0: Finding gamma such that at least 10% of sequences are at zero distance from target1...\n")   
        f.flush()

        samples_0 = samples.clone()
        distance = (L - compute_seqID(samples, target_seq)).cpu().numpy()
        fraction = 0.0 #(distance == args.distance).sum() / args.ngen # fraction of sequences collapsed to the target

        delta_gamma = args.delta_gamma
        while fraction <= 0.1:
            gamma += delta_gamma
            samples = sampler(chains=samples, params=params, target_seq=target_seq, gamma=gamma, distance=0, nsweeps=args.steps,  beta=args.beta)
            distance = (L - compute_seqID(samples, target_seq)).cpu().numpy()
            fraction = (distance == 0).sum() / args.ngen
            if args.verbose:
                print(f"Gamma = {gamma:.3f}, fraction of sequences at distance 0 from target1: {fraction:.3f}")
            f.write(f"Gamma = {gamma:.3f}, fraction of sequences at distance 0 from target1: {fraction:.3f}\n")
            f.flush()

        print("\n" + 50 * "*"), print("S(target1) = 0.0"), print(50 * "*")
        f.write(50 * "*"), f.write("\nS(target1) = 0.0\n"), f.write(50 * "*" + "\n")
        f.flush()

        mean_energy = compute_energy(target_seq.unsqueeze(0), params=params) 
        lnZ = - np.log(fraction) - mean_energy

        fractions = torch.tensor([(distance == d).sum() / args.ngen for d in range(L)])
        idx = torch.where(fractions > 0.1)[0]
        max_d = idx.max().item() if len(idx) > 0 else None
        if args.verbose:
            print("Max distance with 10% coverage:", max_d)
        f.write(f"\nMax distance with 10% coverage: {max_d}\n")

        mean_energy = compute_energy(samples[distance == max_d], params=params).mean() + gamma * max_d
        S = lnZ + mean_energy + np.log((distance == max_d).sum() / args.ngen)
        print("\n" + 50 * "*")
        print(f"S(d1={max_d}) = {S.item():.3f}")
        print(50 * "*")

        f.write("\n" + 50 * "*")
        f.write(f"\nS(d1={max_d}) = {S.item():.3f}\n")
        f.write(50 * "*" + "\n")




        print(f"\nSTEP 1: Sampling with good coverage at distances in the range [0, {args.distance}] from target1\n")
        f.write(f"\nSTEP 1: Sampling with good coverage at distances in the range [0, {args.distance}] from target1\n")
        f.flush()

        while fractions[args.distance] < 0.1: 
            samples = samples_0.clone()
            gamma = 0
            d_to_reach = max_d
            if args.verbose:
                print(f"\nFinding gamma for distance {d_to_reach}")
            f.write(f"Finding gamma for distance {d_to_reach}\n")
            f.flush()

            while True: 
                gamma += delta_gamma
                samples = sampler(chains=samples, params=params, target_seq=target_seq, gamma=gamma, distance=d_to_reach, nsweeps=args.steps,  beta=args.beta)
                distance = (L - compute_seqID(samples, target_seq)).cpu().numpy()
                
                fractions = torch.tensor([(distance == d).sum() / args.ngen for d in range(L)])
                idx = torch.where(fractions > 0.1)[0]
                max_d, min_d = (idx.max().item(), idx.min().item()) if len(idx) > 0 else (None, None)
                if args.verbose:
                    print(f"Gamma: {gamma:.3f} | min_d: {min_d}, max_d: {max_d}")
                f.write(f"Gamma: {gamma:.3f} | min_d: {min_d}, max_d: {max_d}\n")
                f.flush()

                if (idx == d_to_reach).any().item():
                    if args.verbose:
                        print(f"d_to_reach={d_to_reach} has been reached. Distances with >10% coverage: {idx.numpy()}")
                    f.write(f"d_to_reach={d_to_reach} has been reached. Distances with >10% coverage: {idx.numpy()}\n")

                    energy_gamma = compute_energy(samples[distance == d_to_reach], params=params).mean()
                    lnZ = S - torch.log(fractions[d_to_reach]) - energy_gamma
                    d_used = args.distance if fractions[args.distance] > 0.1 else max_d
                    S = lnZ + compute_energy(samples[distance == d_used], params=params).mean() + gamma * np.abs(d_used - d_to_reach) + torch.log(fractions[d_used])
                    print("\n" + 50 * "*")
                    print(f"S(d1={d_used}) = {S.item():.3f}")
                    print(50 * "*")

                    f.write("\n" + 50 * "*")
                    f.write(f"\nS(d1={d_used}) = {S.item():.3f}\n")
                    f.write(50 * "*" + "\n")
                    f.flush()
                    break

            if (max_d <= d_to_reach and gamma == delta_gamma) or (len(idx) < 2):
                if max_d < d_to_reach: 
                    if args.verbose:
                        print(f"max_d < {args.distance}, decreasing delta_gamma...")
                    f.write(f"max_d < {args.distance}, decreasing delta_gamma...\n")
                    f.flush()

                elif len(idx) <= 3: 
                    if args.verbose:
                        print("len(idx) < 2, decreasing delta_gamma...")
                    f.write("len(idx) < 2, decreasing delta_gamma...\n")
                    f.flush()

                gamma, delta_gamma = 0, delta_gamma / 2
                S, max_d = lnZ + torch.log(fractions[d_to_reach]) + energy_gamma, d_to_reach
                if args.verbose:
                    print(f"Updated gamma={gamma:.3f}, delta_gamma={delta_gamma:.3f}, S={S.item():.3f}, target distance={d_to_reach}")
                f.write(f"Updated gamma={gamma:.3f}, delta_gamma={delta_gamma:.3f}, S={S.item():.3f}, target distance={d_to_reach}\n")
                f.flush()

        if args.verbose:
            print(f"Fraction of sequences at distance {args.distance} from target1: {fractions[args.distance].item()}")
        f.write(f"Fraction of sequences at distance {args.distance} from target1: {fractions[args.distance].item()}\n")
        f.flush()




        print(f"\nSTEP 2: Sampling with good coverage at distance {args.distance} from target1 and {args.distance2} from target2\n")
        f.write(f"\nSTEP 2: Sampling with good coverage at distance {args.distance} from target1 and {args.distance2} from target2\n")
        f.flush()

        fractions_d1, fractions_d2, fractions_d12 = fractions, torch.zeros(L), torch.zeros((L,L))
        gamma2, delta_gamma2 = 0, args.delta_gamma
        while fractions_d12[args.distance, args.distance2] < 0.1: 
            gamma2 += delta_gamma2
            if args.verbose:
                print(f"gamma2 = {gamma2}")
            f.write(f"gamma2 = {gamma2}\n")
            f.flush()

            samples = sampler(
                chains=samples, params=params, target_seq=target_seq, gamma=gamma, distance=args.distance,
                target_seq2=target_seq2, gamma2=gamma2, distance2=args.distance2, nsweeps=args.steps, beta=args.beta
            )
            distance1 = (L - compute_seqID(samples, target_seq)).cpu().numpy()
            distance2 = (L - compute_seqID(samples, target_seq2)).cpu().numpy()
            for d in range(1, L):
                fractions_d1[d] = (distance1 == d).sum() / args.ngen
                fractions_d2[d] = (distance2 == d).sum() / args.ngen
                for d2 in range(1, L):
                    fractions_d12[d, d2] = ((distance1 == d) & (distance2 == d2)).sum().item() / args.ngen
            idx1, idx2 = torch.where(fractions_d1 > 0.1)[0], torch.where(fractions_d2 > 0.1)[0]
            idx12_x, idx12_y = torch.where(fractions_d12 > 0.1)

            if args.verbose:
                print(f"Fraction of sequences at distance {args.distance} from target1: {fractions_d1[args.distance].item()}")
                print(f"Fraction of sequences at distance {args.distance2} from target2: {fractions_d2[args.distance2].item()}")
                print(f"Fraction of sequences with both distances: {fractions_d12[args.distance, args.distance2]}")
            f.write(f"Fraction of sequences at distance {args.distance} from target1: {fractions_d1[args.distance].item()}\n")
            f.write(f"Fraction of sequences at distance {args.distance2} from target2: {fractions_d2[args.distance2].item()}\n")
            f.write(f"Fraction of sequences with both distances: {fractions_d12[args.distance, args.distance2]}\n")
            f.flush()

            if idx1.numel() == 0:
                idx1 = torch.tensor([float("-inf")], device=idx1.device)
            while idx1.max().item() < args.distance or args.distance < idx1.min().item() :
                gamma += delta_gamma2
                if args.verbose:
                    print(f"Adjusting gamma: {gamma}")
                f.write(f"Adjusting gamma: {gamma}\n")
                f.flush()
                samples = sampler(
                    chains=samples, params=params, target_seq=target_seq, gamma=gamma, distance=args.distance,
                    target_seq2=target_seq2, gamma2=gamma2, distance2=args.distance2, nsweeps=args.steps, beta=args.beta
                )

                distance1, distance2 = (L - compute_seqID(samples, target_seq)).cpu().numpy(), (L - compute_seqID(samples, target_seq2)).cpu().numpy()
                fractions_d1 = torch.tensor([(distance1 == d).sum() / args.ngen for d in range(L)])
                fractions_d2 = torch.tensor([(distance2 == d).sum() / args.ngen for d in range(L)])
                idx1 = torch.where(fractions_d1 > 0.1)[0]

                if args.verbose:
                    print(f"Fractions at distance {args.distance} from target1: {fractions_d1[args.distance].item()}")
                    print(f"Fraction at distance {args.distance2} from target2: {fractions_d2[args.distance2].item()}")
                f.write(f"Fractions at distance {args.distance} from target1: {fractions_d1[args.distance].item()}\n")
                f.write(f"Fraction at distance {args.distance2} from target2: {fractions_d2[args.distance2].item()}\n")
                f.flush()

        mean_energy = np.mean(compute_energy(samples[(distance1 == args.distance), :, :], params=params).cpu().numpy() + gamma2 * distance2[(distance1 == args.distance)])
        lnZ = S - torch.log(fractions_d1[args.distance]) - mean_energy
        S_final = lnZ + torch.mean(compute_energy(samples[((distance1 == args.distance) & (distance2 == args.distance2)), :, :], params=params)) + torch.log(fractions_d12[args.distance, args.distance2])
        print("\n" + 50 * "*")
        print(f"S(d1={args.distance}, d2={args.distance2}) = {S_final.item()}")
        print(50 * "*")
        f.write("\n" + 50 * "*")
        f.write(f"\nS(d1={args.distance}, d2={args.distance2}) = {S_final.item()}\n")
        f.write(50 * "*" + "\n")
        f.flush()
        
    
if __name__ == "__main__":
    main()