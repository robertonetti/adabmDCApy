import argparse
import copy
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.functional import one_hot as one_hot_torch

# adabmDCA modules
from adabmDCA.fasta import (
    get_tokens,
    import_from_fasta,
    compute_weights,
    write_fasta,
)
from adabmDCA.resampling import (
    compute_mixing_time,
    compute_seqID,
    get_mean_seqid,
)
from adabmDCA.io import load_params, load_chains
from adabmDCA.dataset import DatasetDCA
from adabmDCA.utils import (
    init_parameters,
    init_chains,
    resample_sequences,
    get_device,
    get_dtype,
)
from adabmDCA.sampling import get_sampler, gibbs_sampling_importance
from adabmDCA.functional import one_hot
from adabmDCA.statmech import compute_energy
from adabmDCA.stats import (
    get_freq_single_point,
    get_freq_two_points,
    get_correlation_two_points,
)
from adabmDCA.models.bmDCA import fit
from adabmDCA.checkpoint import LinearCheckpoint
from adabmDCA.parser import add_args_entropy_closest_natural

def _get_deltaE(
        idx: int,
        chain: torch.Tensor,
        residue_old: torch.Tensor,
        residue_new: torch.Tensor,
        params: Dict[str, torch.Tensor],
        L: int,
        q: int,
    ) -> float:
    
        coupling_residue = chain.view(-1, L * q) @ params["coupling_matrix"][:, :, idx, :].view(L * q, q) # (N, q)
        E_old = - residue_old @ params["bias"][idx] - torch.vmap(torch.dot, in_dims=(0, 0))(coupling_residue, residue_old)
        E_new = - residue_new @ params["bias"][idx] - torch.vmap(torch.dot, in_dims=(0, 0))(coupling_residue, residue_new)
        
        return E_new - E_old


def _metropolis_sweep_target_seqs_fast(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    target_seqs: torch.Tensor,
    theta: float,
    distance: int,
    beta: float,
) -> torch.Tensor:
    N, L, q = chains.shape
    residue_idxs = torch.randperm(L, device=chains.device)
    distance_old = compute_min_distance(chains, target_seqs)  # (N,)

    for i in residue_idxs:
        res_old = chains[:, i, :]  # (N, q)
        res_new = one_hot_torch(
            torch.randint(0, q, (N,), device=chains.device), num_classes=q
        ).float()

        # calcola distance_new senza modificare chains
        chains_i_old = chains[:, i, :].clone()
        chains[:, i, :] = res_new
        distance_new = compute_min_distance(chains, target_seqs)
        chains[:, i, :] = chains_i_old  # ripristina subito

        delta_seqID = torch.abs(distance_new - distance) - torch.abs(distance_old - distance)
        delta_E = _get_deltaE(i, chains, res_old, res_new, params, L, q)

        accept_prob = torch.exp(-beta * delta_E - theta * delta_seqID).unsqueeze(-1)
        r = torch.rand((N, 1), device=chains.device, dtype=chains.dtype)

        chains[:, i, :] = torch.where(accept_prob > r, res_new, res_old)
        distance_old = torch.where(accept_prob.squeeze(-1) > r.squeeze(-1), distance_new, distance_old)

    return chains

def _metropolis_sweep_target_seqs_fast1(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    target_seqs: torch.Tensor,
    theta: float,
    distance: int, 
    beta: float,
) -> torch.Tensor:
  
    N, L, q = chains.shape
    residue_idxs = torch.randperm(L)
    distance_old = compute_min_distance(chains, target_seqs) # (N)

    for i in residue_idxs:        
        res_old = chains[:, i, :]
        res_new = one_hot_torch(torch.randint(0, q, (N,), device=chains.device), num_classes=q).float()

        chains_sub = chains.clone()
        chains_sub[:, i, :] = res_new
        
        distance_new = compute_min_distance(chains_sub, target_seqs) # (N, M) distance_new = compute_min_distance(chains, target_seqs) #
        delta_seqID = torch.abs(distance_new - distance) - torch.abs(distance_old - distance)

        delta_E = _get_deltaE(i, chains, res_old, res_new, params, L, q) #delta_E = _get_deltaE(i, chains, res_old, res_new, params, L, q)
        accept_prob = torch.exp(- beta * delta_E - theta * delta_seqID).unsqueeze(-1)
        r = torch.rand((N, 1), device=chains.device, dtype=chains.dtype)
        chains[:, i, :] = torch.where(accept_prob > r, res_new, res_old)
        distance_old = torch.where(accept_prob.squeeze(-1) > r.squeeze(-1), distance_new, distance_old)
    return chains


def metropolis_target_seqs_fast(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    target_seqs: torch.Tensor,
    theta: float,
    distance: int, 
    nsweeps: int,
    beta: float = 1.0,
) -> torch.Tensor:
    for _ in range(nsweeps):
        chains = _metropolis_sweep_target_seqs_fast(chains, params, target_seqs, theta, distance, beta)
    return chains


@torch.jit.script
def compute_min_distance(a1: torch.Tensor, a2: torch.Tensor, chunk_n: int = 100000) -> torch.Tensor:
    """
    Min Hamming distance tra ogni seq di a1 e tutte le seq di a2.
    a1: (N, L, C) one-hot
    a2: (M, L, C) one-hot
    Ritorna: (N,) distanza minima int32

    - Fa chunk solo su N (le sequenze di input).
    - Tutte le M sequenze di a2 vengono considerate in blocco unico.
    - Evita il broadcast 3D: usa prodotto scalare flattenato.
    """
    N, L, C = a1.shape
    M = a2.size(0)

    # Flatten su L*C
    a1f = a1.to(torch.float32).reshape(N, L * C).contiguous()
    a2f_t = a2.to(torch.float32).reshape(M, L * C).transpose(0, 1).contiguous()  # (L*C, M)

    # Inizializza a distanza massima (L)
    min_dist = torch.full((N,), L, device=a1.device, dtype=torch.int32)

    n = 0
    while n < N:
        n_end = min(n + chunk_n, N)

        # matches: (n_chunk, M), ogni entry = numero match
        matches = torch.matmul(a1f[n:n_end], a2f_t)  # float32
        dblock = (L - matches).to(torch.int32)       # Hamming = L - #match

        block_min = dblock.min(dim=1).values         # (n_chunk,)
        min_dist[n:n_end] = block_min

        n = n_end

    return min_dist


# import command-line input arguments
def create_parser():
    # Important arguments
    parser = argparse.ArgumentParser(description='Samples from a DCA model.')
    parser = add_args_entropy_closest_natural(parser)
    return parser


def main():
    """
    Executes the forward entropy sampling routine for a DCA model.
    It loads parameters and natural data, samples sequences at increasing
    distances toward natural configurations, and estimates entropy and free
    energy contributions as a function of distance from the equilibrium ensemble.
    """
    
    # Parse command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Create output folder
    filename = Path(args.output)
    folder = filename.parent / Path(filename.name)
    folder.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    # Open log file
    with open(folder / "log.txt", "w") as f:
        print("\n" + "".join(["*"] * 10) + f" Sampling from DCA model " + "".join(["*"] * 10) + "\n")
        f.write("\n" + "".join(["*"] * 10) + f" Sampling from DCA model " + "".join(["*"] * 10) + "\n")
        
        # Setup computation device, precision, and alphabet tokens
        device = get_device(args.device)
        dtype = get_dtype(args.dtype)
        tokens = get_tokens(args.alphabet)
        
        # Validate input files
        if not Path(args.data).exists():
            raise FileNotFoundError(f"Data file {args.data} not found.")
        if not Path(args.path_params).exists():
            raise FileNotFoundError(f"Parameters file {args.path_params} not found.")
            
        # Load DCA parameters
        print(f"Loading parameters from {args.path_params}...")
        f.write(f"Loading parameters from {args.path_params}...\n")
        params = load_params(fname=args.path_params, tokens=tokens, device=device, dtype=dtype)
        L, q = params["bias"].shape
        
        # Select the sampler
        sampler = get_sampler("gibbs_importance")
        
        # Load dataset and weights
        print(f"Loading data from {args.data}...")
        f.write(f"Loading data from {args.data}...\n")
        dataset = DatasetDCA(
            path_data=args.data,
            path_weights=args.weights,
            alphabet=tokens,
            clustering_th=args.clustering_seqid,
            no_reweighting=True,
            device=device,
            dtype=dtype,
        )
        data = one_hot(dataset.data, num_classes=len(tokens)).to(dtype)
        weights = dataset.weights
        
        # Compute average energy of natural data
        energies = compute_energy(data, params=params).cpu().numpy()
        print(f"Average energy of the data: {energies.mean():.3f} +/- {energies.std():.3f}")
        f.write(f"Average energy of the data: {energies.mean():.3f} +/- {energies.std():.3f}\n")

        # Use default pseudocount if not provided
        if args.pseudocount is None:
            args.pseudocount = 1. / weights.sum()
        print(f"Using pseudocount: {args.pseudocount:.3f}...")
        f.write(f"Using pseudocount: {args.pseudocount:.3f}...\n")
        f.flush()

        # STEP 0: initialize or import reference samples (theta=0)
        if args.target_path is not None: 
            _, samples_0 = import_from_fasta(args.target_path, tokens=tokens, filter_sequences=True)
            samples_0 = one_hot(torch.tensor(samples_0), num_classes=len(tokens)).to(device=device, dtype=dtype).squeeze(0)[:args.n0, :, :]
        else: 
            print("\nSTEP 0: sample until equilibrium with gamma=0...\n")   
            f.write("\nSTEP 0: sample until equilibrium with gamma=0...\n"), f.flush()
            sampler_0 = get_sampler("gibbs")
            samples_0 = init_chains(num_chains=args.n0, L=L, q=q, device=device, dtype=dtype)
            samples_0 = sampler_0(chains=samples_0, params=params, nsweeps=10_000,  beta=args.beta)

        # Initialize new chains for subsequent sampling
        samples = init_chains(
            num_chains=args.ngen,
            L=L,
            q=q,
            device=device,
            dtype=dtype)
        
        # Display base entropy value
        print("\n" + 50 * "*")
        print(f"S(model) = {args.entropy}")
        print(50 * "*")
        f.write(50 * "*")
        f.write(f"\nS(model) = {args.entropy}\n")
        f.write(50 * "*" + "\n")
        f.flush()
        
        # Compute sequence distances between reference and natural data
        distance = compute_min_distance(samples_0, data) 
        hist = torch.bincount(distance.to(torch.int64), minlength=L+1) / args.n0
        for i, d in enumerate(hist):
            if args.verbose:
                print(f"Percentage of sequences with distance {i}: {d.item() * 100:.2f}%\n")
            f.write(f"Percentage of sequences with distance {i}: {d.item() * 100:.2f}%\n")
        f.flush()

        # Identify maximum distance bin with sufficient coverage
        idx = torch.where(hist >= 0.03)[0].cpu()
        max_distance = idx.min().item()
        fraction_max_d = hist[max_distance] 

        # Compute energies and normalization term
        energies_all = compute_energy(samples_0, params=params)
        mean_energy_all = energies_all.mean().item()
        std_energy_all = energies_all.std(unbiased=False).item()
        lnZ = args.entropy - mean_energy_all
        
        # Evaluate energy and probability at max_distance
        energies = compute_energy(samples_0[distance == max_distance], params=params)
        mean_energy = energies.mean().item()
        std_energy = energies.std(unbiased=False).item()
        logP = torch.log(fraction_max_d).item()

        if args.verbose:
            print(f"Distance bins with >3% coverage: {idx.numpy()}%\n")
            f.write(f"Distance bins with >3% coverage: {idx.numpy()}%\n")
            print(f"Mean Energy All : {mean_energy_all:.3f} +/- {std_energy_all:.3f}")
            print(f"lnZ All : {lnZ:.3f}")
            f.write(f"Mean Energy All : {mean_energy_all:.3f} +/- {std_energy_all:.3f}\n")
            f.write(f"lnZ All : {lnZ:.3f}\n")
            print(f"Mean Energy d={max_distance} : {mean_energy:.3f} +/- {std_energy:.3f}")
            print(f"logP d={max_distance} : {logP:.3f}")
            f.write(f"Mean Energy d={max_distance} : {mean_energy:.3f} +/- {std_energy:.3f}\n")
            f.write(f"logP d={max_distance} : {logP:.3f}\n"), f.flush()

        # Initialize entropy containers
        Entropies = {}
        entropies = []
        seqID_closest = []

        S = lnZ + mean_energy + logP
        Entropies[max_distance] = S
        entropies.append(S)
        seqID_closest.append(max_distance)
        
        # Display entropy at max_distance
        print("\n" + 50 * "*")
        print(f"S(d={max_distance}) = {S:.3f}")
        print(50 * "*")
        f.write("\n" + 50 * "*")
        f.write(f"\nS(d={max_distance}) = {S:.3f}\n")
        f.write(50 * "*" + "\n"), f.flush()

        # STEP 1: iterative sampling toward natural sequences
        print(f"\nSTEP 1: Sampling with good coverage at distances in the range [0, {max_distance}] from closest natural\n")
        f.write(f"\nSTEP 1: Sampling with good coverage at distances in the range [0, {max_distance}] from closest natural\n")
        f.flush()

        sampler = metropolis_target_seqs_fast
        delta_theta = args.delta_theta

        # Main backward loop (from max_distance toward 0)
        for d in range(max_distance - 1, -1, -1):
            print(f"\nsampling with good coverage at distance {d} and {d+1}...")
            f.write(f"\nsampling with good coverage at distance {d} and {d+1}...\n")

            theta = 0

            # Adjust theta dynamically until coverage threshold is met
            while True:
                theta += delta_theta 
                samples = sampler(chains=samples, params=params, target_seqs=data, theta=theta, distance=d, nsweeps=args.steps)
                distance = compute_min_distance(samples, data)
                hist = torch.bincount(distance.to(torch.int64), minlength=L+1) / args.ngen

                if args.verbose:
                    print(f"theta={theta}")
                    print(f"coverage at distance {d} and {d+1}: {hist[d]:.3f}, {hist[d+1]:.3f}")
                    print(f"d={torch.where(hist > args.threshold)[0].cpu().numpy()}, coverage={hist[hist>args.threshold].cpu().numpy()}")
                    f.write(f"theta={theta}\n")
                    f.write(f"coverage at distance {d} and {d+1}: {hist[d]:.3f}, {hist[d+1]:.3f}\n")
                    f.write(f"d={torch.where(hist > args.threshold)[0].cpu().numpy()}, coverage={hist[hist>args.threshold].cpu().numpy()}\n"), f.flush()

                if hist[d] >= args.threshold and hist[d+1] >= args.threshold:
                    delta_theta = args.delta_theta
                    break
                if 1 < torch.where(hist > args.threshold)[0].shape[0] <= 2 and hist[d] > 0.5: 
                    theta = 0 
                    delta_theta = delta_theta / 2
                    if args.verbose:
                        print(f"reducing delta_theta to : {delta_theta}")
                        f.write(f"reducing delta_theta to : {delta_theta}\n")

            # Compute entropy transition between d+1 and d
            energies = compute_energy(samples[distance == (d+1)], params=params) + theta
            lnZ = S - energies.mean().item() - torch.log(hist[d+1]).item()
            energies = compute_energy(samples[distance == d], params=params)
            mean_energy = energies.mean().item()
            std_energy = energies.std(unbiased=False).item()
            logP = torch.log(hist[d]).item()
            S = lnZ + mean_energy + logP

            entropies.append(S)
            seqID_closest.append(d)

            if args.verbose:
                print(f"lnZ (d={d+1}) : {lnZ:.3f}")
                print(f"Mean Energy d={d} : {mean_energy:.3f} +/- {std_energy:.3f}")
                print(f"logP (d={d}) : {logP:.3f}")
                f.write(f"lnZ (d={d+1}) : {lnZ:.3f}\n")
                f.write(f"Mean Energy (d={d}) : {mean_energy:.3f} +/- {std_energy:.3f}\n")
                f.write(f"logP (d={d}) : {logP:.3f}\n"), f.flush()

            Entropies[hist[d]] = S
            print("\n" + 50 * "*")
            print(f"S(d={d}) = {S:.3f}")
            print(50 * "*")
            f.write("\n" + 50 * "*")
            f.write(f"\nS(d={d}) = {S:.3f}\n")
            f.write(50 * "*" + "\n"), f.flush()

        # Save summary of results
        f.write(f"\nS(models) = {args.entropy:.3f}\n")
        f.write("\nS(d) = " + ", ".join(f"{x:.3f}" for x in entropies) + "\n")
        f.write("\nseqID closest = " + ", ".join(f"{x:.3f}" for x in seqID_closest) + "\n")
        f.write(50 * "*" + "\n"), f.flush()


def main_inverse():
    """
    Runs the inverse sampling routine for a DCA model.
    It loads model parameters and data, computes baseline entropy, 
    and performs progressive Metropolis sampling at increasing distances 
    from natural sequences to estimate entropy as a function of sequence divergence.
    """
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Create output folder
    filename = Path(args.output)
    folder = filename.parent / Path(filename.name)
    folder.mkdir(parents=True, exist_ok=True)  # Create the folder where to save the samples

    # Open log file
    with open(folder / "log.txt", "w") as f:
        print("\n" + "".join(["*"] * 10) + f" Sampling from DCA model " + "".join(["*"] * 10) + "\n")
        f.write("\n" + "".join(["*"] * 10) + f" Sampling from DCA model " + "".join(["*"] * 10) + "\n")
        
        # Setup device, dtype, and alphabet tokens
        device = get_device(args.device)
        dtype = get_dtype(args.dtype)
        tokens = get_tokens(args.alphabet)
        
        # Check input files
        if not Path(args.data).exists():
            raise FileNotFoundError(f"Data file {args.data} not found.")
        if not Path(args.path_params).exists():
            raise FileNotFoundError(f"Parameters file {args.path_params} not found.")
            
        # Load parameters
        print(f"Loading parameters from {args.path_params}...")
        f.write(f"Loading parameters from {args.path_params}...\n")
        params = load_params(fname=args.path_params, tokens=tokens, device=device, dtype=dtype)
        L, q = params["bias"].shape
        
        # Select the sampler
        sampler = get_sampler("gibbs_importance")
        
        # Load dataset
        print(f"Loading data from {args.data}...")
        f.write(f"Loading data from {args.data}...\n")
        dataset = DatasetDCA(
            path_data=args.data,
            path_weights=args.weights,
            alphabet=tokens,
            clustering_th=args.clustering_seqid,
            no_reweighting=True,
            device=device,
            dtype=dtype,
        )
        data = one_hot(dataset.data, num_classes=len(tokens)).to(dtype)
        weights = dataset.weights
        
        # Compute initial energy statistics
        energies = compute_energy(data, params=params).cpu().numpy()
        print(f"Average energy of the data: {energies.mean():.3f} +/- {energies.std():.3f}")
        f.write(f"Average energy of the data: {energies.mean():.3f} +/- {energies.std():.3f}\n")

        # Compute pseudocount if not provided
        if args.pseudocount is None:
            args.pseudocount = 1. / weights.sum()
        print(f"Using pseudocount: {args.pseudocount:.3f}...")
        f.write(f"Using pseudocount: {args.pseudocount:.3f}...\n")
        f.flush()

        # STEP 0: compute entropy of the natural data
        print("\nSTEP 0: compute entropy of the data...\n")   
        f.write("\nSTEP 0: compute entropy of the data...\n"), f.flush()
        energy_data = compute_energy(data, params=params)
        p_data = energy_data / torch.sum(energy_data)
        entropy_data = - torch.sum(p_data * torch.log(p_data)).cpu().numpy()

        print("\n" + 50 * "*")
        print(f"S(data) = {entropy_data}")
        print(50 * "*")
        f.write(50 * "*")
        f.write(f"\nS(data) = {entropy_data}\n")
        f.write(50 * "*" + "\n")
        f.flush()

        # Initialize containers for entropies
        Entropies = {}
        entropies = []
        seqID_closest = []

        Entropies[0] = entropy_data
        entropies.append(entropy_data)
        seqID_closest.append(0)
        
        # STEP 1: progressive sampling across distances
        print(f"\nSTEP 1: Sampling with good coverage at distances in the range [0, {L}] from closest natural\n")
        f.write(f"\nSTEP 1: Sampling with good coverage at distances in the range [0, {L}] from closest natural\n")
        f.flush()

        S = entropy_data
        sampler = metropolis_target_seqs_fast
        delta_theta = args.delta_theta

        # Main sampling loop
        for d in range(0, L):
            samples = init_chains(num_chains=args.ngen, L=L, q=q, device=device, dtype=dtype)
            print(f"\nsampling with good coverage at distance {d} and {d+1}...")
            f.write(f"\nsampling with good coverage at distance {d} and {d+1}...\n")

            theta = 0

            # Adjust theta until coverage at distance d and d+1 is reached
            while True:
                theta += delta_theta 
                samples = sampler(chains=samples, params=params, target_seqs=data, theta=theta, distance=d, nsweeps=args.steps)
                distance = compute_min_distance(samples, data)
                hist = torch.bincount(distance.to(torch.int64), minlength=L+1) / args.ngen

                if args.verbose:
                    print(f"theta={theta}")
                    print(f"coverage at distance {d} and {d+1}: {hist[d]:.3f}, {hist[d+1]:.3f}")
                    print(f"d={torch.where(hist > args.threshold)[0].cpu().numpy()}, coverage={hist[hist>args.threshold].cpu().numpy()}")
                    f.write(f"theta={theta}\n")
                    f.write(f"coverage at distance {d} and {d+1}: {hist[d]:.3f}, {hist[d+1]:.3f}\n")
                    f.write(f"d={torch.where(hist > args.threshold)[0].cpu().numpy()}, coverage={hist[hist>args.threshold].cpu().numpy()}\n"), f.flush()

                if hist[d] >= args.threshold and hist[d+1] >= args.threshold:
                    delta_theta = args.delta_theta
                    break

                if 1 <= torch.where(hist > args.threshold)[0].shape[0] <= 2 and hist[d] > 0.5: 
                    theta = 0 
                    delta_theta = delta_theta / 2
                    if args.verbose:
                        print(f"reducing delta_theta to : {delta_theta}")
                        f.write(f"reducing delta_theta to : {delta_theta}\n")

            # Compute mean energy and entropy contribution for distance d
            energies = compute_energy(samples[distance == d], params=params)
            mean_energy = energies.mean().item()
            std_energy = energies.std(unbiased=False).item()
            logP = torch.log(hist[d]).item()
            lnZ = S - mean_energy - logP

            if args.verbose:
                print(f"lnZ (d={d}, theta={theta}) : {lnZ:.3f}")
                print(f"Mean Energy d={d} : {mean_energy:.3f} +/- {std_energy:.3f}")
                print(f"logP (d={d}) : {logP:.3f}")
                f.write(f"lnZ (d={d}, theta={theta}) : {lnZ:.3f}\n")
                f.write(f"Mean Energy (d={d}) : {mean_energy:.3f} +/- {std_energy:.3f}\n")
                f.write(f"logP (d={d}) : {logP:.3f}\n"),  f.flush()

            # Update entropy estimate for d+1
            energies = compute_energy(samples[distance == d+1], params=params) + theta
            mean_energy = energies.mean().item()
            std_energy = energies.std(unbiased=False).item()
            logP = torch.log(hist[d+1]).item()
            S = lnZ + mean_energy + logP

            entropies.append(S)
            seqID_closest.append(d)

            if args.verbose:
                print(f"\nMean Energy d={d+1} : {mean_energy:.3f} +/- {std_energy:.3f}")
                print(f"logP (d={d+1}) : {logP:.3f}")
                f.write(f"\nMean Energy (d={d+1}) : {mean_energy:.3f} +/- {std_energy:.3f}\n")
                f.write(f"logP (d={d+1}) : {logP:.3f}\n"),  f.flush()

            Entropies[hist[d+1]] = S
            print("\n" + 50 * "*")
            print(f"S(d={d+1}) = {S:.3f}")
            print(50 * "*")
            f.write("\n" + 50 * "*")
            f.write(f"\nS(d={d+1}) = {S:.3f}\n")
            f.write(50 * "*" + "\n"), f.flush()

        # Save summary of results
        f.write("\nS(d) = " + ", ".join(f"{x:.3f}" for x in entropies) + "\n")
        f.write("\nseqID closest = " + ", ".join(f"{x:.3f}" for x in seqID_closest) + "\n")
        f.write(50 * "*" + "\n"), f.flush()




def main_idependent_new():       
    # ----------------------------
    # Parse command-line arguments
    # ----------------------------
    parser = create_parser()
    args = parser.parse_args()
    
    # ----------------------------
    # Create output folder
    # ----------------------------
    filename = Path(args.output)
    folder = filename.parent / Path(filename.name)
    folder.mkdir(parents=True, exist_ok=True)

    with open(folder / "log.txt", "w") as f:
        print("\n" + "".join(["*"] * 10) + f" Sampling from DCA model " + "".join(["*"] * 10) + "\n")
        f.write("\n" + "".join(["*"] * 10) + f" Sampling from DCA model " + "".join(["*"] * 10) + "\n")
        
        # ----------------------------
        # Setup device, dtype, tokens
        # ----------------------------
        device = get_device(args.device)
        dtype = get_dtype(args.dtype)
        tokens = get_tokens(args.alphabet)
        
        # ----------------------------
        # Check data and parameter files
        # ----------------------------
        if not Path(args.data).exists():
            raise FileNotFoundError(f"Data file {args.data} not found.")
        
        if not Path(args.path_params).exists():
            raise FileNotFoundError(f"Parameters file {args.path_params} not found.")
            
        # ----------------------------
        # Load model parameters
        # ----------------------------
        print(f"Loading parameters from {args.path_params}...")
        f.write(f"Loading parameters from {args.path_params}...\n")
        params = load_params(fname=args.path_params, tokens=tokens, device=device, dtype=dtype)
        L, q = params["bias"].shape
        
        # ----------------------------
        # Load dataset
        # ----------------------------
        print(f"Loading data from {args.data}...")
        f.write(f"Loading data from {args.data}...\n")
        dataset = DatasetDCA(
            path_data=args.data,
            path_weights=args.weights,
            alphabet=tokens,
            clustering_th=args.clustering_seqid,
            no_reweighting=True,
            device=device,
            dtype=dtype,
        )

        data = one_hot(dataset.data, num_classes=len(tokens)).to(dtype)
        weights = dataset.weights        

        if args.pseudocount is None:
            args.pseudocount = 1. / weights.sum()

        print(f"Using pseudocount: {args.pseudocount:.3f}...")
        f.write(f"Using pseudocount: {args.pseudocount:.3f}...\n")
        f.flush()

        # ----------------------------
        # STEP 0 — Compute entropy of data
        # ----------------------------
        print("\nSTEP 0: compute entropy of the data...\n")   
        f.write("\nSTEP 0: compute entropy of the data...\n"), f.flush()

        eps = 1e-12
        energy_data = compute_energy(data, params=params)

        print(f"Average energy of the data: {energy_data.mean():.3f} +/- {energy_data.std():.3f}")
        f.write(f"Average energy of the data: {energy_data.mean():.3f} +/- {energy_data.std():.3f}\n")

        energy_data = energy_data + torch.abs(torch.min(energy_data))
        exp_e = torch.exp(-energy_data)
        p_data = exp_e / torch.sum(exp_e)
        entropy_data = -torch.sum(p_data * torch.log(p_data + eps)).cpu().numpy()

        print("\n" + 50 * "*")
        print(f"S(data) = {entropy_data}")
        print(50 * "*")
        f.write(50 * "*")
        f.write(f"\nS(data) = {entropy_data}\n")
        f.write(50 * "*" + "\n")
        f.flush()
       
        # ----------------------------
        # STEP 1 — Sampling with coverage by distance
        # ----------------------------
        print(f"\nSTEP 1: Sampling with good coverage at distances in the range [0, {L}] from closest natural\n")
        f.write(f"\nSTEP 1: Sampling with good coverage at distances in the range [0, {L}] from closest natural\n")
        f.flush()

        d_max = 20
        n = args.ngen 
        n_tot = n * data.shape[0]

        S_tot = entropy_data * np.ones(d_max + 1)
        S_seq = np.zeros(data.shape[0])
        distance = torch.zeros(n_tot, dtype=torch.long, device=device)
        hist = torch.zeros((data.shape[0], L + 1), device=device, dtype=dtype)
        
        sampler = get_sampler("gibbs_importance_many_targets")

        # ----------------------------
        # Loop over target distances
        # ----------------------------
        for d in range(0, d_max):
            print(f"\nsampling with good coverage at distance {d} and {d+1}...")
            f.write(f"\nsampling with good coverage at distance {d} and {d+1}...\n")

            delta_theta = args.delta_theta * torch.ones(n_tot)
            theta = torch.zeros(n_tot)

            # Iteratively adjust theta until coverage is achieved
            while True:
                theta += delta_theta 
                samples = init_chains(num_chains=n_tot, L=L, q=q, device=device, dtype=dtype)
                samples = sampler(
                    chains=samples,
                    params=params,
                    target_seq=data,
                    gamma=theta,
                    distance=d,
                    nsweeps=args.steps
                )

                # Compute histogram of sequence distances
                for i in range(data.shape[0]):
                    idxs_i = torch.arange(i*n, (i+1)*n)
                    seq_ID = compute_seqID(samples[idxs_i], data[i])
                    distance[idxs_i] = (L - seq_ID).to(torch.long)   # <-- fixed here
                    hist[i, :] = torch.bincount(distance[idxs_i].to(torch.int64), minlength=L+1) / n

                # Verbose progress
                if args.verbose:
                    print(f"theta_mean={theta.mean().item()}")
                    print(f"average coverage at distance {d} and {d+1}: {hist[:, d].mean().item():.3f}, {hist[:, d+1].mean().item():.3f}")
                    f.write(f"theta_mean={theta.mean().item()}\n")
                    f.write(f"average coverage at distance {d} and {d+1}: {hist[:, d].mean().item():.3f}, {hist[:, d+1].mean().item():.3f}\n")

                # ----------------------------
                # Adjust delta_theta adaptively based on coverage
                # ----------------------------
                remaining_list = []
                for i in range(data.shape[0]):
                    idxs_i = torch.arange(i * n, (i + 1) * n)

                    # If histogram indicates partial coverage, reduce step size
                    if (1 <= torch.where(hist[i] > args.threshold)[0].shape[0] <= 2) and hist[i, d].item() > 0.5: 
                        theta[idxs_i] = torch.minimum(
                            torch.tensor(0.0, device=theta.device, dtype=theta.dtype), 
                            theta[idxs_i] - 2 * delta_theta[idxs_i]
                        )
                        delta_theta[idxs_i] = delta_theta[idxs_i] / 2

                        if args.verbose:
                            print(f"reducing delta_theta[{i}] to : {delta_theta[idxs_i]}")
                            f.write(f"reducing delta_theta[{i}] to : {delta_theta[idxs_i]}\n")
                            
                    # Continue adjusting until both distances d and d+1 are covered
                    if hist[i, d] > args.threshold and hist[i, d + 1] > args.threshold:
                        theta[idxs_i] -= delta_theta[idxs_i]
                    else:
                        remaining_list.append(i)

                print("remaining: ", len(remaining_list), flush=True)
                f.write(f"remaining:  {len(remaining_list)}\n")

                # Exit loop when all targets achieved desired coverage
                if remaining_list.__len__() == 0:
                    break

            # ----------------------------
            # Compute entropies per sequence and update totals
            # ----------------------------
            for i in range(data.shape[0]):
                idxs_i = torch.arange(i * n, (i + 1) * n)
                dist_i = distance[idxs_i]
                seqs_i = samples[idxs_i]

                energies_i_d = compute_energy(seqs_i[dist_i == d], params=params)
                mean_energy_i_d = energies_i_d.mean().item()
                logP = torch.log(hist[i, d]).item()
                lnZ = S_seq[i] - mean_energy_i_d - logP

                energies_i_d1 = compute_energy(seqs_i[dist_i == d + 1], params=params) + theta[n * i + 1]
                mean_energy_i_d1 = energies_i_d1.mean().item()
                logP = torch.log(hist[i, d + 1]).item()
                S_seq[i] = lnZ + mean_energy + logP
                S_tot[d + 1] += p_data[i] * S_seq[i]

            # ----------------------------
            # Verbose output of S_tot
            # ----------------------------
            if args.verbose:
                print("\n" + 50 * "*")
                print(f"\\S_tot[{d+1}] : {S_tot[d+1]:.3f}")
                print(50 * "*")
                f.write(50 * "*")
                f.write(f"\\S_tot[{d+1}] : {S_tot[d+1]:.3f}\n")
                f.write(50 * "*" + "\n")
                f.flush()


    




if __name__ == "__main__":
    # main()
    # main_inverse()
    main_idependent_new()