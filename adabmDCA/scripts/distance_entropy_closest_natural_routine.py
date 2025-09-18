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
from adabmDCA.sampling import get_sampler
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
        
        # chains[:, i, :] = res_new

        distance_new = compute_min_distance(chains_sub, target_seqs) # (N, M) distance_new = compute_min_distance(chains, target_seqs) #
        delta_seqID = torch.abs(distance_new - distance) - torch.abs(distance_old - distance)

        # chains[:, i, :] = res_old

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


# @torch.jit.script
# def compute_min_distance(a1: torch.Tensor, a2: torch.Tensor, chunk_m: int = 4096) -> torch.Tensor:
#     """
#     Min Hamming distance tra ogni seq di a1 e tutte le seq di a2.
#     a1: (N, L, C) one-hot
#     a2: (M, L, C) one-hot
#     Ritorna: (N,) distanza minima (intero)
#     """
#     # Assumi one-hot lungo C: comprimiamo a indici token (N,L) e (M,L)
#     # int16 basta per C <= 32767; cambia a int32 se vuoi andare sul sicuro.
#     ids1 = torch.argmax(a1, dim=-1).to(torch.int16)
#     ids2 = torch.argmax(a2, dim=-1).to(torch.int16)
#     N = ids1.size(0)
#     L = ids1.size(1)
#     M = ids2.size(0)
#     # min dist inizializzato a L (massima Hamming)
#     min_dist = torch.full((N,), L, device=ids1.device, dtype=torch.int32)
#     # Per ridurre allocazioni, pre-espandi una sola volta ids1 lungo dim=1
#     ids1_exp = ids1.unsqueeze(1)  # (N,1,L)
#     # Scorri M a blocchi
#     m = 0
#     while m < M:
#         m_end = min(m + chunk_m, M)
#         block = ids2[m:m_end]               # (m_chunk, L)
#         block = block.unsqueeze(0)          # (1, m_chunk, L)
#         # Confronto element-wise; eq è uint8/bool → somma lungo L dà #match
#         eq = ids1_exp == block              # (N, m_chunk, L)
#         matches = eq.sum(dim=-1, dtype=torch.int32)  # (N, m_chunk)
#         # Distanza di Hamming = L - #match
#         dist_block = (L - matches)          # (N, m_chunk)
#         # Min sul blocco corrente, poi aggiorna min globale
#         block_min = dist_block.min(dim=1).values  # (N,)
#         min_dist = torch.minimum(min_dist, block_min)
#         m = m_end
#     return min_dist

# @torch.jit.script
# def compute_min_distance(a1: torch.Tensor, a2: torch.Tensor, chunk_m: int = 1000) -> torch.Tensor:
#     """
#     Min Hamming distance tra ogni seq di a1 e tutte le seq di a2.
#     a1: (N, L, C) one-hot
#     a2: (M, L, C) one-hot
#     Ritorna: (N,) distanza minima (int)
#     """
#     ids1 = torch.argmax(a1, dim=-1).to(torch.int16)  # (N, L)
#     ids2 = torch.argmax(a2, dim=-1).to(torch.int16)  # (M, L)

#     N, L = ids1.shape
#     M = ids2.size(0)

#     min_dist = torch.full((N,), L, device=ids1.device, dtype=torch.int32)

#     # blocchi su M
#     m = 0
#     while m < M:
#         m_end = min(m + chunk_m, M)
#         block = ids2[m:m_end]  # (m_chunk, L)

#         # Broadcasting diretto: (N,1,L) vs (1,m_chunk,L)
#         # Calcola mismatches e somma lungo L
#         dist_block = (ids1.unsqueeze(1) != block.unsqueeze(0)).sum(dim=-1, dtype=torch.int32)  # (N, m_chunk)

#         # minima distanza per ogni riga
#         block_min = dist_block.min(dim=1).values  # (N,)
#         min_dist = torch.minimum(min_dist, block_min)
#         m = m_end

#     return min_dist
# import torch

@torch.jit.script
def compute_min_distance_2(a1: torch.Tensor, a2: torch.Tensor, chunk_m: int = 256, chunk_n: int = 32_768) -> torch.Tensor:
    """
    Min Hamming distance tra ogni seq di a1 e tutte le seq di a2.
    a1: (N, L, C) one-hot (float/bool/byte, su device di a1)
    a2: (M, L, C) one-hot (stesso device)
    Ritorna: (N,) distanza minima int32
    Note:
      - Evita broadcasting 3D; usa GEMM su (L*C).
      - chunk_m: blocchi lungo M (righe di a2)
      - chunk_n: blocchi lungo N (righe di a1)
    """
    N = a1.size(0)
    L = a1.size(1)
    C = a1.size(2)
    M = a2.size(0)

    # Flatten su L*C; restiamo in float32 per conteggi esatti (0..L)
    a1f = a1.to(torch.float32).reshape(N, L * C).contiguous()

    # Distanza minima inizializzata a L
    min_dist = torch.full((N,), L, device=a1.device, dtype=torch.int32)

    m = 0
    while m < M:
        m_end = m + chunk_m
        if m_end > M:
            m_end = M

        # (m_chunk, L*C) e trasposta per GEMM efficiente
        a2f = a2[m:m_end].to(torch.float32).reshape(m_end - m, L * C).contiguous()
        a2f_t = a2f.transpose(0, 1).contiguous()  # (L*C, m_chunk)

        n = 0
        while n < N:
            n_end = n + chunk_n
            if n_end > N:
                n_end = N

            # (n_chunk, L*C) @ (L*C, m_chunk) -> (n_chunk, m_chunk)
            matches = torch.matmul(a1f[n:n_end], a2f_t)  # float32 in [0, L]
            # Hamming = L - #match
            dblock = (L - matches).to(torch.int32)       # (n_chunk, m_chunk)

            # min per riga e aggiornamento running-min
            block_min = dblock.min(dim=1).values         # (n_chunk,)
            cur = min_dist[n:n_end]
            min_dist[n:n_end] = torch.minimum(cur, block_min)

            n = n_end
        m = m_end

    return min_dist

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
            no_reweighting=True,
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

        # Sample with theta = 0
        args.n0 = 100_000
        if args.target_path is not None: 
            _, samples_0 = import_from_fasta(args.target_path, tokens=tokens, filter_sequences=True)
            samples_0 = one_hot(torch.tensor(samples_0), num_classes=len(tokens)).to(device=device, dtype=dtype).squeeze(0)[:args.n0, :, :]
        else: 
            print("\nSTEP 0: sample until equilibrium with gamma=0...\n")   
            f.write("\nSTEP 0: sample until equilibrium with gamma=0...\n"), f.flush()
            sampler_0 = get_sampler("gibbs")
            samples_0 = init_chains(num_chains=args.n0, L=L, q=q, device=device, dtype=dtype)
            samples_0 = sampler_0(chains=samples_0, params=params, nsweeps=10_000,  beta=args.beta)

        samples = init_chains(
            num_chains=args.ngen,
            L=L,
            q=q,
            device=device,
            dtype=dtype)
        
        print("\n" + 50 * "*")
        print(f"S(model) = {args.entropy}")
        print(50 * "*")
        f.write(50 * "*")
        f.write(f"\nS(model) = {args.entropy}\n")
        f.write(50 * "*" + "\n")
        f.flush()
        
        distance = compute_min_distance(samples_0, data) 
        hist = torch.bincount(distance.to(torch.int64), minlength=L+1) / args.n0
        for i, d in enumerate(hist):
            if args.verbose:
                print(f"Percentage of sequences with distance {i}: {d.item() * 100:.2f}%\n" )
            f.write(  f"Percentage of sequences with distance {i}: {d.item() * 100:.2f}%\n" )
        f.flush()

        idx = torch.where(hist >= 0.03)[0].cpu()
        max_distance = idx.min().item()
        fraction_max_d = hist[max_distance] 

        energies_all = compute_energy(samples_0, params=params)
        mean_energy_all = energies_all.mean().item()
        std_energy_all = energies_all.std(unbiased=False).item()
        lnZ = args.entropy - mean_energy_all
        
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
            f.write(f"logP d={max_distance} : {logP:.3f}\n")


        Entropies = {}
        entropies = []
        seqID_closest = []

        S = lnZ + mean_energy + logP

        Entropies[max_distance] = S
        entropies.append(S)
        seqID_closest.append(max_distance)
        
        print("\n" + 50 * "*")
        print(f"S(d={max_distance}) = {S:.3f}")
        print(50 * "*")
        f.write("\n" + 50 * "*")
        f.write(f"\nS(d={max_distance}) = {S:.3f}\n")
        f.write(50 * "*" + "\n")

        print(f"\nSTEP 1: Sampling with good coverage at distances in the range [0, {max_distance}] from closest natural\n")
        f.write(f"\nSTEP 1: Sampling with good coverage at distances in the range [0, {max_distance}] from closest natural\n")
        f.flush()

        sampler = metropolis_target_seqs_fast
        delta_theta = args.delta_theta
        for d in range(max_distance - 1, -1, -1):
            print(f"\nsampling with good coverage at distance {d} and {d+1}...")
            f.write(f"\nsampling with good coverage at distance {d} and {d+1}...\n")

            theta = 0

            while True:
            
                theta += delta_theta 
                samples = sampler(chains=samples, params=params, target_seqs=data, theta=theta, distance=d, nsweeps=args.steps)
                distance = compute_min_distance(samples, data) #/ L * 100
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
                f.write(f"logP (d={d}) : {logP:.3f}\n")

            Entropies[hist[d]] = S
            print("\n" + 50 * "*")
            print(f"S(d={d}) = {S:.3f}")
            print(50 * "*")
            f.write("\n" + 50 * "*")
            f.write(f"\nS(d={d}) = {S:.3f}\n")
            f.write(50 * "*" + "\n")

        f.write(f"\nS(models) = {args.entropy:.3f}\n")
        f.write("\nS(d) = " + ", ".join(f"{x:.3f}" for x in entropies) + "\n")
        f.write("\nseqID closest = " + ", ".join(f"{x:.3f}" for x in seqID_closest) + "\n")
        f.write(50 * "*" + "\n")
        
    
if __name__ == "__main__":
    main()