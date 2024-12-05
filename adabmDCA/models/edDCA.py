from pathlib import Path
from typing import Callable, Dict
import time

import torch

from adabmDCA.stats import get_correlation_two_points
from adabmDCA.training import train_graph
from adabmDCA.utils import get_mask_save
from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points
from adabmDCA.graph import decimate_graph, compute_density
from adabmDCA.statmech import compute_log_likelihood, _update_weights_AIS, compute_entropy
from adabmDCA.checkpoint import Checkpoint

MAX_EPOCHS = 10000

def fit(
    sampler: Callable,
    chains: torch.Tensor,
    log_weights: torch.Tensor,
    fi_target: torch.Tensor,
    fij_target: torch.Tensor,
    params: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    lr: float,
    nsweeps: int,
    target_pearson: float,
    target_density: float,
    drate: float,
    checkpoint: Checkpoint | None = None,
    *args, **kwargs,
):
    """Fits an edDCA model on the training data and saves the results in a file.
    
    Args:
        sampler (Callable): Sampling function to be used.
        chains (torch.Tensor): Initialization of the Markov chains.
        log_weights (torch.Tensor): Log-weights of the chains. Used to estimate the log-likelihood.
        fi_target (torch.Tensor): Single-point frequencies of the data.
        fij_target (torch.Tensor): Two-point frequencies of the data.
        params (Dict[str, torch.Tensor]): Initialization of the model's parameters.
        mask (torch.Tensor): Initialization of the coupling matrix's mask.
        lr (float): Learning rate.
        nsweeps (int): Number of Monte Carlo steps to update the state of the model.
        target_pearson (float): Pearson correlation coefficient on the two-points statistics to be reached.
        target_density (float): Target density of the coupling matrix.
        drate (float): Percentage of active couplings to be pruned at each decimation step.
        checkpoint (Checkpoint | None): Checkpoint class to be used to save the model. Defaults to None.
        """
    time_start = time.time()
    
    # Check the input sizes
    if fi_target.dim() != 2:
        raise ValueError("fi_target must be a 2D tensor")
    if fij_target.dim() != 4:
        raise ValueError("fij_target must be a 4D tensor")
    if chains.dim() != 3:
        raise ValueError("chains must be a 3D tensor")
    
    L, q = params["bias"].shape
    device = fi_target.device
    dtype = fi_target.dtype
    
    
    print("Bringing the model to the convergence threshold...")
    chains, params, log_weights = train_graph(
        sampler=sampler,
        chains=chains,
        log_weights=log_weights,
        mask=mask,
        fi=fi_target,
        fij=fij_target,
        params=params,
        nsweeps=nsweeps,
        lr=lr,
        max_epochs=MAX_EPOCHS,
        target_pearson=target_pearson,
        check_slope=True,
        checkpoint=checkpoint,
    )
    
    # Get the single-point and two-points frequencies of the simulated data
    pi = get_freq_single_point(data=chains, weights=None, pseudo_count=0.)
    pij = get_freq_two_points(data=chains, weights=None, pseudo_count=0.)
    
    # Mask for saving only the upper-diagonal matrix
    mask_save = get_mask_save(L, q, device=device)
    
    # Filenames for the decimated parameters and chains
    parent, name = checkpoint.file_paths["params"].parent, checkpoint.file_paths["params"].name
    new_name = name.replace(".dat", "_dec.dat")
    checkpoint.file_paths["params_dec"] = Path(parent).joinpath(new_name)
    
    name = checkpoint.file_paths["chains"].name
    new_name = name.replace(".fasta", "_dec.fasta")
    checkpoint.file_paths["chains_dec"] = Path(parent).joinpath(new_name)
    
    print(f"\nStarting the decimation (target density = {target_density}):")
    template_log = "{0:10} {1:10} {2:10} {3:10} {4:10} {5:10} {6:10}\n"
    with open(checkpoint.file_paths["log"], "a") as f:
        f.write("\nDecimation\n")
        f.write(f"Target density: {target_density}\n")
        f.write(f"Decimation rate: {drate}\n\n")
        f.write(template_log.format("Epoch", "Pearson", "Slope", "LL", "Entropy", "Density", "Time [s]"))
        
    # Template for wrinting the results
    template = "{0:15} | {1:15} | {2:15} | {3:15} | {4:15}"
    density = compute_density(mask)
    count = 0
    checkpoint.checkpt_interval = 10
    
    while density > target_density:
        count += 1
        
        # Store the previous parameters
        prev_params = {key: value.clone() for key, value in params.items()}
        
        # Decimate the model
        params, mask = decimate_graph(
            pij=pij,
            params=params,
            mask=mask,
            drate=drate
        )
        
        # Equilibrate the model
        chains = sampler(
            chains=chains,
            params=params,
            nsweeps=nsweeps,
        )
        
        # Update the log-weights
        log_weights = _update_weights_AIS(
            prev_params=prev_params,
            curr_params=params,
            chains=chains,
            log_weights=log_weights,
        )
        
        # Bring the model at convergence on the graph
        chains, params, log_weights = train_graph(
            sampler=sampler,
            chains=chains,
            log_weights=log_weights,
            mask=mask,
            fi=fi_target,
            fij=fij_target,
            params=params,
            nsweeps=nsweeps,
            lr=lr,
            max_epochs=MAX_EPOCHS,
            target_pearson=target_pearson,
            check_slope=True,
            progress_bar=False,
            checkpoint=None,
        )
        
        # Compute the single-point and two-points frequencies of the simulated data
        pi = get_freq_single_point(data=chains, weights=None, pseudo_count=0.)
        pij = get_freq_two_points(data=chains, weights=None, pseudo_count=0.)
        
        pearson, slope = get_correlation_two_points(fi=fi_target, pi=pi, fij=fij_target, pij=pij)
        density = compute_density(mask)
        logZ = (torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))).item()
        log_likelihood = compute_log_likelihood(fi=fi_target, fij=fij_target, params=params, logZ=logZ)
        
        print(template.format(f"Step: {count}", f"Density: {density:.3f}", f"LL: {log_likelihood:.3f}", f"Pearson: {pearson:.3f}", f"Slope: {slope:.3f}"))
                
        if checkpoint.check(count, params, chains):
            entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
            checkpoint.save(
                params=params,
                mask=torch.logical_and(mask, mask_save),
                chains=chains,
                log_weights=log_weights,
                epochs=count,
                pearson=pearson,
                slope=slope,
                log_likelihood=log_likelihood,
                entropy=entropy,
                density=density,
                time_start=time_start,
            )
    
    entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
    checkpoint.save(
        params=params,
        mask=torch.logical_and(mask, mask_save),
        chains=chains,
        log_weights=log_weights,
        epochs=count,
        pearson=pearson,
        slope=slope,
        log_likelihood=log_likelihood,
        entropy=entropy,
        density=density,
        time_start=time_start,
    )
    print(f"Completed, decimated model parameters saved in {checkpoint.file_paths['params_dec']}")