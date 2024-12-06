from typing import Callable, Dict

import torch

from adabmDCA.training import train_graph
from adabmDCA.checkpoint import Checkpoint

    
def fit(
    sampler: Callable,
    fi_target: torch.Tensor,
    fij_target: torch.Tensor,
    params: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    chains: torch.Tensor,
    log_weights: torch.Tensor,
    target_pearson: float,
    nsweeps: int,
    nepochs: int,
    lr: float,
    checkpoint: Checkpoint | None = None,
    *args, **kwargs,
) -> None:
    """Trains a bmDCA model on the input MSA and saves the results in a file.

    Args:
        sampler (Callable): Sampling function to be used.
        fi_target (torch.Tensor): Single-point frequencies of the data.
        fij_target (torch.Tensor): Two-point frequencies of the data.
        params (Dict[str, torch.Tensor]): Initialization of the model's parameters.
        mask (torch.Tensor): Initialization of the coupling matrix's mask.
        chains (torch.Tensor): Initialization of the Markov chains.
        log_weights (torch.Tensor): Log-weights of the chains. Used to estimate the log-likelihood.
        target_pearson (float): Pearson correlation coefficient on the two-points statistics to be reached.
        nsweeps (int): Number of Monte Carlo steps to update the state of the model.
        nepochs (int): Maximum number of epochs to be performed.
        lr (float): Learning rate.
        checkpoint (Checkpoint | None): Checkpoint class to be used to save the model. Defaults to None.
    """
    
    # Check the input sizes
    if fi_target.dim() != 2:
        raise ValueError("fi_target must be a 2D tensor")
    if fij_target.dim() != 4:
        raise ValueError("fij_target must be a 4D tensor")
    if chains.dim() != 3:
        raise ValueError("chains must be a 3D tensor")
    
    # Training loop    
    train_graph(
        sampler=sampler,
        chains=chains,
        log_weights=log_weights,
        mask=mask,
        fi=fi_target,
        fij=fij_target,
        params=params,
        nsweeps=nsweeps,
        lr=lr,
        max_epochs=nepochs,
        target_pearson=target_pearson,
        check_slope=False,
        checkpoint=checkpoint,
    )
