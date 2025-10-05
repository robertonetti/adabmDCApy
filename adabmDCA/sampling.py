from typing import Dict, Callable

import torch
from torch.nn.functional import one_hot as one_hot_torch

from adabmDCA.functional import one_hot
from adabmDCA.resampling import compute_seqID 

@torch.jit.script
def _gibbs_sweep(
    chains: torch.Tensor,
    residue_idxs: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float,
) -> torch.Tensor:
    """Performs a Gibbs sweep over the chains.

    Args:
        chains (torch.Tensor): One-hot encoded sequences.
        residue_idxs (torch.Tensor): List of residue indices in random order.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        beta (float): Inverse temperature.

    Returns:
        torch.Tensor: Updated chains.
    """
    N, L, q = chains.shape
    for i in residue_idxs:
        # Select the couplings attached to the residue we are considering (i) and flatten along the other residues ({j})
        couplings_residue = params["coupling_matrix"][i].view(q, L * q)
        # Update the chains
        logit_residue = beta * (params["bias"][i].unsqueeze(0) + chains.reshape(N, L * q) @ couplings_residue.T) # (N, q)
        chains[:, i, :] = one_hot(torch.multinomial(torch.softmax(logit_residue, -1), 1), num_classes=q).to(logit_residue.dtype).squeeze(1)
        
    return chains


def gibbs_sampling(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """Gibbs sampling.
    
    Args:
        chains (torch.Tensor): Initial chains.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        nsweeps (int): Number of sweeps.
        beta (float, optional): Inverse temperature. Defaults to 1.0.
    
    Returns:
        torch.Tensor: Updated chains.
    """
    L = params["bias"].shape[0]
    
    for t in torch.arange(nsweeps):
        # Random permutation of the residues
        residue_idxs = torch.randperm(L)
        chains = _gibbs_sweep(chains, residue_idxs, params, beta)
        
    return chains


def _gibbs_sweep_importance(
    chains: torch.Tensor,
    residue_idxs: torch.Tensor,
    params: Dict[str, torch.Tensor],
    target_seq: torch.Tensor, # (L, q)
    gamma: float,
    distance: int,

    target_seq2: torch.Tensor = None,
    gamma2: float = None,
    distance2: int = None,
    beta: float = 1,
) -> torch.Tensor:
    """
    Perform a Gibbs sweep with importance weighting toward one or two target sequences.
    Args:
        chains (torch.Tensor): (N, L, q) one-hot encoded sequences.
        residue_idxs (torch.Tensor): Indices of residues to update.
        params (Dict[str, torch.Tensor]): Model parameters:
            - "bias": (L, q)
            - "coupling_matrix": (L, L, q, q)
        target_seq (torch.Tensor): (L, q) reference sequence.
        gamma (float): Weight for distance penalty from `target_seq`.
        distance (int): Target distance (L - seqID) from `target_seq`.
        target_seq2 (torch.Tensor, optional): Second reference sequence. Default: None.
        gamma2 (float, optional): Weight for second distance penalty. Default: None.
        distance2 (int, optional): Second target distance. Default: None.
        beta (float, optional): Inverse temperature for energy scaling. Default: 1.
    Returns:
        torch.Tensor: Updated chains of shape (N, L, q) after one full sweep.
    """
    N, L, q = chains.shape
    distance_target = L - compute_seqID(chains, target_seq)  # (N,)
    if target_seq2 is not None:
        distance_target2 = L - compute_seqID(chains, target_seq2)
    for i in residue_idxs:
        # Select the couplings attached to the residue we are considering (i) and flatten along the other residues ({j})
        couplings_residue = params["coupling_matrix"][i].view(q, L * q)
        residue_target = target_seq[i]
        res_old = chains[:, i, :].clone()

        flag = (res_old * residue_target).sum(-1)  # (N,)  1 if the residue is the same as target, 0 otherwise 
        distance_target_new = distance_target.unsqueeze(1) - torch.where(flag.unsqueeze(-1) == 1, residue_target.unsqueeze(0) - 1, residue_target.unsqueeze(0))  # (N, q)
        delta_distance = torch.abs(distance_target_new - distance)  # (N, q)  


        # Update the chains
        if target_seq2 is not None:
            residue_target2 = target_seq2[i]
            flag2 = (res_old * residue_target2).sum(-1)  # (N,)  1 if the residue is the same as target, 0 otherwise 
            distance_target2_new = distance_target2.unsqueeze(1) - torch.where(flag2.unsqueeze(-1) == 1, residue_target2.unsqueeze(0) - 1, residue_target2.unsqueeze(0))  # (N, q)
            delta_distance2 = torch.abs(distance_target2_new - distance2)
            logit_residue = beta * (params["bias"][i].unsqueeze(0) + chains.reshape(N, L * q) @ couplings_residue.T) - gamma * delta_distance - gamma2 * delta_distance2 # (N, q)
        else:
            logit_residue = beta * (params["bias"][i].unsqueeze(0) + chains.reshape(N, L * q) @ couplings_residue.T) - gamma * delta_distance

        chains[:, i, :] = one_hot(torch.multinomial(torch.softmax(logit_residue, -1), 1), num_classes=q).to(logit_residue.dtype).squeeze(1)
        delta_seqID = (chains[:, i, :] * target_seq[i]).sum(-1) - (res_old * target_seq[i]).sum(-1)  
        distance_target = distance_target - delta_seqID
        if target_seq2 is not None:
            delta_seqID2 = (chains[:, i, :] * target_seq2[i]).sum(-1) - (res_old * target_seq2[i]).sum(-1)  
            distance_target2 = distance_target2 - delta_seqID2
      
    return chains


def gibbs_sampling_importance(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    target_seq: torch.Tensor, # (L, q)
    gamma: float,
    distance: int,
    target_seq2: torch.Tensor = None,
    gamma2: float = None,
    distance2: int = None,
    nsweeps: int = 1,
    beta: float = 1.0,
) -> torch.Tensor:
    """Gibbs sampling.
    
    Args:
        chains (torch.Tensor): Initial chains.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        nsweeps (int): Number of sweeps.
        beta (float, optional): Inverse temperature. Defaults to 1.0.
    
    Returns:
        torch.Tensor: Updated chains.
    """
    L = params["bias"].shape[0]
    
    for t in torch.arange(nsweeps):
        # Random permutation of the residues
        residue_idxs = torch.randperm(L)
        chains = _gibbs_sweep_importance(chains, residue_idxs, params, target_seq, gamma, distance, target_seq2, gamma2, distance2, beta)
        
    return chains





def _gibbs_sweep_importance_many_targets(
    chains: torch.Tensor,
    residue_idxs: torch.Tensor,
    params: Dict[str, torch.Tensor],
    target_seqs: torch.Tensor, # (L, q)
    gamma: float,
    distance: int,
    beta: float = 1,
) -> torch.Tensor:
    

    N, L, q = chains.shape
    n_x_tgt = N // target_seqs.shape[0]

    distance_target = torch.zeros(N, dtype=torch.long, device=chains.device)
    for i in range(target_seqs.shape[0]):
        target_seq = target_seqs[i]
        chains_i = chains[i * n_x_tgt:(i + 1) * n_x_tgt]
        distance_target[i * n_x_tgt:(i + 1) * n_x_tgt] = L - compute_seqID(chains_i, target_seq)  # (n_x_tgt,)

    for i in residue_idxs:
        # Select the couplings attached to the residue we are considering (i) and flatten along the other residues ({j})
        couplings_residue = params["coupling_matrix"][i].view(q, L * q)
        residue_target = target_seqs[:, i, :]  # (n_targets, q)
        residue_target = residue_target.repeat_interleave(n_x_tgt, dim=0)  # (N, q)
        res_old = chains[:, i, :].clone() # (N, q)
        flag = (res_old * residue_target).sum(-1)  # (N,)  1 if the residue is the same as target, 0 otherwise 

        distance_target_new = distance_target.unsqueeze(1) - torch.where(flag.unsqueeze(-1) == 1, residue_target - 1, residue_target)  # (N, q)
        delta_distance = torch.abs(distance_target_new - distance)  # (N, q)  

       
        logit_residue = beta * (params["bias"][i].unsqueeze(0) + chains.reshape(N, L * q) @ couplings_residue.T) - gamma.unsqueeze(1).to(delta_distance.device) * delta_distance

        chains[:, i, :] = one_hot(torch.multinomial(torch.softmax(logit_residue, -1), 1), num_classes=q).to(logit_residue.dtype).squeeze(1)

        delta_seqID = (chains[:, i, :] * residue_target).sum(-1) - (res_old * residue_target).sum(-1)
        distance_target = distance_target - delta_seqID
       
    return chains

def gibbs_sampling_importance_many_targets(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    target_seq: torch.Tensor, # (L, q)
    gamma: float,
    distance: int,
    nsweeps: int = 1,
    beta: float = 1.0,
) -> torch.Tensor:

    L = params["bias"].shape[0]
    
    for t in torch.arange(nsweeps):
        # Random permutation of the residues
        residue_idxs = torch.randperm(L)
        chains = _gibbs_sweep_importance_many_targets(chains, residue_idxs, params, target_seq, gamma, distance, beta)
        
    return chains






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
    

def _metropolis_sweep(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float,
) -> torch.Tensor:
    """Performs a Metropolis sweep over the chains.

    Args:
        chains (torch.Tensor): One-hot encoded sequences.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        beta (float): Inverse temperature.

    Returns:
        torch.Tensor: Updated chains.
    """
    
    N, L, q = chains.shape
    residue_idxs = torch.randperm(L)
    for i in residue_idxs:
        res_old = chains[:, i, :]
        res_new = one_hot_torch(torch.randint(0, q, (N,), device=chains.device), num_classes=q).float()
        delta_E = _get_deltaE(i, chains, res_old, res_new, params, L, q)
        accept_prob = torch.exp(- beta * delta_E).unsqueeze(-1)
        chains[:, i, :] = torch.where(accept_prob > torch.rand((N, 1), device=chains.device, dtype=chains.dtype), res_new, res_old)

    return chains
    

def metropolis(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """Metropolis sampling.

    Args:
        chains (torch.Tensor): One-hot encoded sequences.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        nsweeps (int): Number of sweeps to be performed.
        beta (float, optional): Inverse temperature. Defaults to 1.0.

    Returns:
        torch.Tensor: Updated chains.
    """

    for _ in range(nsweeps):
        chains = _metropolis_sweep(chains, params, beta)

    return chains






def _metropolis_importance_sweep(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    target_seq: torch.Tensor, # (L, q)
    gamma: float,
    target_seq2: torch.Tensor = None,
    gamma2: float = None,
    distance: int = 0, 
    beta: float = 1,
) -> torch.Tensor:
    
    N, L, q = chains.shape
    residue_idxs = torch.randperm(L)

    if target_seq2 is None:
        seqID_target = compute_seqID(chains, target_seq)  # (N,)
        for i in residue_idxs:
            res_old = chains[:, i, :]
            res_new = one_hot_torch(torch.randint(0, q, (N,), device=chains.device), num_classes=q).float()
            delta_seqID = (res_new * target_seq[i]).sum(-1) - (res_old * target_seq[i]).sum(-1)   # (N,)

            delta_E = _get_deltaE(i, chains, res_old, res_new, params, L, q)

            distance_old = L - (seqID_target)
            distance_new = L - (seqID_target + delta_seqID)
            delta_d_old = torch.abs(distance_old - distance) 
            delta_d_new = torch.abs(distance_new - distance)
            mask_d = (delta_d_new < delta_d_old).float()
            term_distance = delta_d_new ** 2 #delta_d_new - delta_d_old

            accept_prob = torch.exp(- beta * delta_E + gamma * mask_d).unsqueeze(-1)
            accepted_mask = accept_prob > torch.rand((N, 1), device=chains.device, dtype=chains.dtype)
            chains[:, i, :] = torch.where(accepted_mask, res_new, res_old)
            seqID_target = torch.where(accepted_mask.squeeze(-1), seqID_target + delta_seqID, seqID_target)  # (N,)
    else: 
        raise ValueError("Not implemented yet.")
    return chains
    

def metropolis_importance(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    target_seq: torch.Tensor, # (L, q)
    gamma: float,
    target_seq2: torch.Tensor = None,
    gamma2: float = None,
    distance: int = 0, 
    nsweeps : int = 1,
    beta: float = 1,
) -> torch.Tensor:

    for _ in range(nsweeps):
        chains = _metropolis_importance_sweep(chains, params, target_seq, gamma, target_seq2, gamma2, distance, beta)

    return chains


def get_sampler(sampling_method: str) -> Callable:
    """Returns the sampling function corresponding to the chosen method.

    Args:
        sampling_method (str): String indicating the sampling method. Choose between 'metropolis' and 'gibbs'.

    Raises:
        KeyError: Unknown sampling method.

    Returns:
        Callable: Sampling function.
    """
    if sampling_method == "gibbs":
        return gibbs_sampling
    elif sampling_method == "metropolis":
        return metropolis
    elif sampling_method == "importance":
        return metropolis_importance
    elif sampling_method == "gibbs_importance":
        return gibbs_sampling_importance
    elif sampling_method == "gibbs_importance_many_targets":
        return gibbs_sampling_importance_many_targets
    else:
        raise KeyError("Unknown sampling method. Choose between 'metropolis' and 'gibbs'.")