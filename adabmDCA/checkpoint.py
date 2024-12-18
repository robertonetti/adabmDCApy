from typing import Dict, Any
from abc import ABC, abstractmethod
import torch
import h5py

from adabmDCA.io import save_chains, save_params
from adabmDCA.statmech import _get_acceptance_rate


class Checkpoint(ABC):
    """Helper class to save the model's parameters and chains at regular intervals during training and to log the
    progress of the training.
    """
    def __init__(
        self,
        file_paths: dict,
        tokens: str,
        args: dict,
        params: Dict[str, torch.Tensor] | None = None,
        chains: Dict[str, torch.Tensor] | None = None,
    ):
        """Initializes the Checkpoint class.

        Args:
            file_paths (dict): Dictionary containing the paths of the files to be saved.
            tokens (str): Alphabet to be used for encoding the sequences.
            args (dict): Dictionary containing the arguments of the training.
            params (Dict[str, torch.Tensor] | None, optional): Parameters of the model. Defaults to None.
            chains (Dict[str, torch.Tensor] | None, optional): Chains. Defaults to None.
        """
        if not isinstance(args, dict):
            args = vars(args)
            
        self.file_paths = file_paths
        self.tokens = tokens
        if params is not None:
            self.params = {key: value.clone() for key, value in params.items()}
        else:
            self.params = None
        if chains is not None:
            self.chains = chains.clone()
        else:
            self.chains = None
        self.max_epochs = args["nepochs"]
        self.updates = 0
        
        self.logs = {
            "Pearson": 0.0,
            "Slope": 0.0,
            "LL_train": 0.0,
            "LL_test": 0.0,
            "ESS": 0.0,
            "Entropy": 0.0,
            "Density": 0.0,
            "Time": 0.0,
        }
        
        template = "{0:<20} {1:<50}\n"  
        with open(file_paths["log"], "w") as f:
            if args["label"] is not None:
                f.write(template.format("label:", args["label"]))
            else:
                f.write(template.format("label:", "N/A"))
            
            f.write(template.format("model:", str(args["model"])))
            f.write(template.format("input MSA:", str(args["data"])))
            f.write(template.format("alphabet:", args["alphabet"]))
            f.write(template.format("sampler:", args["sampler"]))
            f.write(template.format("nchains:", args["nchains"]))
            f.write(template.format("nsweeps:", args["nsweeps"]))
            f.write(template.format("lr:", args["lr"]))
            f.write(template.format("pseudo count:", args["pseudocount"]))
            f.write(template.format("data type:", args["dtype"]))
            f.write(template.format("target Pearson Cij:", args["target"]))
            if args["model"] == "eaDCA":
                f.write(template.format("gsteps:", args["gsteps"]))
                f.write(template.format("factivate:", args["factivate"]))
            f.write(template.format("random seed:", args["seed"]))
            f.write("\n")
            # write the header of the log file
            header_string = " ".join([f"{key:<10}" for key in self.logs.keys()])
            f.write("{0:<10} {1}\n".format("Epoch", header_string))
        
        
    def log(
        self,
        key: str,
        value: Any,
    ) -> None:
        """Adds a key-value pair to the log dictionary

        Args:
            key (str): Key of the value.
            value (Any): Value to be stored.
        """
        if key not in self.logs.keys():
            raise ValueError(f"Key {key} not recognized.")
        
        if isinstance(value, torch.Tensor):
            self.logs[key] = value.item()
        else:
            self.logs[key] = value
        
    
    @abstractmethod
    def check(
        self,
        updates: int,
        *args,
        **kwargs,
    ) -> bool:
        """Checks if a checkpoint has been reached.
        
        Args:
            updates (int): Number of gradient updates performed.

        Returns:
            bool: Whether a checkpoint has been reached.
        """
        pass
        
        
    @abstractmethod 
    def save(
        self,
        params: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        chains: Dict[str, torch.Tensor],
        log_weights: torch.Tensor,
    ) -> None:
        """Saves the chains and the parameters of the model.

        Args:
            params (Dict[str, torch.Tensor]): Parameters of the model.
            mask (torch.Tensor): Mask of the model's coupling matrix representing the interaction graph
            chains (Dict[str, torch.Tensor]): Chains.
            log_weights (torch.Tensor): Log of the chain weights. Used for AIS.
        """
        pass
    
    
class LinearCheckpoint(Checkpoint):
    def __init__(
        self,
        file_paths: dict,
        tokens: str,
        args: dict,
        params: Dict[str, torch.Tensor] | None = None,
        chains: Dict[str, torch.Tensor] | None = None,
        checkpt_interval: int = 50,
        **kwargs,
    ):
        super().__init__(
            file_paths=file_paths,
            tokens=tokens,
            args=args,
            params=params,
            chains=chains,
        )
        self.checkpt_interval = checkpt_interval
        
    
    def check(
        self,
        updates: int,
        *args,
        **kwargs,
    ) -> bool:
        """Checks if a checkpoint has been reached.
        
        Args:
            updates (int): Number of gradient updates performed.

        Returns:
            bool: Whether a checkpoint has been reached.
        """
        self.updates = updates
        return (updates % self.checkpt_interval == 0) or (updates == self.max_epochs)
    
    
    def save(
        self,
        params: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        chains: Dict[str, torch.Tensor],
        log_weights: torch.Tensor,
    ) -> None:
        """Saves the chains and the parameters of the model.

        Args:
            params (Dict[str, torch.Tensor]): Parameters of the model.
            mask (torch.Tensor): Mask of the model's coupling matrix representing the interaction graph
            chains (Dict[str, torch.Tensor]): Chains.
            log_weights (torch.Tensor): Log of the chain weights. Used for AIS.
        """
        save_params(fname=self.file_paths["params"], params=params, mask=mask, tokens=self.tokens)
        save_chains(fname=self.file_paths["chains"], chains=chains.argmax(dim=-1), tokens=self.tokens, log_weights=log_weights)
        
        out_string = " ".join([f"{value:<10.3f}" for value in self.logs.values()])
        with open(self.file_paths["log"], "a") as f:
            f.write(f"{self.updates:<10} {out_string}\n")
            
            
class AcceptanceCheckpoint(Checkpoint):
    def __init__(
        self,
        file_paths: Dict,
        tokens: str,
        args: Dict,
        params: Dict[str, torch.Tensor] | None = None,
        chains: Dict[str, torch.Tensor] | None = None,
        target_acc_rate: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            file_paths=file_paths,
            tokens=tokens,
            args=args,
            params=params,
            chains=chains,
        )
        self.target_acc_rate = target_acc_rate
        self.num_saved_models = 0
        # Create a .h5 archive for storing the history of the parameters
        self.file_paths["params_history"] = self.file_paths["params"].with_suffix(".h5")
        with h5py.File(self.file_paths["params_history"], "w") as f:
            f["alphabet"] = self.tokens
            f.create_group(f"update_{self.updates}")
            for key, value in params.items():
                f[f"update_{self.updates}"].create_dataset(key, data=value.cpu().numpy())
        
    def check(
        self,
        updates: int,
        curr_params: Dict[str, torch.Tensor],
        curr_chains: Dict[str, torch.Tensor],
        *args,
        **kwargs,
    ) -> bool:
        """Checks if a checkpoint has been reached by computing the acceptance rate of swapping the 
        configurations of the present model and the last saved model.
        
        Args:
            updates (int): Number of gradient updates performed.
            curr_params (Dict[str, torch.Tensor]): Current parameters of the model.
            curr_chains (Dict[str, torch.Tensor]): Current chains of the model.

        Returns:
            bool: Whether a checkpoint has been reached.
        """
        acc_rate = _get_acceptance_rate(
            prev_params=self.params,
            curr_params=curr_params,
            prev_chains=self.chains,
            curr_chains=curr_chains,
        )
        self.updates = updates
        return (acc_rate < self.target_acc_rate) or (updates == self.max_epochs)
    
    
    def save(
        self,
        params: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        chains: Dict[str, torch.Tensor],
        log_weights: torch.Tensor,
        *args,
        **kwargs,
    ) -> None:
        """Saves the chains and the parameters of the model and appends the current parameters to the
        file containing the parameters history.

        Args:
            params (Dict[str, torch.Tensor]): Parameters of the model.
            mask (torch.Tensor): Mask of the model's coupling matrix representing the interaction graph.
            chains (Dict[str, torch.Tensor]): Chains.
            log_weights (torch.Tensor): Log of the chain weights. Used for AIS.
        """
        self.num_saved_models += 1
        # Store the current parameters and chains
        self.params = {key: value.clone() for key, value in params.items()}
        self.chains = chains.clone()
        # Append the current parameters to the history
        with h5py.File(self.file_paths["params_history"], "a") as f:
            f.create_group(f"update_{self.updates}")
            for key, value in params.items():
                f[f"update_{self.updates}"].create_dataset(key, data=value.cpu().numpy())
        # Save the current parameters and chains
        save_params(fname=self.file_paths["params"], params=params, mask=mask, tokens=self.tokens)
        save_chains(fname=self.file_paths["chains"], chains=chains.argmax(dim=-1), tokens=self.tokens, log_weights=log_weights)
        # Update the log file
        out_string = " ".join([f"{value:<10.3f}" for value in self.logs.values()])
        with open(self.file_paths["log"], "a") as f:
            f.write(f"{self.updates:<10} {out_string}\n")
        
            
def get_checkpoint(chpt: str) -> Checkpoint:
    if chpt == "linear":
        return LinearCheckpoint
    elif chpt == "acceptance":
        return AcceptanceCheckpoint
    else:
        raise ValueError(f"Checkpoint type {chpt} not recognized.")        