import argparse
from pathlib import Path
import numpy as np

import torch

from adabmDCA.fasta_utils import get_tokens
from adabmDCA.utils import get_device
from adabmDCA.io import load_params, import_from_fasta
from adabmDCA.fasta_utils import encode_sequence
from adabmDCA.functional import one_hot
from adabmDCA.statmech import compute_energy
from adabmDCA.parser import add_args_energies


# import command-line input arguments
def create_parser():
    parser = argparse.ArgumentParser(description="Computes the DCA energies of a sequence dataset.")
    parser = add_args_energies(parser)
    
    return parser


def main():     
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Set the device
    device = get_device(args.device)
    
    # import data
    tokens = get_tokens(args.alphabet)
    names, sequences = import_from_fasta(args.data)
    data = np.vectorize(encode_sequence, excluded=["tokens"], signature="(), () -> (n)")(sequences, tokens)
    data = torch.tensor(data, device=device)
    
    # import parameters and compute DCA energies
    params = load_params(args.path_params, tokens=tokens, device=device)
    q = params["bias"].shape[1]
    data = one_hot(data, num_classes=q)
    energies = compute_energy(data, params).cpu().numpy()
    
    # Save results in a file
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    fname_out = folder / Path(args.data.stem + "_energies.fasta")
    with open(fname_out, "w") as f:
        for n, s, e in zip(names, sequences, energies):
            f.write(f">{n} | DCAenergy: {e:.3f}\n")
            f.write(f"{s}\n")
    
    print(f"Process completed. Output saved in {fname_out}")
    
    
if __name__ == "__main__":
    main()