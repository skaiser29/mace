# pylint: disable=wrong-import-position
import argparse
import copy
import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import torch
from e3nn.util import jit

from mace.calculators import LAMMPS_MACE
from mace.calculators.lammps_mliap_mace import LAMMPS_MLIAP_MACE
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model to be converted to LAMMPS",
    )
    parser.add_argument(
        "--head",
        type=str,
        nargs="?",
        help="Head of the model to be converted to LAMMPS",
        default=None,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        nargs="?",
        help="Data type of the model to be converted to LAMMPS",
        default="float64",
    )
    parser.add_argument(
        "--format",
        type=str,
        help="Old libtorch format, or new mliap format",
        default="libtorch",
    )
    parser.add_argument(
        "--total-charge",
        type=float,
        default=0.0,
        help="Fixed total charge to embed in the exported LAMMPS wrapper",
    )
    parser.add_argument(
        "--total-spin",
        type=float,
        default=1.0,
        help="Fixed total spin to embed in the exported LAMMPS wrapper",
    )
    parser.add_argument(
        "--external-field",
        type=float,
        nargs=3,
        default=None,
        metavar=("EX", "EY", "EZ"),
        help="Fixed uniform external field to embed in the exported LAMMPS wrapper",
    )
    return parser.parse_args()


def select_head(model):
    if hasattr(model, "heads"):
        heads = model.heads
    else:
        heads = [None]

    if len(heads) == 1:
        print(f"Only one head found in the model: {heads[0]}. Skipping selection.")
        return heads[0]

    print("Available heads in the model:")
    for i, head in enumerate(heads):
        print(f"{i + 1}: {head}")

    # Ask the user to select a head
    selected = input(
        f"Select a head by number (Defaulting to head: {len(heads)}, press Enter to accept): "
    )

    if selected.isdigit() and 1 <= int(selected) <= len(heads):
        return heads[int(selected) - 1]
    if selected == "":
        print("No head selected. Proceeding without specifying a head.")
        return None
    print(f"No valid selection made. Defaulting to the last head: {heads[-1]}")
    return heads[-1]


def main():
    args = parse_args()
    model_path = args.model_path  # takes model name as command-line input
    model = torch.load(
        model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    if args.dtype == "float64":
        model = model.double().to("cpu")
    elif args.dtype == "float32":
        print("Converting model to float32, this may cause loss of precision.")
        model = model.float().to("cpu")

    if args.format == "mliap":
        # Enabling cuequivariance by default. TODO: switch?
        model = run_e3nn_to_cueq(copy.deepcopy(model))
        model.lammps_mliap = True

    if args.head is None:
        head = select_head(model)
    else:
        head = args.head
        print(
            f"Selected head: {head} from command line in the list available heads: {model.heads}"
        )

    lammps_class = LAMMPS_MLIAP_MACE if args.format == "mliap" else LAMMPS_MACE
    wrapper_kwargs = {
        "total_charge": torch.tensor([args.total_charge], dtype=model.r_max.dtype),
        "total_spin": torch.tensor([args.total_spin], dtype=model.r_max.dtype),
    }
    if args.external_field is not None:
        wrapper_kwargs["external_field"] = torch.tensor(
            [args.external_field], dtype=model.r_max.dtype
        )
    if head is not None:
        wrapper_kwargs["head"] = head
    lammps_model = lammps_class(model, **wrapper_kwargs)
    if args.format == "mliap":
        torch.save(lammps_model, model_path + "-mliap_lammps.pt")
    else:
        lammps_model_compiled = jit.compile(lammps_model)
        lammps_model_compiled.save(model_path + "-lammps.pt")


if __name__ == "__main__":
    main()
