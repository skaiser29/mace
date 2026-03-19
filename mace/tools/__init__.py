"""Lazy tool exports.

Avoid importing training/foundation utilities during lightweight runtime use
cases such as embedded LAMMPS model loading.
"""

from importlib import import_module

__all__ = [
    "TensorDict",
    "AtomicNumberTable",
    "atomic_numbers_to_indices",
    "to_numpy",
    "to_one_hot",
    "build_default_arg_parser",
    "check_args",
    "DefaultKeys",
    "set_seeds",
    "init_device",
    "setup_logger",
    "get_tag",
    "count_parameters",
    "MetricsLogger",
    "get_atomic_number_table_from_zs",
    "train",
    "evaluate",
    "SWAContainer",
    "CheckpointHandler",
    "CheckpointIO",
    "CheckpointState",
    "set_default_dtype",
    "compute_mae",
    "compute_rel_mae",
    "compute_rmse",
    "compute_rel_rmse",
    "compute_q95",
    "compute_c",
    "U_matrix_real",
    "spherical_to_cartesian",
    "cartesian_to_spherical",
    "voigt_to_matrix",
    "init_wandb",
    "load_foundations",
    "load_foundations_elements",
    "build_preprocess_arg_parser",
]

_ATTR_MODULES = {
    "build_default_arg_parser": ".arg_parser",
    "build_preprocess_arg_parser": ".arg_parser",
    "check_args": ".arg_parser_tools",
    "U_matrix_real": ".cg",
    "CheckpointHandler": ".checkpoint",
    "CheckpointIO": ".checkpoint",
    "CheckpointState": ".checkpoint",
    "DefaultKeys": ".default_keys",
    "load_foundations": ".finetuning_utils",
    "load_foundations_elements": ".finetuning_utils",
    "TensorDict": ".torch_tools",
    "cartesian_to_spherical": ".torch_tools",
    "count_parameters": ".torch_tools",
    "init_device": ".torch_tools",
    "init_wandb": ".torch_tools",
    "set_default_dtype": ".torch_tools",
    "set_seeds": ".torch_tools",
    "spherical_to_cartesian": ".torch_tools",
    "to_numpy": ".torch_tools",
    "to_one_hot": ".torch_tools",
    "voigt_to_matrix": ".torch_tools",
    "SWAContainer": ".train",
    "evaluate": ".train",
    "train": ".train",
    "AtomicNumberTable": ".utils",
    "MetricsLogger": ".utils",
    "atomic_numbers_to_indices": ".utils",
    "compute_c": ".utils",
    "compute_mae": ".utils",
    "compute_q95": ".utils",
    "compute_rel_mae": ".utils",
    "compute_rel_rmse": ".utils",
    "compute_rmse": ".utils",
    "get_atomic_number_table_from_zs": ".utils",
    "get_tag": ".utils",
    "setup_logger": ".utils",
}

_MODULE_EXPORTS = {
    "torch_geometric": ".torch_geometric",
    "torch_tools": ".torch_tools",
    "utils": ".utils",
}


def __getattr__(name):
    module_name = _ATTR_MODULES.get(name)
    if module_name is not None:
        return getattr(import_module(module_name, __name__), name)
    module_name = _MODULE_EXPORTS.get(name)
    if module_name is not None:
        return import_module(module_name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
