import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols
from e3nn.util.jit import compile_mode
from mace import data as mace_data
from mace.tools import AtomicNumberTable, torch_geometric

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError:

    class MLIAPUnified:
        def __init__(self):
            pass


class MACELammpsConfig:
    """Configuration settings for MACE-LAMMPS integration."""

    def __init__(self):
        self.debug_time = self._get_env_bool("MACE_TIME", False)
        self.debug_profile = self._get_env_bool("MACE_PROFILE", False)
        self.debug_polar_batch = self._get_env_bool("MACE_DEBUG_POLAR_BATCH", False)
        self.profile_start_step = int(os.environ.get("MACE_PROFILE_START", "5"))
        self.profile_end_step = int(os.environ.get("MACE_PROFILE_END", "10"))
        self.allow_cpu = self._get_env_bool("MACE_ALLOW_CPU", False)
        self.force_cpu = self._get_env_bool("MACE_FORCE_CPU", False)

    @staticmethod
    def _get_env_bool(var_name: str, default: bool) -> bool:
        return os.environ.get(var_name, str(default)).lower() in (
            "true",
            "1",
            "t",
            "yes",
        )


@contextmanager
def timer(name: str, enabled: bool = True):
    """Context manager for timing code blocks."""
    if not enabled:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.info(f"Timer - {name}: {elapsed*1000:.3f} ms")


@compile_mode("script")
class MACEEdgeForcesWrapper(torch.nn.Module):
    """Wrapper that adds per-pair force computation to a MACE model."""

    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers)
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)
        self.register_buffer(
            "total_charge",
            kwargs.get(
                "total_charge", torch.tensor([0.0], dtype=torch.get_default_dtype())
            ),
        )
        self.register_buffer(
            "total_spin",
            kwargs.get(
                "total_spin", torch.tensor([1.0], dtype=torch.get_default_dtype())
            ),
        )

        if not hasattr(model, "heads"):
            model.heads = ["Default"]

        head_name = kwargs.get("head", model.heads[-1])
        head_idx = model.heads.index(head_name)
        self.register_buffer("head", torch.tensor([head_idx], dtype=torch.long))

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute energies and per-pair forces."""
        data["head"] = self.head
        data["total_charge"] = self.total_charge
        data["total_spin"] = self.total_spin

        out = self.model(
            data,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            compute_edge_forces=True,
            lammps_mliap=True,
        )

        node_energy = out["node_energy"]
        pair_forces = out["edge_forces"]
        total_energy = out["energy"][0]

        if pair_forces is None:
            pair_forces = torch.zeros_like(data["vectors"])

        return total_energy, node_energy, pair_forces


class LAMMPS_MLIAP_MACE(MLIAPUnified):
    """MACE integration for LAMMPS using the MLIAP interface."""

    def __init__(self, model, **kwargs):
        super().__init__()
        self.raw_model = model
        self.is_polar = model.__class__.__name__ == "PolarMACE"
        self.model = (
            model if self.is_polar else MACEEdgeForcesWrapper(model, **kwargs)
        )
        self.element_types = [chemical_symbols[s] for s in model.atomic_numbers]
        self.num_species = len(self.element_types)
        self.atomic_numbers = np.asarray(
            model.atomic_numbers.detach().cpu().numpy(), dtype=np.int64
        )
        self.z_table = AtomicNumberTable([int(z) for z in self.atomic_numbers])
        self.rcutfac = 0.5 * float(model.r_max)
        self.ndescriptors = 1
        self.nparams = 1
        self.dtype = model.r_max.dtype
        self.device = "cpu"
        self.total_charge = self._normalize_scalar_metadata(
            kwargs.get("total_charge", 0.0)
        )
        self.total_spin = self._normalize_scalar_metadata(kwargs.get("total_spin", 1.0))
        self.external_field = self._normalize_external_field(
            kwargs.get("external_field", None)
        )
        self.fermi_level = self._normalize_scalar_metadata(kwargs.get("fermi_level", 0.0))
        self.available_heads = getattr(model, "heads", ["Default"])
        self.head_name = kwargs.get("head", self.available_heads[-1])
        self.info_keys = {
            "total_spin": "spin",
            "total_charge": "charge",
            "external_field": "external_field",
        }
        self.initialized = False
        self.step = 0
        self._mpi_comm = None
        self._mpi_checked = False
        self._refresh_runtime_config()

        for p in self.raw_model.parameters():
            p.requires_grad = False

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._refresh_runtime_defaults()

    def _refresh_runtime_defaults(self):
        self._refresh_runtime_config()
        if not hasattr(self, "available_heads"):
            self.available_heads = getattr(self.raw_model, "heads", ["Default"])
        if not hasattr(self, "head_name"):
            self.head_name = self.available_heads[-1]
        if not hasattr(self, "info_keys"):
            self.info_keys = {
                "total_spin": "spin",
                "total_charge": "charge",
                "external_field": "external_field",
            }
        if not hasattr(self, "fermi_level"):
            self.fermi_level = self._normalize_scalar_metadata(0.0)
        if not hasattr(self, "_mpi_comm"):
            self._mpi_comm = None
        if not hasattr(self, "_mpi_checked"):
            self._mpi_checked = False

    def _refresh_runtime_config(self):
        self.config = MACELammpsConfig()

    def _normalize_scalar_metadata(self, value) -> torch.Tensor:
        if torch.is_tensor(value):
            value = value.detach().to(dtype=self.dtype).reshape(-1)
            if value.numel() == 1:
                return value
            raise ValueError("Expected scalar metadata tensor")
        return torch.tensor([float(value)], dtype=self.dtype)

    def _normalize_external_field(self, value) -> torch.Tensor:
        if value is None:
            return torch.zeros((1, 3), dtype=self.dtype)
        if torch.is_tensor(value):
            field = value.detach().to(dtype=self.dtype)
        else:
            field = torch.tensor(value, dtype=self.dtype)
        if field.shape == (3,):
            field = field.unsqueeze(0)
        if field.shape != (1, 3):
            raise ValueError("external_field must have shape (3,) or (1, 3)")
        return field

    def _get_mpi_comm(self):
        if self._mpi_checked:
            return self._mpi_comm
        self._mpi_checked = True
        try:
            from mpi4py import MPI
        except ImportError:
            world_size = int(
                os.environ.get(
                    "OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "1")
                )
            )
            if world_size > 1:
                raise ImportError(
                    "mpi4py is required for multi-rank POLAR MACE ML-IAP runs"
                )
            self._mpi_comm = None
            return self._mpi_comm
        self._mpi_comm = MPI.COMM_WORLD
        return self._mpi_comm

    def _initialize_device(self, data):
        using_kokkos = "kokkos" in data.__class__.__module__.lower()

        if using_kokkos and not self.config.force_cpu and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            if using_kokkos and not self.config.allow_cpu and not self.config.force_cpu:
                raise ValueError(
                    "GPU requested but CUDA is unavailable. Set MACE_ALLOW_CPU=true to allow CPU computation."
                )

        self.device = device
        self.model = self.model.to(device)
        self.raw_model = self.raw_model.to(device)
        self.total_charge = self.total_charge.to(device)
        self.total_spin = self.total_spin.to(device)
        self.external_field = self.external_field.to(device)
        self.fermi_level = self.fermi_level.to(device)
        logging.info(f"MACE model initialized on device: {device}")
        self.initialized = True

    def compute_forces(self, data):
        self._refresh_runtime_config()
        natoms = data.nlocal
        ntotal = data.ntotal
        nghosts = ntotal - natoms
        npairs = data.npairs
        species = torch.as_tensor(data.elems, dtype=torch.int64)

        if not self.initialized:
            self._initialize_device(data)

        self.step += 1
        self._manage_profiling()

        if natoms == 0:
            return

        with timer("total_step", enabled=self.config.debug_time):
            if self.is_polar:
                with timer("prepare_batch", enabled=self.config.debug_time):
                    batch, owned_indices = self._prepare_polar_batch(data, natoms)

                with timer("model_forward", enabled=self.config.debug_time):
                    out = self.model(
                        batch,
                        training=False,
                        compute_force=True,
                        compute_virials=False,
                        compute_stress=False,
                        compute_displacement=False,
                    )
                    atom_energies = out["node_energy"].index_select(0, owned_indices)
                    atom_forces = out["forces"].index_select(0, owned_indices)

                    if self.device.type != "cpu":
                        torch.cuda.synchronize()

                with timer("update_lammps", enabled=self.config.debug_time):
                    self._update_lammps_polar(data, atom_energies, atom_forces)
            else:
                if npairs <= 1:
                    return

                with timer("prepare_batch", enabled=self.config.debug_time):
                    batch = self._prepare_batch(data, natoms, nghosts, species)

                with timer("model_forward", enabled=self.config.debug_time):
                    _, atom_energies, pair_forces = self.model(batch)

                    if self.device.type != "cpu":
                        torch.cuda.synchronize()

                with timer("update_lammps", enabled=self.config.debug_time):
                    self._update_lammps_data(data, atom_energies, pair_forces, natoms)

    def _prepare_batch(self, data, natoms, nghosts, species):
        """Prepare the input batch for the MACE model."""
        return {
            "vectors": torch.as_tensor(data.rij).to(self.dtype).to(self.device),
            "node_attrs": torch.nn.functional.one_hot(
                species.to(self.device), num_classes=self.num_species
            ).to(self.dtype),
            "edge_index": torch.stack(
                [
                    torch.as_tensor(data.pair_j, dtype=torch.int64).to(self.device),
                    torch.as_tensor(data.pair_i, dtype=torch.int64).to(self.device),
                ],
                dim=0,
            ),
            "batch": torch.zeros(natoms, dtype=torch.int64, device=self.device),
            "lammps_class": data,
            "natoms": (natoms, nghosts),
        }

    def _gather_full_system(self, data, natoms):
        comm = self._get_mpi_comm()
        local_tags = np.asarray(data.owned_tags, dtype=np.int64)[:natoms]
        local_positions = np.asarray(data.owned_positions, dtype=np.float64)[:natoms]
        local_elems = np.asarray(data.owned_elems, dtype=np.int64)[:natoms]

        if comm is not None and comm.Get_size() > 1:
            all_tags = np.concatenate(comm.allgather(local_tags), axis=0)
            all_positions = np.concatenate(comm.allgather(local_positions), axis=0)
            all_elems = np.concatenate(comm.allgather(local_elems), axis=0)
        else:
            all_tags = local_tags.copy()
            all_positions = local_positions.copy()
            all_elems = local_elems.copy()

        order = np.argsort(all_tags, kind="stable")
        all_tags = all_tags[order]
        all_positions = all_positions[order]
        all_elems = all_elems[order]
        owned_indices = np.searchsorted(all_tags, local_tags)
        return all_tags, all_positions, all_elems, owned_indices

    def _prepare_polar_batch(self, data, natoms):
        _, all_positions, all_elems, owned_indices = self._gather_full_system(data, natoms)
        numbers = self.atomic_numbers[all_elems]

        atoms = Atoms(
            numbers=numbers,
            positions=all_positions,
            cell=np.asarray(data.cell, dtype=np.float64),
            pbc=np.asarray(data.pbc, dtype=bool),
        )
        atoms.info["charge"] = float(self.total_charge[0].detach().cpu().item())
        atoms.info["spin"] = float(self.total_spin[0].detach().cpu().item())
        atoms.info["external_field"] = (
            self.external_field.detach().cpu().numpy().astype(np.float64)
        )

        keyspec = mace_data.KeySpecification(info_keys=self.info_keys, arrays_keys={})
        config = mace_data.config_from_atoms(
            atoms, key_specification=keyspec, head_name=self.head_name
        )
        graph = mace_data.AtomicData.from_config(
            config,
            z_table=self.z_table,
            cutoff=self.raw_model.r_max.item(),
            heads=self.available_heads,
        )
        batch = torch_geometric.Batch.from_data_list([graph]).to(self.device)
        batch_dict = batch.to_dict()
        for key, value in list(batch_dict.items()):
            if torch.is_tensor(value) and torch.is_floating_point(value):
                batch_dict[key] = value.to(dtype=self.dtype)
        batch_dict["total_charge"] = self.total_charge.to(dtype=self.dtype)
        batch_dict["total_spin"] = self.total_spin.to(dtype=self.dtype)
        batch_dict["external_field"] = self.external_field.to(dtype=self.dtype)
        batch_dict["fermi_level"] = self.fermi_level.to(dtype=self.dtype)
        self._debug_polar_batch(data, all_elems, numbers, batch_dict)
        owned_indices_t = torch.as_tensor(
            owned_indices, dtype=torch.long, device=self.device
        )
        return batch_dict, owned_indices_t

    def _debug_polar_batch(self, data, all_elems, numbers, batch_dict):
        if not MACELammpsConfig._get_env_bool("MACE_DEBUG_POLAR_BATCH", False):
            return
        node_attrs = batch_dict.get("node_attrs")
        edge_index = batch_dict.get("edge_index")
        msg = [
            f"[MACE_DEBUG_POLAR_BATCH] natoms={data.nlocal} ntotal={data.ntotal} npairs={data.npairs}",
            f"[MACE_DEBUG_POLAR_BATCH] all_elems_unique={np.unique(all_elems).tolist()} atomic_numbers_unique={np.unique(numbers).tolist()}",
        ]
        if torch.is_tensor(node_attrs):
            node_attrs_cpu = node_attrs.detach().cpu()
            node_sums = node_attrs_cpu.sum(dim=1)
            msg.append(
                "[MACE_DEBUG_POLAR_BATCH] node_attrs "
                f"shape={tuple(node_attrs_cpu.shape)} dtype={node_attrs_cpu.dtype} "
                f"finite={bool(torch.isfinite(node_attrs_cpu).all().item())} "
                f"min={float(node_attrs_cpu.min().item())} max={float(node_attrs_cpu.max().item())} "
                f"row_sum_min={float(node_sums.min().item())} row_sum_max={float(node_sums.max().item())}"
            )
        if torch.is_tensor(edge_index):
            edge_index_cpu = edge_index.detach().cpu()
            msg.append(
                "[MACE_DEBUG_POLAR_BATCH] edge_index "
                f"shape={tuple(edge_index_cpu.shape)} min={int(edge_index_cpu.min().item())} "
                f"max={int(edge_index_cpu.max().item())}"
            )
        print("\n".join(msg), flush=True)

    def _update_lammps_data(self, data, atom_energies, pair_forces, natoms):
        """Update LAMMPS data structures with computed energies and forces."""
        if self.dtype == torch.float32:
            pair_forces = pair_forces.double()
        eatoms = torch.as_tensor(data.eatoms)
        atom_energies_real = atom_energies[:natoms].detach()
        eatoms.copy_(atom_energies_real)
        data.energy = atom_energies_real.sum().item()
        data.update_pair_forces(pair_forces)

    def _update_lammps_polar(self, data, atom_energies, atom_forces):
        atom_energies_real = atom_energies.detach().to(dtype=torch.float64)
        atom_forces_real = atom_forces.detach().to(dtype=torch.float64)
        force_array = data.f
        eatoms_array = data.eatoms
        if (
            self.device.type == "cpu"
            and force_array is not None
            and force_array.__class__.__module__.startswith("cupy")
        ):
            try:
                import cupy
            except ImportError as exc:
                raise RuntimeError(
                    "cupy is required to write CPU POLAR outputs back into Kokkos GPU arrays"
                ) from exc
            eatoms_array[: atom_energies_real.shape[0]] = cupy.asarray(
                atom_energies_real.cpu().numpy()
            )
            force_array[: atom_forces_real.shape[0], :] += cupy.asarray(
                atom_forces_real.cpu().numpy()
            )
            data.energy = float(atom_energies_real.sum().item())
            return
        data.update_atom_energy(atom_energies_real)
        data.update_atom_forces(atom_forces_real)

    def _manage_profiling(self):
        if not self.config.debug_profile:
            return

        if self.step == self.config.profile_start_step:
            logging.info(f"Starting CUDA profiler at step {self.step}")
            torch.cuda.profiler.start()

        if self.step == self.config.profile_end_step:
            logging.info(f"Stopping CUDA profiler at step {self.step}")
            torch.cuda.profiler.stop()
            logging.info("Profiling complete. Exiting.")
            sys.exit()

    def compute_descriptors(self, data):
        pass

    def compute_gradients(self, data):
        pass
