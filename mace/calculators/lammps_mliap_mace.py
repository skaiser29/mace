import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ase.data import chemical_symbols
from e3nn.util.jit import compile_mode
from matscipy.neighbours import neighbour_list

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


def _get_neighborhood(positions, cutoff, pbc, cell):
    pbc = tuple(bool(v) for v in np.asarray(pbc, dtype=bool).reshape(3).tolist())
    cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)

    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
    )

    true_self_edge = sender == receiver
    true_self_edge &= np.all(unit_shifts == 0, axis=1)
    keep_edge = ~true_self_edge

    sender = sender[keep_edge]
    receiver = receiver[keep_edge]
    unit_shifts = unit_shifts[keep_edge]

    edge_index = np.stack((sender, receiver))
    shifts = np.dot(unit_shifts, cell)
    return edge_index, shifts, unit_shifts, cell


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
        self.rcutfac = 0.5 * float(model.r_max)
        self.ndescriptors = 1
        self.nparams = 1
        self.dtype = model.r_max.dtype
        self.device = "cpu"
        self._polar_cutoff = float(model.r_max)
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
        self.initialized = False
        self.step = 0
        self._mpi_comm = None
        self._mpi_checked = False
        self._initialize_polar_cache_state()
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
        if not hasattr(self, "fermi_level"):
            self.fermi_level = self._normalize_scalar_metadata(0.0)
        if not hasattr(self, "_mpi_comm"):
            self._mpi_comm = None
        if not hasattr(self, "_mpi_checked"):
            self._mpi_checked = False
        if not hasattr(self, "_polar_cutoff"):
            self._polar_cutoff = float(self.raw_model.r_max)
        if not hasattr(self, "_polar_global_tags"):
            self._initialize_polar_cache_state()

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

    def _initialize_polar_cache_state(self):
        self._polar_global_tags = None
        self._polar_global_elems = None
        self._polar_node_attrs = None
        self._polar_batch_index = None
        self._polar_ptr = None
        self._polar_head = None
        self._polar_cell_cache_key = None
        self._polar_cell = None
        self._polar_rcell = None
        self._polar_volume = None
        self._polar_pbc = None

    def _invalidate_polar_device_cache(self):
        self._polar_node_attrs = None
        self._polar_batch_index = None
        self._polar_ptr = None
        self._polar_head = None
        self._polar_cell_cache_key = None
        self._polar_cell = None
        self._polar_rcell = None
        self._polar_volume = None
        self._polar_pbc = None

    def _rebuild_polar_global_layout(self, all_tags, all_elems):
        order = np.argsort(all_tags, kind="stable")
        global_tags = np.asarray(all_tags, dtype=np.int64)[order]
        global_elems = np.asarray(all_elems, dtype=np.int64)[order]
        if global_tags.size > 1 and np.any(global_tags[1:] == global_tags[:-1]):
            raise RuntimeError("Duplicate atom tags detected in POLAR MLIAP gather")
        self._polar_global_tags = global_tags
        self._polar_global_elems = global_elems
        self._invalidate_polar_device_cache()
        return order

    def _map_polar_tags(self, tags):
        tags = np.asarray(tags, dtype=np.int64)
        if self._polar_global_tags is None:
            raise RuntimeError("POLAR global tag layout is uninitialized")
        if tags.size == 0:
            return np.empty(0, dtype=np.int64)
        indices = np.searchsorted(self._polar_global_tags, tags)
        if np.any(indices >= self._polar_global_tags.shape[0]):
            raise KeyError("POLAR gather saw atom tags outside cached layout")
        if not np.array_equal(self._polar_global_tags[indices], tags):
            raise KeyError("POLAR gather saw atom tags not present in cached layout")
        return indices

    def _ensure_polar_static_tensors(self, num_atoms):
        needs_refresh = (
            self._polar_node_attrs is None
            or self._polar_node_attrs.shape[0] != num_atoms
            or self._polar_node_attrs.device != self.device
            or self._polar_node_attrs.dtype != self.dtype
        )
        if not needs_refresh:
            return
        elem_indices = torch.as_tensor(
            self._polar_global_elems, dtype=torch.long, device=self.device
        )
        self._polar_node_attrs = torch.nn.functional.one_hot(
            elem_indices, num_classes=self.num_species
        ).to(dtype=self.dtype)
        self._polar_batch_index = torch.zeros(
            num_atoms, dtype=torch.long, device=self.device
        )
        self._polar_ptr = torch.tensor([0, num_atoms], dtype=torch.long, device=self.device)
        head_index = self.available_heads.index(self.head_name)
        self._polar_head = torch.tensor([head_index], dtype=torch.long, device=self.device)

    def _get_polar_cell_tensors(self, cell_np, pbc_np):
        cell_np = np.asarray(cell_np, dtype=np.float64).reshape(3, 3)
        pbc_np = np.asarray(pbc_np, dtype=bool).reshape(3)
        key = (
            tuple(cell_np.reshape(-1).tolist()),
            tuple(bool(v) for v in pbc_np.tolist()),
            self.device.type,
            str(self.dtype),
        )
        if key != self._polar_cell_cache_key:
            cell = torch.as_tensor(cell_np, dtype=self.dtype, device=self.device).view(
                1, 3, 3
            )
            volume = torch.linalg.det(cell[0]).reshape(1)
            if float(torch.abs(volume[0]).item()) > 0.0:
                rcell = (2 * torch.pi * torch.linalg.inv(cell[0].mT)).reshape(1, 3, 3)
            else:
                rcell = torch.zeros((1, 3, 3), dtype=self.dtype, device=self.device)
            pbc = torch.as_tensor(pbc_np, dtype=torch.bool, device=self.device).view(
                1, 3
            )
            self._polar_cell_cache_key = key
            self._polar_cell = cell
            self._polar_rcell = rcell
            self._polar_volume = volume
            self._polar_pbc = pbc
        return self._polar_cell, self._polar_rcell, self._polar_volume, self._polar_pbc

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
        if self.is_polar:
            self._invalidate_polar_device_cache()
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
                comm = self._get_mpi_comm()
                multi_rank = comm is not None and comm.Get_size() > 1
                rank = comm.Get_rank() if multi_rank else 0

                with timer("prepare_batch", enabled=self.config.debug_time):
                    if multi_rank:
                        batch, owned_indices_per_rank = self._prepare_polar_batch_root(
                            data, natoms, comm, rank
                        )
                    else:
                        batch, owned_indices = self._prepare_polar_batch(data, natoms)

                with timer("model_forward", enabled=self.config.debug_time):
                    if multi_rank:
                        if rank == 0:
                            out = self.model(
                                batch,
                                training=False,
                                compute_force=True,
                                compute_virials=True,
                                compute_stress=False,
                                compute_displacement=False,
                            )
                            atom_energies, atom_forces = self._scatter_polar_outputs(
                                comm,
                                rank,
                                out["node_energy"],
                                out["forces"],
                                owned_indices_per_rank,
                            )
                            global_virial = self._extract_global_polar_virial(out)
                        else:
                            atom_energies, atom_forces = self._scatter_polar_outputs(
                                comm, rank, None, None, None
                            )
                            global_virial = None
                    else:
                        out = self.model(
                            batch,
                            training=False,
                            compute_force=True,
                            compute_virials=True,
                            compute_stress=False,
                            compute_displacement=False,
                        )
                        atom_energies = out["node_energy"].index_select(0, owned_indices)
                        atom_forces = out["forces"].index_select(0, owned_indices)
                        global_virial = self._extract_global_polar_virial(out)

                    if self.device.type != "cpu" and (
                        self.config.debug_time or self.config.debug_profile
                    ):
                        torch.cuda.synchronize()

                with timer("update_lammps", enabled=self.config.debug_time):
                    self._update_lammps_polar(
                        data, atom_energies, atom_forces, global_virial
                    )
            else:
                if npairs <= 1:
                    return

                with timer("prepare_batch", enabled=self.config.debug_time):
                    batch = self._prepare_batch(data, natoms, nghosts, species)

                with timer("model_forward", enabled=self.config.debug_time):
                    _, atom_energies, pair_forces = self.model(batch)

                    if self.device.type != "cpu" and (
                        self.config.debug_time or self.config.debug_profile
                    ):
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
        local_tags = np.asarray(data.owned_tags, dtype=np.int64)[:natoms].copy()
        local_positions = np.asarray(data.owned_positions, dtype=np.float64)[
            :natoms
        ].copy()
        need_layout = (
            self._polar_global_tags is None
            or self._polar_global_elems is None
        )

        if comm is not None and comm.Get_size() > 1:
            gathered_tags = np.concatenate(comm.allgather(local_tags), axis=0)
            gathered_positions = np.concatenate(comm.allgather(local_positions), axis=0)
        else:
            gathered_tags = local_tags
            gathered_positions = local_positions

        if need_layout or gathered_tags.shape[0] != self._polar_global_tags.shape[0]:
            local_elems = np.asarray(data.owned_elems, dtype=np.int64)[:natoms].copy()
            if comm is not None and comm.Get_size() > 1:
                gathered_elems = np.concatenate(comm.allgather(local_elems), axis=0)
            else:
                gathered_elems = local_elems
            order = self._rebuild_polar_global_layout(gathered_tags, gathered_elems)
            all_positions = gathered_positions[order]
        else:
            try:
                global_indices = self._map_polar_tags(gathered_tags)
            except KeyError:
                local_elems = np.asarray(data.owned_elems, dtype=np.int64)[:natoms].copy()
                if comm is not None and comm.Get_size() > 1:
                    gathered_elems = np.concatenate(comm.allgather(local_elems), axis=0)
                else:
                    gathered_elems = local_elems
                order = self._rebuild_polar_global_layout(gathered_tags, gathered_elems)
                all_positions = gathered_positions[order]
            else:
                all_positions = np.empty(
                    (self._polar_global_tags.shape[0], 3), dtype=np.float64
                )
                all_positions[global_indices] = gathered_positions

        owned_indices = self._map_polar_tags(local_tags)
        return all_positions, self._polar_global_elems, owned_indices

    def _gather_full_system_root(self, data, natoms, comm, rank):
        local_tags = np.asarray(data.owned_tags, dtype=np.int64)[:natoms].copy()
        local_positions = np.asarray(data.owned_positions, dtype=np.float64)[
            :natoms
        ].copy()
        local_elems = np.asarray(data.owned_elems, dtype=np.int64)[:natoms].copy()

        if comm is not None and comm.Get_size() > 1:
            gathered_tags = comm.gather(local_tags, root=0)
            gathered_positions = comm.gather(local_positions, root=0)
            gathered_elems = comm.gather(local_elems, root=0)
            if rank != 0:
                return None, None, None
        else:
            gathered_tags = [local_tags]
            gathered_positions = [local_positions]
            gathered_elems = [local_elems]

        flat_tags = np.concatenate(gathered_tags, axis=0)
        flat_positions = np.concatenate(gathered_positions, axis=0)
        flat_elems = np.concatenate(gathered_elems, axis=0)

        need_layout = (
            self._polar_global_tags is None
            or self._polar_global_elems is None
            or flat_tags.shape[0] != self._polar_global_tags.shape[0]
        )
        if need_layout:
            order = self._rebuild_polar_global_layout(flat_tags, flat_elems)
            all_positions = flat_positions[order]
        else:
            try:
                global_indices = self._map_polar_tags(flat_tags)
            except KeyError:
                order = self._rebuild_polar_global_layout(flat_tags, flat_elems)
                all_positions = flat_positions[order]
            else:
                all_positions = np.empty(
                    (self._polar_global_tags.shape[0], 3), dtype=np.float64
                )
                all_positions[global_indices] = flat_positions

        owned_indices_per_rank = [
            self._map_polar_tags(tags) for tags in gathered_tags
        ]
        return all_positions, self._polar_global_elems, owned_indices_per_rank

    def _prepare_polar_batch(self, data, natoms):
        all_positions, all_elems, owned_indices = self._gather_full_system(data, natoms)
        cell_np = np.asarray(data.cell, dtype=np.float64).reshape(3, 3)
        pbc_np = np.asarray(data.pbc, dtype=bool).reshape(3)
        edge_index, shifts, unit_shifts, graph_cell = _get_neighborhood(
            positions=all_positions,
            cutoff=self._polar_cutoff,
            pbc=tuple(bool(v) for v in pbc_np.tolist()),
            cell=cell_np.copy(),
        )
        self._ensure_polar_static_tensors(all_positions.shape[0])
        cell, rcell, volume, pbc = self._get_polar_cell_tensors(graph_cell, pbc_np)
        batch_dict = {
            "edge_index": torch.as_tensor(
                edge_index, dtype=torch.long, device=self.device
            ),
            "positions": torch.as_tensor(
                all_positions, dtype=self.dtype, device=self.device
            ),
            "shifts": torch.as_tensor(shifts, dtype=self.dtype, device=self.device),
            "unit_shifts": torch.as_tensor(
                unit_shifts, dtype=self.dtype, device=self.device
            ),
            "cell": cell,
            "node_attrs": self._polar_node_attrs,
            "batch": self._polar_batch_index,
            "ptr": self._polar_ptr,
            "head": self._polar_head,
            "pbc": pbc,
            "rcell": rcell,
            "volume": volume,
            "total_charge": self.total_charge.to(dtype=self.dtype),
            "total_spin": self.total_spin.to(dtype=self.dtype),
            "external_field": self.external_field.to(dtype=self.dtype),
            "fermi_level": self.fermi_level.to(dtype=self.dtype),
        }
        numbers = self.atomic_numbers[all_elems]
        self._debug_polar_batch(data, all_elems, numbers, batch_dict)
        owned_indices_t = torch.as_tensor(
            owned_indices, dtype=torch.long, device=self.device
        )
        return batch_dict, owned_indices_t

    def _prepare_polar_batch_root(self, data, natoms, comm, rank):
        all_positions, all_elems, owned_indices_per_rank = self._gather_full_system_root(
            data, natoms, comm, rank
        )
        if rank != 0:
            return None, None

        cell_np = np.asarray(data.cell, dtype=np.float64).reshape(3, 3)
        pbc_np = np.asarray(data.pbc, dtype=bool).reshape(3)
        edge_index, shifts, unit_shifts, graph_cell = _get_neighborhood(
            positions=all_positions,
            cutoff=self._polar_cutoff,
            pbc=tuple(bool(v) for v in pbc_np.tolist()),
            cell=cell_np.copy(),
        )
        self._ensure_polar_static_tensors(all_positions.shape[0])
        cell, rcell, volume, pbc = self._get_polar_cell_tensors(graph_cell, pbc_np)
        batch_dict = {
            "edge_index": torch.as_tensor(
                edge_index, dtype=torch.long, device=self.device
            ),
            "positions": torch.as_tensor(
                all_positions, dtype=self.dtype, device=self.device
            ),
            "shifts": torch.as_tensor(shifts, dtype=self.dtype, device=self.device),
            "unit_shifts": torch.as_tensor(
                unit_shifts, dtype=self.dtype, device=self.device
            ),
            "cell": cell,
            "node_attrs": self._polar_node_attrs,
            "batch": self._polar_batch_index,
            "ptr": self._polar_ptr,
            "head": self._polar_head,
            "pbc": pbc,
            "rcell": rcell,
            "volume": volume,
            "total_charge": self.total_charge.to(dtype=self.dtype),
            "total_spin": self.total_spin.to(dtype=self.dtype),
            "external_field": self.external_field.to(dtype=self.dtype),
            "fermi_level": self.fermi_level.to(dtype=self.dtype),
        }
        numbers = self.atomic_numbers[all_elems]
        self._debug_polar_batch(data, all_elems, numbers, batch_dict)
        return batch_dict, owned_indices_per_rank

    def _scatter_polar_outputs(
        self,
        comm,
        rank: int,
        node_energy: Optional[torch.Tensor],
        forces: Optional[torch.Tensor],
        owned_indices_per_rank: Optional[List[np.ndarray]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if comm is None or comm.Get_size() <= 1:
            if node_energy is None or forces is None or owned_indices_per_rank is None:
                raise RuntimeError("Missing POLAR outputs for single-rank scatter")
            owned_indices_t = torch.as_tensor(
                owned_indices_per_rank[0], dtype=torch.long, device=self.device
            )
            return (
                node_energy.index_select(0, owned_indices_t),
                forces.index_select(0, owned_indices_t),
            )

        payload = None
        if rank == 0:
            if node_energy is None or forces is None or owned_indices_per_rank is None:
                raise RuntimeError("Root rank is missing POLAR forward outputs")
            node_energy_np = node_energy.detach().to(dtype=torch.float64).cpu().numpy()
            forces_np = forces.detach().to(dtype=torch.float64).cpu().numpy()
            payload = [
                (node_energy_np[indices].copy(), forces_np[indices].copy())
                for indices in owned_indices_per_rank
            ]
        local_energy_np, local_forces_np = comm.scatter(payload, root=0)
        local_energy = torch.as_tensor(
            local_energy_np, dtype=self.dtype, device=self.device
        )
        local_forces = torch.as_tensor(
            local_forces_np, dtype=self.dtype, device=self.device
        )
        return local_energy, local_forces

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

    def _extract_global_polar_virial(self, out):
        virials = out.get("virials")
        if virials is None:
            return None
        virials = virials.detach().to(dtype=torch.float64)
        if virials.ndim == 3:
            if virials.shape[0] != 1:
                raise ValueError(
                    f"Expected one graph virial tensor, got shape {tuple(virials.shape)}"
                )
            virials = virials[0]
        elif virials.ndim != 2:
            raise ValueError(f"Unexpected virial tensor shape {tuple(virials.shape)}")
        virials = 0.5 * (virials + virials.transpose(0, 1))
        return torch.stack(
            [
                virials[0, 0],
                virials[1, 1],
                virials[2, 2],
                virials[0, 1],
                virials[0, 2],
                virials[1, 2],
            ]
        )

    def _update_lammps_polar(
        self, data, atom_energies, atom_forces, global_virial=None
    ):
        atom_energies_real = atom_energies.detach().to(dtype=torch.float64)
        atom_forces_real = atom_forces.detach().to(dtype=torch.float64)
        global_virial_host = None
        if global_virial is not None:
            if torch.is_tensor(global_virial):
                global_virial_host = global_virial.detach().cpu().numpy()
            else:
                global_virial_host = np.asarray(global_virial, dtype=np.float64)
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
            if global_virial_host is not None and hasattr(data, "update_global_virial"):
                data.update_global_virial(global_virial_host)
            return
        data.update_atom_energy(atom_energies_real)
        data.update_atom_forces(atom_forces_real)
        if global_virial_host is not None and hasattr(data, "update_global_virial"):
            data.update_global_virial(global_virial_host)

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
