from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipy.constants as cst
from monty.io import zopen
from scipy.stats import norm

from pymatgen.core import Composition, Element, Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.core import Spin
from pymatgen.util.coord import get_angle
from pymatgen.util.plotting import pretty_plot
from pymatgen.core.trajectory import Trajectory

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from typing_extensions import Self

    from pymatgen.core.structure import Structure
    from pymatgen.core.trajectory import Trajectory
    from pymatgen.util.typing import PathLike

def write_Gaussian_vib_log(
    struc: Structure,
    filename: PathLike = "calc.log",
    site_property_map: dict[str, str] | None = None,
) -> None:
    """Write Trajectory object to a Gaussian-formatted log file.

    The produced file will be readable by Gaussview.

    Args:
        filename: File to write. File does not need to end with '.log' to be readable by Gaussview.
            The suffix ".logx" is recommended to distinguish from real Gaussian log files.
        site_property_map: A mapping between implemented Gaussian properties and pymatgen properties.
            i.e. providing {"mulliken charges": "charges"} will write the data stored in
            `Trajectory[i].site_properties["charges"]` as Gaussian logs the Mulliken charges for each site.
            The currently implemented property keys are "mulliken charges", "mulliken spin", "nbo charges",
            "nbo spin", and "esp charges". By default all charge keys are mapped to "charges", and all spin
            keys are mapped to "magmom". All keys must map to site properties of `list[float]`
        write_lattice: Whether to write the lattice information in the log file. Setting this to false will
            not write the lattice information in the log file, but will enable the "forces" site properties
            to be readable by Gaussview.
    """
    # Import inside function as this is an expensive import
    # (causes import time for Trajectory to surpass threshold for 3 test setups)
    # from pymatgen.io.gaussian import _traj_to_gaussian_log
    # struc.to_positions()
    _sp_map = site_property_map or {}
    lines = _struc_to_gaussian_vib_log(struc, _sp_map)
    with open(filename, "w") as file:
        file.write(lines)
    file.close()

def _struc_to_gaussian_vib_log(struc: Structure, sp_map: dict | None) -> str:
    write_traj = _struc_to_vib_traj(struc)
    vib_idx_offsets = [i * (3 * len(struc) -3) for i in range(len(write_traj))]
    return _traj_to_gaussian_log(write_traj, False, [0]*len(write_traj), sp_map, vib_idx_offsets = vib_idx_offsets)


def _struc_to_vib_traj(struc: Structure) -> Trajectory:
    frame = struc
    nModes = len(frame.properties["vibrational_modes"])
    nAtoms = len(frame)
    nModesPerFrame = 3*nAtoms - 3
    nFramesTot = int(np.ceil(nModes / nModesPerFrame))
    base_properties = frame.properties.copy()
    all_modes = base_properties.pop("vibrational_modes")
    base_frame = frame.copy()
    frames = []
    for iFrame in range(nFramesTot):
        start_idx = iFrame * nModesPerFrame
        end_idx = min(start_idx + nModesPerFrame, nModes)
        frame = base_frame.copy()
        frame.properties = base_properties.copy()
        frame.properties["vibrational_modes"] = all_modes[start_idx:end_idx]
        frames.append(frame)
    write_traj = Trajectory.from_structures(
                structures=frames,
                frame_properties=[frame.properties for frame in frames],
            )
    return write_traj

def _traj_to_gaussian_log(traj: Trajectory, do_cell: bool, energies: list[float], sp_map: dict | None, vib_idx_offsets: None | list[int] = None) -> str:
    dump_lines = ["", " Entering Link 1 ", " ", *_log_input_orientation(traj[0], do_cell=do_cell)]
    if vib_idx_offsets is None:
        vib_idx_offsets = [0] * len(traj)
    for i in range(len(traj)):
        dump_lines += [
            *[*_log_input_orientation(traj[i], do_cell=do_cell), f"SCF Done:  E =  {energies[i]:.6f}  "],
            *_log_mulliken(traj[i], "charges", "spin"),
            *_log_esp(traj[i], "charges"),
            *_log_nbo(traj[i], "charges", "spin"),
            *_log_vibrations(traj[i], do_cell=do_cell, idx_offset=vib_idx_offsets[i]),
            *[*_log_forces(traj[i]), "", " " + "Grad" * 18, "", f" Step number {i + 1}", "", " " + "Grad" * 18],
        ]
    dump_lines[-2:-2] = [" Optimization completed.", "    -- Stationary point found."]
    dump_lines += [*_log_input_orientation(traj[-1], do_cell=do_cell), " Normal termination of Gaussian 16"]
    return "\n".join(dump_lines)






def _traj_to_gaussian_log(traj: Trajectory, do_cell: bool, energies: list[float], sp_map: dict | None, vib_idx_offsets: None | list[int] = None) -> str:
    dump_lines = ["", " Entering Link 1 ", " ", *_log_input_orientation(traj[0], do_cell=do_cell)]
    if vib_idx_offsets is None:
        vib_idx_offsets = [0] * len(traj)
    for i in range(len(traj)):
        dump_lines += [
            *[*_log_input_orientation(traj[i], do_cell=do_cell), f"SCF Done:  E =  {energies[i]:.6f}  "],
            *_log_mulliken(traj[i], "charges", "spin"),
            *_log_esp(traj[i], "charges"),
            *_log_nbo(traj[i], "charges", "spin"),
            *_log_vibrations(traj[i], do_cell=do_cell, idx_offset=vib_idx_offsets[i]),
            *[*_log_forces(traj[i]), "", " " + "Grad" * 18, "", f" Step number {i + 1}", "", " " + "Grad" * 18],
        ]
    dump_lines[-2:-2] = [" Optimization completed.", "    -- Stationary point found."]
    dump_lines += [*_log_input_orientation(traj[-1], do_cell=do_cell), " Normal termination of Gaussian 16"]
    return "\n".join(dump_lines)


def _log_input_orientation(frame: Structure | Molecule, do_cell=True) -> list[str]:
    at_ns = frame.atomic_numbers
    dump_lines = [
        *[" " * 24 + "Standard orientation:" + " " * 26, " " + "-" * 69],
        " Center     Atomic      Atomic             Coordinates (Angstroms)",
        *[" Number     Number       Type             X           Y           Z", " " + "-" * 69],
        *[f"{i + 1} {at_ns[i]} 0 {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} " for i, p in enumerate(frame.cart_coords)],
    ]
    if do_cell:
        cell = frame.lattice.matrix
        dump_lines += [f"{i} -2 0 {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} " for i, v in enumerate(cell, start=len(at_ns) + 1)]
    dump_lines.append(" ---------------------------------------------------------------------")
    return dump_lines


def _log_mulliken(frame: Structure | Molecule, charge_key: str, spin_key: str) -> list[str]:
    dump_lines = []
    if charge_key in frame.site_properties:
        charges = np.array(frame.site_properties[charge_key])
        dump_lines += ["Mulliken charges:", "1"]
        spins: None | NDArray = None
        if spin_key in frame.site_properties:
            spins = np.array(frame.site_properties[spin_key])
            dump_lines[-2] = dump_lines[-2].replace(":", " and spin densities:")
            dump_lines[-1] += " 2"
        for i, charge in enumerate(charges):
            dump_lines.append(f"{i + 1} {frame.symbol_set[i]} {charge} " + "" or f"{spins[i]} ")
        dump_lines.append(f"Sum of Mulliken charges = {np.sum(charges)}" + "" if spins is None else f" {np.sum(spins)}")
    return dump_lines


def _log_esp(frame: Structure | Molecule, charge_key: str) -> list[str]:
    dump_lines = []
    if charge_key in frame.site_properties:
        charges = np.array(frame.site_properties[charge_key])
        dump_lines += ["", " Charges from ESP fit", "  ESP charges:", "               1"]
        dump_lines += [f"     {i + 1} {s} {charges[i]:.6f}" for i, s in enumerate(frame.symbol_set)]
        dipole = np.sum(frame.cart_coords * charges[:, np.newaxis], axis=0)
        tot = np.linalg.norm(dipole)
        dump_lines.append(f" Charge= {np.sum(charges)} Dipole= {' '.join(str(v) for v in dipole)} Tot=   {tot}")
    return dump_lines


def _log_nbo_fields(frame: Structure | Molecule, charges: NDArray, totals: NDArray | None = None) -> list[str]:
    _totals = totals if isinstance(totals, np.ndarray) else np.zeros_like(charges)
    dump_lines = [
        *[" Summary of Natural Population Analysis:  ", "", " " * 39 + "Natural Population "],
        " " * 17 + "Natural " + "-" * 47 + " ",
        *["    Atom  No    Charge         Core      Valence    Rydberg      Total", " " + "-" * 71],
    ]
    for i, s in enumerate(frame.symbol_set):
        # Values for core, valence, and rydberg fields don't affect anything in gview.
        dump_lines.append(f"{s} {i + 1} {charges[i]:.6f} {0.0:.6f} {0.0:.6f} {0.0:.6f} {_totals[i]:.6f}")
    dump_lines += [
        *[" " + "-" * 71, f"   * Total *   {np.sum(charges):.6f} " + f"{0.0:.6f} " * 3 + f"{np.sum(_totals):.6f}"],
    ]
    return dump_lines


def _log_nbo(frame: Structure | Molecule, charge_key: str, spin_key: str) -> list[str]:
    dump_lines = []
    if charge_key in frame.site_properties:
        nbo_charges = np.array(frame.site_properties[charge_key])
        dump_lines += _log_nbo_fields(frame, nbo_charges)
        if spin_key in frame.site_properties:  # NBO spins read as difference between alpha/beta totals
            spins = np.array(frame.site_properties[spin_key])
            alpha_spins = np.maximum(np.zeros(len(spins)), spins)
            beta_spins = np.abs(np.minimum(np.zeros(len(spins)), spins))
            for spin_type, spin_arr in zip(("Alpha", "Beta"), (alpha_spins, beta_spins), strict=False):
                dump_lines += [f" {spin_type} spin orbitals "]
                dump_lines += _log_nbo_fields(frame, np.zeros_like(spin_arr), totals=spin_arr)
    return dump_lines


def _log_forces(frame: Structure | Molecule, do_cell: bool = True) -> list[str]:
    # TODO: Find the trick for getting Gaussview to read forces on PBC calculations
    dump_lines = []
    if "forces" in frame.site_properties:
        forces = np.array(frame.site_properties["forces"])
        dump_lines = [
            *["-" * 67, " Center     Atomic" + " " * 19 + "Forces (Hartrees/Bohr)"],
            *[" Number     " + (" " * 14).join(["Number", "X", "Y", "Z"]), "-" * 67],
        ]
        for i, n in enumerate(frame.atomic_numbers):
            dump_lines.append(f" {i + 1} {n}\t" + "\t".join(f"{forces[i][j]:.9f}" for j in range(3)))
        if do_cell:
            dump_lines += [" " * 25 + "-2" + " " * 10 + "   ".join(f"{0.0:.9f}" for j in range(3)) for i in range(3)]
        nforces = np.linalg.norm(forces, axis=1)
        dump_lines += [" " + "-" * 67, f" Cartesian Forces:  Max {max(nforces):.9f} RMS {np.std(nforces):.9f}"]
    return dump_lines


def _log_vib_helper(modes: list[dict], idx_offset: int = 0):
    frequencies = []
    IR_intensities = []
    disp_str = "Displacements"
    freq_str = "cm^-1"
    ir_str = "IR intensity"
    nModes = len(modes)
    nAtoms = len(modes[0].get(disp_str, [])) if nModes > 0 else 0
    displacements = np.zeros((nModes, nAtoms, 3))
    for i, mode in enumerate(modes):
        freq = mode.get(freq_str, 0.0)
        if np.imag(freq) != 0.0:
            freq = np.abs(freq) * -1.0
        frequencies.append(freq)
        ir = mode.get(ir_str, 0.0)
        IR_intensities.append(ir)
        disp = mode.get(disp_str, [])
        if not len(disp):
            raise ValueError(f"No displacements found in mode {i} ({mode})")
        # add atom displacements individually in case of missing atoms
        for j, atom_disp in enumerate(disp):
            displacements[i, j] += np.array(atom_disp)
        displacements[i] /= np.linalg.norm(displacements[i])
    # log_str = " ".join([f"{i+1}" for i in range(nModes)]) + " \n"
    log_lines = [
        " Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering",
        " activities (A**4/AMU), depolarization ratios for plane and unpolarized",
        " incident light, reduced masses (AMU), force constants (mDyne/A),",
        " and normal coordinates:"]
    # nSlices = nModes // 3
    nSlices = int(np.ceil(nModes / 3))
    for iSlice in range(nSlices):
        start_idx = iSlice * 3
        end_idx = min(start_idx + 3, nModes)
        mod_idcs = list(range(start_idx + 1 + idx_offset, idx_offset + min(end_idx + 1, nModes + 1)))
        displacements_slice = displacements[start_idx:end_idx, :, :]
        freqs_slice = frequencies[start_idx:end_idx]
        ir_intensities_slice = IR_intensities[start_idx:end_idx]
        log_lines += _log_vib_helper_helper(displacements_slice, mod_idcs, freqs_slice, ir_intensities_slice, nAtoms)
    return log_lines

def _log_vib_helper_helper(displacements_slice: NDArray[np.float64], mod_idcs: list[int], freqs: list[float], ir_intensities: list[float], nAtoms: int):
    nModes = len(mod_idcs)
    log_lines = [
        " ".join([f"{idx}" for idx in mod_idcs]),
        " ".join([f"A1" for _ in range(nModes)]),
        " Frequencies -- " + " ".join([f"{f:.4f}" for f in freqs]),
        " IR Inten    -- " + " ".join([f"{ir:.4f}" for ir in ir_intensities]),
        "  Atom  AN " + " ".join(["X", "Y", "Z "])* len(mod_idcs)
        ]
    for iAtom in range(nAtoms):
        idisps = displacements_slice[:, iAtom, :]
        log_lines.append(f"     {iAtom+1}   1 " + " ".join(" ".join([f"{idisps[i][j]:.2f}" for j in range(3)]) + " " for i in range(nModes)))
    return log_lines



def _log_vibrations(frame: Structure | Molecule, do_cell=True, idx_offset: int = 0) -> list[str]:
    if do_cell or (not "vibrational_modes" in frame.properties):
        return []
    modes = frame.properties["vibrational_modes"]
    return _log_vib_helper(modes, idx_offset=idx_offset)


def _struc_to_vib_traj(struc: Structure) -> Trajectory:
    frame = struc
    nModes = len(frame.properties["vibrational_modes"])
    nAtoms = len(frame)
    nModesPerFrame = 3*nAtoms - 3
    nFramesTot = int(np.ceil(nModes / nModesPerFrame))
    base_properties = frame.properties.copy()
    all_modes = base_properties.pop("vibrational_modes")
    base_frame = frame.copy()
    frames = []
    for iFrame in range(nFramesTot):
        start_idx = iFrame * nModesPerFrame
        end_idx = min(start_idx + nModesPerFrame, nModes)
        frame = base_frame.copy()
        frame.properties = base_properties.copy()
        frame.properties["vibrational_modes"] = all_modes[start_idx:end_idx]
        frames.append(frame)
    write_traj = Trajectory.from_structures(
                structures=frames,
                frame_properties=[frame.properties for frame in frames],
            )
    return write_traj


def _struc_to_gaussian_vib_log(struc: Structure, sp_map: dict | None) -> str:
    write_traj = _struc_to_vib_traj(struc)
    vib_idx_offsets = [i * (3 * len(struc) -3) for i in range(len(write_traj))]
    return _traj_to_gaussian_log(write_traj, False, [0]*len(write_traj), sp_map, vib_idx_offsets = vib_idx_offsets)

def write_Gaussian_vib_log(
    struc: Structure,
    filename: PathLike = "calc.log",
    site_property_map: dict[str, str] | None = None,
) -> None:
    """Write Trajectory object to a Gaussian-formatted log file.

    The produced file will be readable by Gaussview.

    Args:
        filename: File to write. File does not need to end with '.log' to be readable by Gaussview.
            The suffix ".logx" is recommended to distinguish from real Gaussian log files.
        site_property_map: A mapping between implemented Gaussian properties and pymatgen properties.
            i.e. providing {"mulliken charges": "charges"} will write the data stored in
            `Trajectory[i].site_properties["charges"]` as Gaussian logs the Mulliken charges for each site.
            The currently implemented property keys are "mulliken charges", "mulliken spin", "nbo charges",
            "nbo spin", and "esp charges". By default all charge keys are mapped to "charges", and all spin
            keys are mapped to "magmom". All keys must map to site properties of `list[float]`
        write_lattice: Whether to write the lattice information in the log file. Setting this to false will
            not write the lattice information in the log file, but will enable the "forces" site properties
            to be readable by Gaussview.
    """
    # Import inside function as this is an expensive import
    # (causes import time for Trajectory to surpass threshold for 3 test setups)
    # from pymatgen.io.gaussian import _traj_to_gaussian_log
    # struc.to_positions()
    _sp_map = site_property_map or {}
    lines = _struc_to_gaussian_vib_log(struc, _sp_map)
    with open(filename, "w") as file:
        file.write(lines)
    file.close()

def write_Gaussian_log(
    traj: Trajectory,
    filename: PathLike = "calc.log",
    site_property_map: dict[str, str] | None = None,
    write_lattice: bool = True,
) -> None:
    """Write Trajectory object to a Gaussian-formatted log file.

    The produced file will be readable by Gaussview.

    Args:
        filename: File to write. File does not need to end with '.log' to be readable by Gaussview.
            The suffix ".logx" is recommended to distinguish from real Gaussian log files.
        site_property_map: A mapping between implemented Gaussian properties and pymatgen properties.
            i.e. providing {"mulliken charges": "charges"} will write the data stored in
            `Trajectory[i].site_properties["charges"]` as Gaussian logs the Mulliken charges for each site.
            The currently implemented property keys are "mulliken charges", "mulliken spin", "nbo charges",
            "nbo spin", and "esp charges". By default all charge keys are mapped to "charges", and all spin
            keys are mapped to "magmom". All keys must map to site properties of `list[float]`
        write_lattice: Whether to write the lattice information in the log file. Setting this to false will
            not write the lattice information in the log file, but will enable the "forces" site properties
            to be readable by Gaussview.
    """
    # Import inside function as this is an expensive import
    # (causes import time for Trajectory to surpass threshold for 3 test setups)
    # from pymatgen.io.gaussian import _traj_to_gaussian_log

    # Each frame must have an energy to be read properly by Gaussview.
    if hasattr(traj, "frame_properties") and traj.frame_properties is not None:
        energies = [fp.get("energy", 0) for fp in traj.frame_properties]
    else:
        energies = [0] * len(traj)
    traj.to_positions()
    _sp_map = site_property_map or {}
    lines = _traj_to_gaussian_log(traj, write_lattice, energies, _sp_map)
    with open(filename, "w") as file:
        file.write(lines)
    file.close()