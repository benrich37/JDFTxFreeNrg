from pathlib import Path
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from pymatgen.io.jdftx.inputs import JDFTXInfile
import numpy as np
from pymatgen.core.structure import Structure
from scipy.constants import Rydberg
from JDFTxFreeNrg.projection import get_projector, project_out_subspace, project_on_subspace
from JDFTxFreeNrg.glogwrite import write_Gaussian_vib_log


def print_freqs(freqs: list[np.complex128], zero_thresh: float | None = 1e-3):
    for i, f in enumerate(freqs):
        skip = False
        if (zero_thresh is not None) and abs(f) < zero_thresh:
            skip = True
        if not skip:
            if f.imag == 0:
                print(f"Mode {i}: {f.real:.2f} cm^-1")
            elif f.real == 0:
                print(f"Mode {i}: {f.imag:.2f} i cm^-1")
            else:
                print(f"Mode {i}: {f.real:.2f} + {f.imag:.2f} i cm^-1")



def get_invsqrtM(structure: Structure) -> np.ndarray:
    # TODO: Replace the 1822.888 magic number with a proper constant import
    mass_vector = np.array([site.specie.atomic_mass for site in structure.sites]) * 1822.888  # convert to amu
    invsqrtM = np.zeros((len(mass_vector)*3, len(mass_vector)*3))
    for iAtom in range(len(mass_vector)):
        for iCart in range(3):
            idx = iAtom*3 + iCart
            invsqrtM[idx, idx] = 1.0 / np.sqrt(mass_vector[iAtom])
    return invsqrtM

def get_freqs(omegaSqEigs: np.ndarray) -> np.ndarray:
    complex_omegaSqEigs = omegaSqEigs.astype(np.complex128)
    freqs = np.sqrt(complex_omegaSqEigs)
    return freqs

nrg_to_cm_conv = Rydberg / 50.

def freq_nrg_to_cm(freqs: np.ndarray) -> np.ndarray:
    return freqs * nrg_to_cm_conv  # Hartree to cm^-1


def get_projected_K(structure: Structure, K: np.ndarray, molecule_sets: list[dict] | None = None, reverse: bool = False) -> np.ndarray:
    # TODO: Protect reverse from being fed an empty projector
    projector = get_projector(structure, molecule_sets=molecule_sets)
    if reverse:
        K_proj = project_on_subspace(K, projector)
    else:
        K_proj = project_out_subspace(K, projector)
    return K_proj

def get_omegaSq(structure: Structure, K: np.ndarray) -> np.ndarray:
    invsqrtM = get_invsqrtM(structure)
    omegaSq = np.dot(invsqrtM, np.dot(K, invsqrtM))
    return omegaSq

def solve_vib_modes(structure: Structure, K: np.ndarray):
    omegaSq = get_omegaSq(structure, K)
    omegaSqEigs, omegaSqEvecs = np.linalg.eigh(omegaSq)
    return omegaSqEigs, omegaSqEvecs


def get_free_idcs(structure):
    sel_dyn = structure.site_properties.get("selective_dynamics", None)
    if sel_dyn is None:
        return list(range(len(structure)))
    free_idcs = [i for i, sd in enumerate(sel_dyn) if any(sd)]
    return free_idcs

def read_K(calc_dir, structure, expand_to_full: bool = False):
    # nAtoms = len(structure)
    nAtoms = len(get_free_idcs(structure))
    K = np.fromfile(calc_dir / "K", dtype=np.complex128).reshape((nAtoms*3, nAtoms*3)).real
    if expand_to_full:
        K = expand_K(K, structure)
    return K

def expand_K(K_small, structure):
    nAtoms = len(structure)
    full_size = nAtoms * 3
    K_full = np.zeros((full_size, full_size))
    free_idcs = get_free_idcs(structure)
    if len(free_idcs) == nAtoms:
        return K_small
    for i_small, i_full in enumerate(free_idcs):
        for j_small, j_full in enumerate(free_idcs):
            K_full[i_full*3:(i_full+1)*3, j_full*3:(j_full+1)*3] = K_small[i_small*3:(i_small+1)*3, j_small*3:(j_small+1)*3]
    return K_full

def reduce_K(K_full, structure):
    free_idcs = get_free_idcs(structure)
    nFree = len(free_idcs)
    K_small = np.zeros((nFree*3, nFree*3))
    for i_small, i_full in enumerate(free_idcs):
        for j_small, j_full in enumerate(free_idcs):
            K_small[i_small*3:(i_small+1)*3, j_small*3:(j_small+1)*3] = K_full[i_full*3:(i_full+1)*3, j_full*3:(j_full+1)*3]
    return K_small

def get_reduced_structure(structure):
    free_idcs = get_free_idcs(structure)
    reduced_structure = Structure.from_sites([site for i, site in enumerate(structure.sites) if i in free_idcs])
    return reduced_structure

def solve_vib_modes(structure: Structure, K_proj: np.ndarray):
    """ Solve for vibrational modes with a cleaned Hessian.

    Args:
        structure (Structure): pymatgen Structure
        K_proj (np.ndarray): Hessian matrix

    Returns:
        tuple[np.ndarray, np.ndarray]: eigenvalues and eigenvectors of Hessian converted to mass-weighted coordinates
    """
    omegaSq = get_omegaSq(structure, K_proj)
    omegaSqEigs, omegaSqEvecs = np.linalg.eigh(omegaSq)
    return omegaSqEigs, omegaSqEvecs

def solve_vib_modes_debug(structure: Structure, K_proj: np.ndarray):
    """ Solve for vibrational modes with a cleaned Hessian.

    Behaves like solve_vib_modes, but also returns the eigenvectors of the Hessian in normal coordinates as the third return value.
    This is useful for deducing which degrees of freedom need to be projected out to eliminate spurious imaginary modes. 

    Args:
        structure (Structure): pymatgen Structure
        K_proj (np.ndarray): Hessian matrix

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: eigenvalues and eigenvectors of Hessian converted to mass-weighted coordinates, and eigenvectors of the Hessian in normal coordinates
    """
    omegaSq = get_omegaSq(structure, K_proj)
    omegaSqEigs, omegaSqEvecs = np.linalg.eigh(omegaSq)
    _, evecs = np.linalg.eigh(K_proj)
    return omegaSqEigs, omegaSqEvecs, evecs


def get_structure_for_gaussian_vib_log(structure: Structure, freqs_cm: np.ndarray, evecs: np.ndarray, zero_thresh: float = 1e-3) -> Structure:
    free_idcs = get_free_idcs(structure)
    vib_modes = []
    for i in range(len(freqs_cm)):
        if abs(freqs_cm[i]) > zero_thresh:
            displacements = [np.zeros(3) for _ in range(len(structure.sites))]
            _displacements = evecs.T[i].reshape((len(free_idcs), 3))
            for j, idx in enumerate(free_idcs):
                displacements[idx] += _displacements[j]
            mode = {
                "cm^-1": freqs_cm[i],
                "Displacements": displacements,
            }
            vib_modes.append(mode)
    structure.properties = {"vibrational_modes": vib_modes}
    return structure


def get_omegaSqEigs_evecs_from_calc_dir(calc_dir: Path, molecule_sets: list[dict] | None = None, reverse: bool = False) -> tuple[np.ndarray, np.ndarray]:
    infile = JDFTXInfile.from_file(calc_dir / "in")
    structure = infile.structure
    # This is vestigial from when "trans" and "rot" args were present for "get_projected_K" - should we still do something with this?
    fixed = (len(get_free_idcs(structure)) != len(structure))
    K = get_projected_K(structure, read_K(calc_dir, structure, expand_to_full=True), molecule_sets=molecule_sets, reverse=reverse)
    K = reduce_K(K, structure)
    substructure = get_reduced_structure(structure)
    omegaSqEigs, omegaSqEvecs, evecs = solve_vib_modes_debug(substructure, K)
    return omegaSqEigs, omegaSqEvecs, evecs



def get_omegaSqEigs_from_calc_dir(calc_dir: Path, molecule_sets: list[dict] | None = None, reverse: bool = False, use_in: bool = True, trim_zero: bool = True, zero_thresh: float = 2e-17,
                                  ) -> tuple[np.ndarray, np.ndarray]:
    # 2e-17 approximately corresponds to 1e-3 cm^-1
    if use_in:
        structure = JDFTXInfile.from_file(calc_dir / "in").structure
    else:
        structure = JDFTXOutfile.from_file(calc_dir / "out").structure
    fixed = (len(get_free_idcs(structure)) != len(structure))
    if proj_trans is None:
        proj_trans = not fixed
    if proj_rot is None:
        proj_rot = not fixed
    K = get_projected_K(
        structure, 
        read_K(calc_dir, structure, expand_to_full=True), 
        molecule_sets=molecule_sets, reverse=reverse
        )
    K = reduce_K(K, structure)
    substructure = get_reduced_structure(structure)
    _omegaSqEigs, _ = solve_vib_modes(substructure, K)
    omegaSqEigs = [o for o in _omegaSqEigs if (not trim_zero) or (abs(o) > zero_thresh)]
    return omegaSqEigs

def get_freqs_cm_from_calc_dir(calc_dir: Path, molecule_sets: list[dict] | None = None, reverse: bool = False, use_in: bool = True, trim_zero: bool = True, zero_thresh: float = 1e-1,
                               ) -> np.ndarray:
    _zero_thresh = (zero_thresh / nrg_to_cm_conv)**2
    omegaSqEigs = get_omegaSqEigs_from_calc_dir(calc_dir, molecule_sets=molecule_sets, reverse=reverse, use_in=use_in, trim_zero=trim_zero, zero_thresh=_zero_thresh, proj_rot=proj_rot, proj_trans=proj_trans)
    freqs = np.array(freq_nrg_to_cm(get_freqs(np.array(omegaSqEigs))))
    return freqs


def write_Gaussian_vib_log_from_calc_dir(log_path: Path, calc_dir: Path, molecule_sets: list[dict] | None = None, reverse: bool = False, zero_thresh: float = 1e-3, use_in: bool = True):
    omegaSqEigs, omegaSqEvecs, evecs = get_omegaSqEigs_evecs_from_calc_dir(calc_dir, molecule_sets=molecule_sets, reverse=reverse)
    if use_in:
        structure = JDFTXInfile.from_file(calc_dir / "in").structure
    else:
        structure = JDFTXOutfile.from_file(calc_dir / "out").structure
    freqs_cm = freq_nrg_to_cm(get_freqs(omegaSqEigs))
    structure = get_structure_for_gaussian_vib_log(structure, freqs_cm, omegaSqEvecs, zero_thresh=zero_thresh)
    write_Gaussian_vib_log(structure, log_path)


def pert_along_vec(structure: Structure, vec: np.ndarray, disp: float) -> Structure:
    """ Perturb structure along a given vector.

    Args:
        structure (Structure): pymatgen Structure
        vec (np.ndarray): vector to perturb along, shape (nAtoms,3)

    Returns:
        Structure: perturbed structure
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase import Atoms
    atoms: Atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.positions += disp * (vec / np.linalg.norm(vec))
    return AseAtomsAdaptor.get_structure(atoms)

def pert_along_vib_mode(structure: Structure, omegaSqEvecs: np.ndarray, mode_idx: int, disp: float) -> Structure:
    """ Perturb structure along a given imaginary frequency mode.

    Args:
        structure (Structure): pymatgen Structure
        omegaSqEvecs (np.ndarray): eigenvectors of mass-weighted Hessian
        freqs (np.ndarray): frequencies in cm^-1
        mode_idx (int): index of mode to perturb along
        disp (float): displacement magnitude (negative values perturb in opposite direction)

    Returns:
        Structure: perturbed structure
    """
    free_idcs = get_free_idcs(structure)
    vec = np.zeros((len(structure), 3))
    _vec = omegaSqEvecs.T[mode_idx].reshape((len(free_idcs), 3))
    for j, idx in enumerate(free_idcs):
        vec[idx] += _vec[j]
    return pert_along_vec(structure, vec, disp)

# TODO: Use this function in pre-existing functions that perform this operation
def get_imaginary_mode_idcs(freqs: list[np.complex128], zero_thresh: 1e-3) -> list[int]:
    imag_mode_idcs = [i for i, f in enumerate(freqs) if abs(f.imag) >= zero_thresh]
    return imag_mode_idcs

# TODO: Rename this function to something more informative
def _pert_along_im_freqs_helper(structure: Structure, K_proj: np.ndarray, disps: float | list[float] = 0.1, zero_thresh: float = 1e-3) -> list[Structure]:
    omegaSqEigs, omegaSqEvecs = solve_vib_modes(structure, K_proj)
    freqs = get_freqs(omegaSqEigs)
    imag_mode_idcs = get_imaginary_mode_idcs(freqs, zero_thresh)
    if isinstance(disps, float):
        disps = [disps] * len(imag_mode_idcs)
    elif len(disps) > len(imag_mode_idcs):
        print(f"Warning: More displacements provided ({len(disps)}) than imaginary modes found ({len(imag_mode_idcs)}). Truncating displacements list.")
        disps = disps[:len(imag_mode_idcs)]
    elif len(disps) < len(imag_mode_idcs):
        raise ValueError(f"Error: Fewer displacements provided ({len(disps)}) than imaginary modes found ({len(imag_mode_idcs)}).")
    pert_structure = structure.copy()
    for mode_idx, disp in zip(imag_mode_idcs, disps):
        pert_structure = pert_along_vib_mode(pert_structure, omegaSqEvecs, mode_idx, disp)
    return pert_structure

def _pert_along_im_freqs(structure: Structure, K: np.ndarray, molecule_sets: list[dict] | None = None, disps: float | list[float] = 0.1, zero_thresh: float = 1e-3) -> Structure:
    K_proj = get_projected_K(structure, K, molecule_sets=molecule_sets)
    pert_structure = _pert_along_im_freqs_helper(structure, K_proj, disps=disps, zero_thresh=zero_thresh)
    return pert_structure

def pert_along_im_freqs(structure: Structure, K: np.ndarray, molecule_sets: list[dict] | None = None, disps: float | list[float] = 0.1, zero_thresh: float = 1e-3) -> Structure:
    try:
        pert_structure = _pert_along_im_freqs(structure, K, molecule_sets=molecule_sets, disps=disps, zero_thresh=zero_thresh)
    except ValueError as e:
        print(f"Error generating structure perturbed along imaginary frequencies")
        raise e
    return pert_structure