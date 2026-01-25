from pathlib import Path
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from pymatgen.io.jdftx.inputs import JDFTXInfile
import numpy as np
from pymatgen.core.structure import Structure
from scipy.constants import Rydberg
from JDFTxFreeNrg.projection import get_projector, project_out_subspace, project_on_subspace
from JDFTxFreeNrg.glogwrite import write_Gaussian_vib_log
from warnings import warn
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms


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
                print(f"Mode {i}: {f.real:.2f} + {f.imag:.2f} i cm^-1") # I don't see how this could ever happen


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
    """ Convert eigenvalues of mass-weighted Hessian to frequencies in Hartree.
    
    Args:
        omegaSqEigs (np.ndarray): Eigenvalues of mass-weighted Hessian
        
    Returns:
        np.ndarray: Frequencies in Hartree
    """
    freqs = np.sqrt(omegaSqEigs.astype(np.complex128))
    return freqs

nrg_to_cm_conv = Rydberg / 50.

def freq_nrg_to_cm(freqs: np.ndarray) -> np.ndarray:
    """ Convert frequencies from Hartree to cm^-1
    
    Args:
        freqs (np.ndarray): Frequencies in Hartree
    
    Returns:
        np.ndarray: Frequencies in cm^-1
    """
    return freqs * nrg_to_cm_conv  # Hartree to cm^-1


def get_projected_K(structure: Structure, K: np.ndarray, molecule_sets: list[dict] | None = None, reverse: bool = False) -> np.ndarray:
    """ Returns the Hessian projected onto or out of the subspace defined by the molecule sets.
    
    Args:
        structure (Structure): pymatgen Structure
        K (np.ndarray): Hessian matrix in full form.
        molecule_sets (list[dict] | None): List of molecule sets to project out (or onto).
        reverse (bool): If True, project onto the subspace instead of out of it.
        
    Returns:
        np.ndarray: Projected Hessian matrix
    """
    # TODO: Protect reverse from being fed an empty projector
    projector = get_projector(structure, molecule_sets=molecule_sets)
    if reverse:
        K_proj = project_on_subspace(K, projector)
    else:
        K_proj = project_out_subspace(K, projector)
    return K_proj

def get_omegaSq(structure: Structure, K: np.ndarray) -> np.ndarray:
    """ Get mass-weighted Hessian.

    Args:
        structure (Structure): pymatgen Structure
        K (np.ndarray): Hessian matrix

    Returns:
        np.ndarray: Mass-weighted Hessian matrix
    """
    invsqrtM = get_invsqrtM(structure)
    omegaSq = np.dot(invsqrtM, np.dot(K, invsqrtM))
    return omegaSq

def solve_vib_modes(structure: Structure, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Get vibrational modes and eigenvalues from Hessian and its structure.
    
    Args:
        structure (Structure): pymatgen Structure
        K (np.ndarray): Hessian matrix
        
    Returns:
        tuple[np.ndarray, np.ndarray]: eigenvalues and eigenvectors of Hessian converted to mass-weighted coordinates"""
    omegaSqEigs, omegaSqEvecs = np.linalg.eigh(get_omegaSq(structure, K))
    return omegaSqEigs, omegaSqEvecs


def get_free_idcs(structure: Structure) -> list[int]:
    """ Get the indices of free atoms in the structure based on selective dynamics flags.
    
    Args:
        structure (Structure): pymatgen Structure
        
    Returns:
        list[int]: List of indices of free atoms
    """
    sel_dyn = structure.site_properties.get("selective_dynamics", None)
    if sel_dyn is None:
        return list(range(len(structure)))
    free_idcs = [i for i, sd in enumerate(sel_dyn) if any(sd)]
    return free_idcs

def read_K(calc_dir: Path | str, structure: Structure, expand_to_full: bool = False, calc_prefix: str | None = None) -> np.ndarray:
    """ Read the Hessian matrix from a JDFTx calculation directory.

    Args:
        calc_dir (Path | str): Path to the JDFTx calculation directory
        structure (Structure): pymatgen Structure
        expand_to_full (bool): Whether to expand the Hessian to include fixed atoms (currently expected by all projection functions)
        calc_prefix (str | None): Prefix for the calculation files
    
    Returns:
        np.ndarray: Hessian matrix
    """
    if calc_prefix is None:
        calc_prefix = ""
    elif not calc_prefix.endswith("."):
        calc_prefix += "."
    nAtoms = len(get_free_idcs(structure))
    K = np.fromfile(Path(calc_dir) / f"{calc_prefix}K", dtype=np.complex128).reshape((nAtoms*3, nAtoms*3)).real
    if expand_to_full:
        K = expand_K(K, structure)
    return K

def expand_K(K_small: np.ndarray, structure: Structure) -> np.ndarray:
    """ Expand a reduced Hessian to the full Hessian including fixed atoms.
    
    Expand a reduced Hessian matrix to the full Hessian matrix by inserting zero rows and columns for the fixed 
    atoms as specified by the selective dynamics flags in the structure.
    
    Args:
        K_small (np.ndarray): Reduced Hessian matrix
        structure (Structure): pymatgen Structure
    
    Returns:
        np.ndarray: Full Hessian matrix
    """
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

def reduce_K(K_full: np.ndarray, structure: Structure) -> np.ndarray:
    """ Reduce a full Hessian to only the free atoms.

    Reduce a full Hessian matrix to only the free atoms as specified by the selective dynamics flags in the structure.
    
    Args:
        K_full (np.ndarray): Full Hessian matrix
        structure (Structure): pymatgen Structure
    
    Returns:
        np.ndarray: Reduced Hessian matrix
    """
    free_idcs = get_free_idcs(structure)
    nFree = len(free_idcs)
    K_small = np.zeros((nFree*3, nFree*3))
    for i_small, i_full in enumerate(free_idcs):
        for j_small, j_full in enumerate(free_idcs):
            K_small[i_small*3:(i_small+1)*3, j_small*3:(j_small+1)*3] = K_full[i_full*3:(i_full+1)*3, j_full*3:(j_full+1)*3]
    return K_small

def expand_posn_vec(structure: Structure, vec_red: np.ndarray) -> np.ndarray:
    """ Expand a reduced position vector to the full structure including fixed atoms.
    
    Args:
        structure (Structure): pymatgen Structure
        vec_red (np.ndarray): Reduced position vector of shape (nFreeAtoms, 3)
        
    Returns:
        np.ndarray: Full position vector of shape (nAtoms, 3)
    """
    nAtoms = len(structure)
    vec_full = np.zeros((nAtoms, 3))
    free_idcs = get_free_idcs(structure)
    for i, idx in enumerate(free_idcs):
        vec_full[idx] = vec_red[i]
    return vec_full

def reduce_posn_vec(structure: Structure, vec_full: np.ndarray) -> np.ndarray:
    """ Reduce a full position vector to only the free atoms.
    
    Args:
        structure (Structure): pymatgen Structure
        vec_full (np.ndarray): Full position vector of shape (nAtoms, 3)
        
    Returns:
        np.ndarray: Reduced position vector of shape (nFreeAtoms, 3)
    """
    free_idcs = get_free_idcs(structure)
    vec_red = np.array([vec_full[idx] for idx in free_idcs])
    return vec_red

def get_reduced_structure(structure: Structure) -> Structure:
    """ Get a reduced structure containing only the free atoms.
    
    Args:
        structure (Structure): pymatgen Structure
        
    Returns:
        Structure: Reduced pymatgen Structure
    """
    free_idcs = get_free_idcs(structure)
    reduced_structure = Structure.from_sites([site for i, site in enumerate(structure.sites) if i in free_idcs])
    return reduced_structure

def solve_vib_modes(structure: Structure, K_proj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    omegaSqEigs = get_omegaSqEigs_from_calc_dir(calc_dir, molecule_sets=molecule_sets, reverse=reverse, use_in=use_in, trim_zero=trim_zero, zero_thresh=_zero_thresh)
    freqs = np.array(freq_nrg_to_cm(get_freqs(np.array(omegaSqEigs))))
    return freqs


def write_Gaussian_vib_log_from_calc_dir(log_path: Path, calc_dir: Path, molecule_sets: list[dict] | None = None, reverse: bool = False, zero_thresh: float = 1e-3, use_in: bool = True) -> None:
    """ Write a Gaussian-format vibrational log file from a JDFTx calculation directory.

    Helpful for debugging vibrational modes / constructing required projector sets.
    
    Args:
        log_path (Path): Path to write the Gaussian vibrational log file to
        calc_dir (Path): Path to the JDFTx calculation directory
        molecule_sets (list[dict] | None): List of molecule sets to project out (or onto).
        reverse (bool): If True, project onto the subspace instead of out of it.
        zero_thresh (float): Threshold for identifying zero frequencies in cm^-1
        use_in (bool): Whether to read the structure from the "in" file (True) or "out" file (False)
    """
    omegaSqEigs, omegaSqEvecs, _ = get_omegaSqEigs_evecs_from_calc_dir(calc_dir, molecule_sets=molecule_sets, reverse=reverse)
    if use_in:
        structure = JDFTXInfile.from_file(calc_dir / "in").structure
    else:
        structure = JDFTXOutfile.from_file(calc_dir / "out").structure
    freqs_cm = freq_nrg_to_cm(get_freqs(omegaSqEigs))
    structure = get_structure_for_gaussian_vib_log(structure, freqs_cm, omegaSqEvecs, zero_thresh=zero_thresh)
    write_Gaussian_vib_log(structure, log_path)


def _pert_along_vec(structure: Structure, dvec: np.ndarray) -> Structure:
    atoms = AseAtomsAdaptor.get_atoms(structure, msonable=False)
    atoms.positions += dvec
    return AseAtomsAdaptor.get_structure(atoms)


# No longer being used but keeping just in case
def pert_along_vec(structure: Structure, vec: np.ndarray, disp: float) -> Structure:
    """ Perturb structure along a given vector.

    Args:
        structure (Structure): pymatgen Structure
        vec (np.ndarray): vector to perturb along, shape (nAtoms,3)

    Returns:
        Structure: perturbed structure
    """
    dvec = disp * (vec / np.linalg.norm(vec))
    return _pert_along_vec(structure, dvec)

# No longer being used but keeping just in case
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
    """ Get the indices of the frequencies that are imaginary.
    
    Args:
        freqs (list[np.complex128]): List of frequencies in cm^-1
        zero_thresh (float): threshold for identifying imaginary frequencies in cm^-1

    Returns:
        list[int]: List of indices of imaginary frequencies
    """
    imag_mode_idcs = [i for i, f in enumerate(freqs) if abs(f.imag) >= zero_thresh]
    return imag_mode_idcs

# No longer being used but keeping just in case
def _pert_along_im_freqs_helper(structure: Structure, K_proj: np.ndarray, disps: float | list[float] = 0.1, zero_thresh: float = 1e-3, cumulative_displacement: bool = False) -> list[Structure]:
    omegaSqEigs, omegaSqEvecs = solve_vib_modes(structure, K_proj)
    freqs = get_freqs(omegaSqEigs)
    imag_mode_idcs = get_imaginary_mode_idcs(freqs, zero_thresh)
    if len(imag_mode_idcs) == 0:
        warn("Warning: No imaginary frequency modes found to perturb along.", stacklevel=2)
    if isinstance(disps, float):
        disps = [disps] * len(imag_mode_idcs)
    elif len(disps) > len(imag_mode_idcs):
        warn(f"Warning: More displacements provided ({len(disps)}) than imaginary modes found ({len(imag_mode_idcs)}). Truncating displacements list.", stacklevel=2)
        disps = disps[:len(imag_mode_idcs)]
    elif len(disps) < len(imag_mode_idcs):
        raise ValueError(f"Error: Fewer displacements provided ({len(disps)}) than imaginary modes found ({len(imag_mode_idcs)}).")
    pert_structure = structure.copy()
    for mode_idx, disp in zip(imag_mode_idcs, disps):
        pert_structure = pert_along_vib_mode(pert_structure, omegaSqEvecs, mode_idx, disp)
    return pert_structure



def _pert_along_im_freqs_get_disp_list(disps: float | list[float], n_vectors: int) -> list[float]:
    if isinstance(disps, float):
        disp_list = [disps] * n_vectors
    elif len(disps) > n_vectors:
        warn(f"Warning: More displacements provided ({len(disps)}) than imaginary modes found ({n_vectors}). Truncating displacements list.", stacklevel=2)
        disp_list = disps[:n_vectors]
    elif len(disps) < n_vectors:
        raise ValueError(f"Error: Fewer displacements provided ({len(disps)}) than imaginary modes found ({n_vectors}).")
    else:
        disp_list = disps.copy()
    return disp_list

def _pert_along_im_freqs_get_norm_vib_vecs(omegaSqEvecs: np.ndarray, nAtoms: int) -> list[np.ndarray]:
    vib_vecs = [v.reshape((nAtoms, 3)) for v in omegaSqEvecs.T]
    norm_vib_vecs = [v / np.linalg.norm(v) for v in vib_vecs]
    return norm_vib_vecs

def _pert_along_im_freqs_get_use_vectors(structure: Structure, K: np.ndarray, zero_thresh: float = 1e-3) -> list[np.ndarray]:
    nAtoms = len(structure)
    omegaSqEigs, omegaSqEvecs = solve_vib_modes(structure, K)
    freqs = freq_nrg_to_cm(get_freqs(omegaSqEigs))
    imag_mode_idcs = get_imaginary_mode_idcs(freqs, zero_thresh)
    if len(imag_mode_idcs) == 0:
        raise ValueError("No imaginary frequency modes found to perturb along.")
    norm_vib_vecs = _pert_along_im_freqs_get_norm_vib_vecs(omegaSqEvecs, nAtoms)
    use_vectors = [norm_vib_vecs[i] for i in imag_mode_idcs]
    return use_vectors

def _pert_along_im_freqs_disp_vec_constructor(use_vectors: list[np.ndarray], disps: float | list[float], norm_method: str = "default") -> np.ndarray:
    dvec = np.zeros(use_vectors[0].shape)
    disp_list = _pert_along_im_freqs_get_disp_list(disps, len(use_vectors))
    for i, vec in enumerate(use_vectors):
        dvec += disp_list[i] * vec
    if norm_method.lower().startswith("c"):
        assert isinstance(disps, float), "cumulative norm_method requires disps to be a single float value."
        dvec *= disps / np.linalg.norm(dvec)
    elif norm_method.lower().startswith("m"):
        assert isinstance(disps, float), "max norm_method requires disps to be a single float value."
        max_disp = np.max(np.linalg.norm(dvec, axis=1))
        dvec *= disps / max_disp
    return dvec


def _pert_along_im_freqs(structure: Structure, K: np.ndarray, molecule_sets: list[dict] | None = None, disps: float | list[float] = 0.1, zero_thresh: float = 1e-3, norm_method: str | None = None) -> Structure:
    K_proj = get_projected_K(structure, K, molecule_sets=molecule_sets)
    if norm_method is None:
        norm_method = "default"
    if norm_method.lower().startswith("c") and isinstance(disps, list):
        raise ValueError("'cumulative' norm_method requires disps to be a single float value.")
    use_vectors = _pert_along_im_freqs_get_use_vectors(structure, K_proj, zero_thresh=zero_thresh)
    dvec = _pert_along_im_freqs_disp_vec_constructor(use_vectors, disps, norm_method=norm_method)
    pert_structure = _pert_along_vec(structure, dvec)
    return pert_structure

def pert_along_im_freqs(
        structure: Structure, K: np.ndarray, 
        molecule_sets: list[dict] | None = None, disps: float | list[float] = 0.1, 
        zero_thresh: float = 1e-3,
        norm_method: str | None = None,
        ) -> Structure:
    """ Perturb structure along all imaginary frequency modes.
    
    Args:
        structure (Structure): Unperturbed pymatgen Structure
        K (np.ndarray): Hessian matrix expanded to full structure
        molecule_sets (list[dict] | None): List of molecule sets to project out.
        disps (float | list[float]): displacement magnitude(s) (negative values perturb in opposite direction)
            Providing a single float applies the same displacement to all modes.
            A list of floats longer than the number of imaginary modes will be truncated.
        zero_thresh (float): threshold for identifying imaginary frequencies in cm^-1
        norm_method (str | None): Method for normalizing the perturbation vector. Options are:
            None: Length of each imaginary mode vector is set to value in disps (default behavior).
            "cumulative": Length of total perturbation vector is set to value in disps.
            "max": Displacement of the atom with the largest displacement is set to value in disps.
        
    Returns:
        Structure: perturbed structure
    """
    try:
        pert_structure = _pert_along_im_freqs(structure, K, molecule_sets=molecule_sets, disps=disps, zero_thresh=zero_thresh, norm_method=norm_method)
    except ValueError as e:
        print("Error generating structure perturbed along imaginary frequencies")
        raise e
    return pert_structure