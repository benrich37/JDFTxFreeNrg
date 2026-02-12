
from JDFTxFreeNrg._orthog import remove_parallel_vectors_loop, progressively_orthogonalize_vectors, orthogonalize_projector, safe_orthogonalize_projector
from JDFTxFreeNrg._common import dagger, box, dot, remove_phase, _error_on_nan_in_array, _error_on_nan_in_list_of_vectors
import numpy as np
from pymatgen.core.structure import Structure, Molecule
from pymatgen.core.units import bohr_to_ang

ref_axs = {
    "x": np.array([1,0,0], dtype=np.float64),
    "y": np.array([0,1,0], dtype=np.float64),
    "z": np.array([0,0,1], dtype=np.float64)
}

class Mode:
    def __init__(self, n: np.ndarray, iAtom: int):
        self.n = n
        self.iAtom = iAtom

def _get_modes(structure: Structure) -> list[Mode]:
    modes = []
    nAtoms = len(structure.sites)
    for iAtom in range(nAtoms):
        for iDir in range(3):
            vec = np.zeros(3)
            vec[iDir] = 1.0
            mode = Mode(vec, iAtom)
            modes.append(mode)
    return modes

def get_modes(structure: Structure) -> list[Mode]:
    try:
        return _get_modes(structure)
    except Exception as e:
        print("Error in constructing modes:")
        raise e

def get_CMcoords(structure: Structure) -> np.ndarray:
    mass_vector = np.array([site.specie.atomic_mass for site in structure.sites])
    total_mass = np.sum(mass_vector)
    CMcoords = np.zeros((len(structure.sites), 3))
    CM = np.zeros(3)
    for i, site in enumerate(structure.sites):
        CM += site.coords * mass_vector[i]
    CM /= total_mass
    for i, site in enumerate(structure.sites):
        CMcoords[i, :] = site.coords - CM
    return CMcoords / bohr_to_ang

def get_inertia_tensor(struc: Structure, center: np.ndarray | None = None) -> np.ndarray:
    """
    Calculate the average moment of inertia of a molecule.

    Args:
        mol (Molecule): Pymatgen Molecule

    Returns:
        int, list: average moment of inertia, eigenvalues of the inertia tensor
    """
    if not isinstance(center, np.ndarray):
        center = resolve_center(struc, center)
    try:
        inertia_tensor = _get_inertia_tensor(struc, center=center)
    except Exception as e:
        print("Error in calculating inertia tensor:")
        raise e
    _error_on_nan_in_array(inertia_tensor, context="inertia tensor calculation")
    return inertia_tensor

def _get_inertia_tensor(struc: Structure, center: np.ndarray | None = None) -> np.ndarray:
    mol = Molecule.from_sites(struc.sites)
    if center is None:
        centered_mol = mol.get_centered_molecule()
    else:
        centered_mol = mol.copy()
        centered_mol.translate_sites(range(len(centered_mol.sites)), -center)
    inertia_tensor = np.zeros((3, 3))
    for site in centered_mol:
        c = site.coords
        wt = site.specie.atomic_mass
        for dim in range(3):
            inertia_tensor[dim, dim] += wt * (c[(dim + 1) % 3] ** 2 + c[(dim + 2) % 3] ** 2)
        for ii, jj in [(0, 1), (1, 2), (0, 2)]:
            inertia_tensor[ii, jj] += -wt * c[ii] * c[jj]
            inertia_tensor[jj, ii] += -wt * c[jj] * c[ii]
    return inertia_tensor / (bohr_to_ang ** 2)




def get_centered_coords(structure: Structure, center: np.ndarray) -> np.ndarray:
    centered_coords = np.zeros((len(structure.sites), 3))
    for i, site in enumerate(structure.sites):
        centered_coords[i, :] = site.coords - center
    return centered_coords / bohr_to_ang

def get_CMcoords(structure: Structure) -> np.ndarray:
    return get_centered_coords(structure, get_center_of_mass(structure))

def get_center_of_mass(structure: Structure) -> np.ndarray:
    mass_vector = np.array([site.specie.atomic_mass for site in structure.sites])
    total_mass = np.sum(mass_vector)
    CM = np.zeros(3)
    for i, site in enumerate(structure.sites):
        CM += site.coords * mass_vector[i]
    CM /= total_mass
    return CM

def get_rotation_vector(axis: np.ndarray, structure: Structure, mol_indices: list[int] | None = None) -> np.ndarray:
    if mol_indices is None:
        mol_indices = list(range(len(structure.sites)))
    modes = get_modes(structure)
    CMcoords = get_CMcoords(structure)
    return _get_rotation_vector(axis, modes, CMcoords, mol_indices)

def _get_rotation_vector(axis: np.ndarray, modes: list[Mode], CMcoords: np.ndarray, mol_indices: list[int]) -> np.ndarray:
    vec = np.zeros(len(modes))
    for i, mode in enumerate(modes):
        if mode.iAtom in mol_indices:
            vec[i] = box(modes[i].n, axis.T, CMcoords[mol_indices.index(modes[i].iAtom)])
    return vec


# TODO: Partition out individual operations being performed within this function into helper functions for testing and clarity
def get_rotations_molecule(
        modes: list[Mode], molecule_structure: Structure, mol_indices: list[int], symmThreshold: float = 1e-5,
        center: np.ndarray | None = None, axes: list[np.ndarray] | None = None, ortho: bool = True, safe_ortho: bool = True
        ):
    if center is None:
        center = get_center_of_mass(molecule_structure)
    CMcoords = get_centered_coords(molecule_structure, center)
    if axes is None:
        axes = []
        InertiaTensor = get_inertia_tensor(molecule_structure)
        Ieigs, Ievecs = np.linalg.eigh(InertiaTensor)
        for j in range(3):
            if Ieigs[j] > symmThreshold:
                axes.append(remove_phase(3, Ievecs[:, j]).real)
    projectors = []
    for j, axis in enumerate(axes):
        vec = _get_rotation_vector(axis, modes, CMcoords, mol_indices)
        _error_on_nan_in_array(np.array(vec), context=f"rotation projector for axis {j}")
        projectors.append(vec)
    if ortho:
        if safe_ortho:
            projectors = safe_orthogonalize_projector(np.array(projectors).T)
        else:
            projectors = orthogonalize_projector(np.array(projectors).T)
        projectors = [v for v in projectors.T]
    return projectors

def get_translations_molecule(modes: list[Mode], mol_indices: list[int], axes: list[np.ndarray] | None = None) -> np.ndarray:
    projectors = []
    if axes is None:
        axes = list(np.eye(3))
    else:
        axes = progressively_orthogonalize_vectors(axes)
    for k, axis in enumerate(axes):
        vec = np.zeros(len(modes))
        for i, mode in enumerate(modes):
            if mode.iAtom in mol_indices:
                vec[i] = dot(mode.n, axis)
        projectors.append(vec)
    return projectors

def project_out_vector_from_vector(vector_save: np.ndarray, vector_proj: np.ndarray) -> np.ndarray:
    overlap = dot(vector_save, vector_proj) / dot(vector_proj, vector_proj)
    projected_vector = vector_save - overlap * vector_proj
    return projected_vector / np.linalg.norm(projected_vector)

# TODO: Test this function
def gen_ortho_axes_R3(axis: np.ndarray) -> list[np.ndarray]:
    R3 = np.eye(3)
    axis = axis / np.linalg.norm(axis)
    overlaps = [abs(np.dot(axis, R3[:, i])) for i in range(3)]
    idcs = np.argsort(overlaps)
    i1, i2 = idcs[0], idcs[1]
    v1 = project_out_vector_from_vector(R3[:, i1], axis)
    v2 = project_out_vector_from_vector(R3[:, i2], axis)
    v2 = project_out_vector_from_vector(v2, v1)
    return [v1, v2]

def resolve_center(structure: Structure, center: int | list[int] | np.ndarray | None) -> np.ndarray | None:
    if isinstance(center, np.ndarray):
        return center
    elif isinstance(center, int):
        return structure.cart_coords[center]
    elif isinstance(center, list):
        if not len(center):
            return None
        if isinstance(center[0], int):
            coords = np.array([structure.cart_coords[i] for i in center])
            return np.mean(coords, axis=0)
        if isinstance(center[0], float):
            center_array = np.array(center)
            if center_array.shape != (3,):
                raise ValueError(f"Center numpy array must be shape (3,), got shape {center_array.shape}")
            return center_array
    return None


def resolve_axis(structure: Structure, axis: str | list[int] | np.ndarray | dict) -> list[np.ndarray]:
    axis_data, gen_ortho_set = _resolve_axis_data(axis)
    axis_vec = _axis_data_to_vector(structure, axis_data)
    if gen_ortho_set:
        return gen_ortho_axes_R3(axis_vec)
    else:
        return [axis_vec]
    
def _resolve_axis_data(axis: str | list[int] | np.ndarray | dict) -> tuple[np.ndarray, bool]:
    axis_data = axis
    gen_ortho_set = False
    if isinstance(axis, dict):
        gen_ortho_set = axis.get("ortho", False)
        if not "axis" in axis:
            raise ValueError("An axis provided as a dictionary must provide axis data (str, list[int], np.ndarray) under the key 'axis'.")
        axis_data = axis["axis"]
    return axis_data, gen_ortho_set

def _axis_data_to_vector(structure: Structure, axis: str | list[int] | np.ndarray) -> np.ndarray:
    # _axis = None
    if isinstance(axis, np.ndarray):
        if not np.shape(axis) in [(3,), (3,1)]:
            raise ValueError(
                f"Axis numpy array must be shape (3,), got shape {np.shape(axis)} "
                "(if you were trying to provide multiple axes, use a list of arrays instead."
                "If you were trying to provide indices, use a list of two integers instead.)"
                )
        # return axis/np.linalg.norm(axis)
    elif isinstance(axis, str):
        if not axis in ref_axs:
            raise ValueError(f"Unknown named axis: {axis}. Valid options are: {list(ref_axs.keys())}")
        axis = ref_axs[axis]
    elif isinstance(axis, list):
        if not all([isinstance(i, int) for i in axis]):
            raise TypeError(f"Axis list must contain integers, got: {axis}")
        if len(axis) != 2:
            raise ValueError(f"Axis list must contain exactly two indices, got: {axis}")
        p0 = structure.cart_coords[axis[0]]
        p1 = structure.cart_coords[axis[1]]
        axis = p1 - p0
    else:
        raise TypeError(f"Unexpected axis data type: {type(axis)}")
    return axis / np.linalg.norm(axis)
    
def resolve_idcss(structure: Structure, molecule_sets: list[dict]) -> None:
    try:
        _resolve_idcss(structure, molecule_sets)
    except Exception as e:
        print(f"Error in resolving indices for molecule sets. {molecule_sets}")
        raise e

def _resolve_idcss(structure: Structure, molecule_sets: list[dict]) -> None:
    for mset in molecule_sets:
        idcs = mset.get("indices", None)
        if idcs is None:
            resolved_idcs = list(range(len(structure.sites)))
        else:
            resolved_idcs = resolve_idcs(structure, idcs)
        mset["indices"] = resolved_idcs

def resolve_idcs(structure: Structure, idcs: list[int] | str) -> None:
    if isinstance(idcs, str):
        if idcs == "all":
            return list(range(len(structure.sites)))
        else:
            raise ValueError(f"Unknown string for indices: {idcs}")
    else:
        try:
            return list([int(i) for i in idcs])
        except Exception as e:
            print(f"Error in resolving indices: {idcs}")
            raise e
    
def resolve_axes(structure: Structure, molecule_sets: list[dict]) -> None:
    try:
        _resolve_axes(structure, molecule_sets)
    except Exception as e:
        print(f"Error in resolving axes for molecule sets. {molecule_sets}")
        raise e

def _resolve_axes(structure: Structure, molecule_sets: list[dict]) -> None:
    for mset in molecule_sets:
        if "axes" in mset:
            axes = mset["axes"]
            if axes is None:
                resolved_axes = None
            else:
                resolved_axes = []
                for axis in axes:
                    resolved_axes += resolve_axis(structure, axis)
            mset["axes"] = resolved_axes


    
def get_clean_mol_set(mol_set: dict) -> dict:
    try:
        return _get_clean_mol_set(mol_set)
    except Exception as e:
        print(f"Error in cleaning molecule set: {mol_set}")
        raise e

def _get_clean_mol_set(mol_set: dict) -> dict:
    clean_set = {}
    clean_set['indices'] = mol_set.get("indices", None)
    if (clean_set['indices'] is None) or (isinstance(clean_set["indices"], list) and len(clean_set['indices']) == 0):
        raise ValueError("molecule set must have non-empty 'indices' list. (Use 'all' to select all atoms.)")
    clean_set['trans'] = mol_set.get("trans", False)
    clean_set['rot'] = mol_set.get("rot", False)
    if all ([not clean_set["trans"], not clean_set["rot"]]):
        raise ValueError("molecule set must have at least one of 'trans' or 'rot' set to True")
    clean_set['center'] = mol_set.get("center", None)
    clean_set['axes'] = mol_set.get("axes", None)
    if (isinstance(clean_set['axes'], list) and len(clean_set['axes']) == 0):
        raise ValueError("molecule set must have non-empty 'axes' list. (Use [\"x\",\"y\",\"z\"] for standard axes, or leave as None/missing to auto-generate based on inertia tensor.)")
    clean_set['ortho'] = mol_set.get("ortho", True)
    return clean_set

def _append_projectors_mol_set(projectors: list[np.ndarray], modes: list[Mode], structure: Structure, mol_set: dict):
    clean_set = get_clean_mol_set(mol_set)
    projectors = _append_projectors_mol_set_trans(projectors, modes, clean_set)
    projectors = _append_projectors_mol_set_rot(projectors, modes, clean_set, structure)
    return projectors

def _append_projectors_mol_set_trans(projectors: list[np.ndarray], modes: list[Mode], clean_set: dict):
    if clean_set["trans"]:
        try:
            projectors += get_translations_molecule(modes, clean_set["indices"], axes=clean_set["axes"])
        except Exception as e:
            print(f"Error in getting translation projectors for molecule set: {clean_set}")
            raise e
    return projectors

def _append_projectors_mol_set_rot(projectors: list[np.ndarray], modes: list[Mode], clean_set: dict, structure: Structure):
    if clean_set["rot"]:
        try:
            mol_structure = structure.copy()
            mol_structure.remove_sites([i for i in range(len(structure.sites)) if i not in clean_set["indices"]])
            projectors += get_rotations_molecule(modes, mol_structure, clean_set["indices"], center=clean_set["center"], axes=clean_set["axes"], ortho=clean_set["ortho"])
        except Exception as e:
            print(f"Error in getting rotation projectors for molecule set: {clean_set}")
            raise e
    return projectors

def get_projector_raw(structure: Structure, molecule_sets: list[dict] | None = None) -> np.ndarray:
    try:
        return _get_projector_raw(structure, molecule_sets=molecule_sets)
    except Exception as e:
        print("Error in constructing raw projector:")
        raise e

def _get_projector_raw(structure: Structure, molecule_sets: list[dict] | None = None) -> np.ndarray:
    if molecule_sets is None:
        molecule_sets = []
    modes = get_modes(structure)
    projectors = []
    resolve_idcss(structure, molecule_sets)
    resolve_axes(structure, molecule_sets)
    for i, mol_set in enumerate(molecule_sets):
        projectors = _append_projectors_mol_set(projectors, modes, structure, mol_set)
    projector = np.array(projectors).T
    norm_projector = projector / np.linalg.norm(projector, axis=0)
    return norm_projector


def get_projector(structure: Structure, molecule_sets: list[dict] | None = None) -> np.ndarray:
    projector_raw = get_projector_raw(structure, molecule_sets=molecule_sets)
    vectors = [v for v in projector_raw.T]
    vectors = remove_parallel_vectors_loop(vectors, cutoff=1e-4)
    projector = np.array(vectors).T
    return projector


def project_on_subspace(mat: np.ndarray, subspace: np.ndarray) -> np.ndarray:
    ppDag = subspace @ dagger(subspace)
    # mat_proj = ppDag @ mat @ ppDag
    mat_proj = ppDag @ mat @ ppDag.T
    return mat_proj

def project_out_subspace(mat: np.ndarray, subspace: np.ndarray) -> np.ndarray:
    ppDag = subspace @ dagger(subspace)
    IminPpdag = np.eye(mat.shape[0]) - ppDag
    # mat_proj = IminPpdag @ mat @ IminPpdag
    mat_proj = IminPpdag @ mat @ IminPpdag.T
    return mat_proj

def get_subspace_overlap(projector1: np.ndarray, projector2: np.ndarray):
    # For different projector shapes, put the smaller subspace in projector1
    svals = np.linalg.svd(dagger(projector1) @ projector2, compute_uv=False)
    frac = np.sum(svals**2)/(np.sqrt(len(projector1.T)))
    return frac