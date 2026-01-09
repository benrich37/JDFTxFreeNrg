
from JDFTxFreeNrg._orthog import remove_parallel_vectors_loop, progressively_orthogonalize_vectors, orthogonalize_projector
from JDFTxFreeNrg._common import dagger, box, dot, remove_phase
import numpy as np
from pymatgen.core.structure import Structure, Molecule
from pymatgen.core.units import bohr_to_ang

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

def get_inertia_tensor(struc: Structure):
    """
    Calculate the average moment of inertia of a molecule.

    Args:
        mol (Molecule): Pymatgen Molecule

    Returns:
        int, list: average moment of inertia, eigenvalues of the inertia tensor
    """
    mol = Molecule.from_sites(struc.sites)
    centered_mol = mol.get_centered_molecule()
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

class Mode:
    def __init__(self, n: np.ndarray, iAtom: int):
        self.n = n
        self.iAtom = iAtom

def get_modes(structure: Structure) -> list[Mode]:
    modes = []
    nAtoms = len(structure.sites)
    for iAtom in range(nAtoms):
        for iDir in range(3):
            vec = np.zeros(3)
            vec[iDir] = 1.0
            mode = Mode(vec, iAtom)
            modes.append(mode)
    return modes

def fill_translations(projector: np.ndarray, modes: list[Mode], i_start: int = 0):
    for k in range(3):
        axis = np.zeros(3)
        axis[k] = 1.0
        for i, mode in enumerate(modes):
            projector[i, i_start + k] = dot(mode.n, axis)
    return projector

def fill_translations_molecule(projector: np.ndarray, modes: list[Mode], mol_indices: list[int], i_start: int = 0, dofs: list[int] | None = None):
    if dofs is None:
        dofs = [0, 1, 2]
    for k in dofs:
        axis = np.zeros(3)
        axis[k] = 1.0
        for i, mode in enumerate(modes):
            if mode.iAtom in mol_indices:
                projector[i, i_start + k] = dot(mode.n, axis)
    return projector

def fill_rotations(projector: np.ndarray, modes: list[Mode], structure: Structure, symmThreshold: float = 1e-5, i_start: int = 0):
    CMcoords = get_CMcoords(structure)
    InertiaTensor = get_inertia_tensor(structure)
    Ieigs, Ievecs = np.linalg.eigh(InertiaTensor)
    for j in range(3):
        if Ieigs[j] > symmThreshold:
            axis = remove_phase(3, Ievecs[:, j]).real
            for i, mode in enumerate(modes):
                projector[i, i_start + j] = box(modes[i].n, axis.T, CMcoords[mode.iAtom])
    return projector

def fill_rotations_molecule(projector: np.ndarray, modes: list[Mode], molecule_structure: Structure, mol_indices: list[int], symmThreshold: float = 1e-5, i_start: int = 0, dofs: list[int] | None = None):
    if dofs is None:
        dofs = [0, 1, 2]
    CMcoords = get_CMcoords(molecule_structure)
    InertiaTensor = get_inertia_tensor(molecule_structure)
    Ieigs, Ievecs = np.linalg.eigh(InertiaTensor)
    for j in dofs:
        if Ieigs[j] > symmThreshold:
            axis = remove_phase(3, Ievecs[:, j]).real
            for i, mode in enumerate(modes):
                if mode.iAtom in mol_indices:
                    val = box(modes[i].n, axis.T, CMcoords[mol_indices.index(modes[i].iAtom)])
                    projector[i, i_start + j] = val
    return projector


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
                # projector[i, i_start + k] = dot(mode.n, axis)
        projectors.append(vec)
    return projectors

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

def get_rotations_molecule(
        modes: list[Mode], molecule_structure: Structure, mol_indices: list[int], symmThreshold: float = 1e-5,
        center: np.ndarray | None = None, axes: list[np.ndarray] | None = None, ortho: bool = True
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
        vec = np.zeros(len(modes))
        for i, mode in enumerate(modes):
            if mode.iAtom in mol_indices:
                vec[i] = box(modes[i].n, axis.T, CMcoords[mol_indices.index(modes[i].iAtom)])
        projectors.append(vec)
    if ortho:
        projectors = orthogonalize_projector(np.array(projectors).T)
        projectors = [v for v in projectors.T]
    return projectors

ref_axs = {
    "x": np.array([1,0,0], dtype=np.float64),
    "y": np.array([0,1,0], dtype=np.float64),
    "z": np.array([0,0,1], dtype=np.float64)
}

def resolve_axis(structure: Structure, axis: str | list[int] | np.ndarray) -> np.ndarray:
    if isinstance(axis, np.ndarray):
        return axis
    elif isinstance(axis, str):
        return ref_axs[axis]
    elif isinstance(axis, list):
        p0 = structure.cart_coords[axis[0]]
        p1 = structure.cart_coords[axis[1]]
        return p1 - p0

def resolve_axes(structure: Structure, molecule_sets: list[dict]) -> None:
    for mset in molecule_sets:
        if "axes" in mset:
            axes = mset["axes"]
            resolved_axes = []
            for axis in axes:
                resolved_axes.append(resolve_axis(structure, axis))
            mset["axes"] = resolved_axes

def get_projector_raw(structure: Structure, trans = True, rot = True, print_removal: bool = True, molecule_sets: list[dict] | None = None) -> np.ndarray:
    if molecule_sets is None:
        molecule_sets = []
    modes = get_modes(structure)
    projectors = []
    resolve_axes(structure, molecule_sets)
    if trans:
        molecule_sets = [{"indices": list(range(len(structure.sites))), "trans": True, "rot": False}] + molecule_sets
    if rot:
        molecule_sets = [{"indices": list(range(len(structure.sites))), "trans": False, "rot": True}] + molecule_sets
    for i, mol_set in enumerate(molecule_sets):
        mol_indices = mol_set['indices']
        mol_structure = structure.copy()
        mol_structure.remove_sites([i for i in range(len(structure.sites)) if i not in mol_indices])
        mol_trans = mol_set.get("trans", False)
        mol_rot = mol_set.get("rot", False)
        center = mol_set.get("center", None)
        axes = mol_set.get("axes", None)
        ortho = mol_set.get("ortho", True)
        if mol_trans:
            projectors += get_translations_molecule(modes, mol_indices, axes=axes)
        if mol_rot:
            projectors += get_rotations_molecule(modes, mol_structure, mol_indices, center=center, axes=axes, ortho=ortho)
    projector = np.array(projectors).T
    norm_projector = projector / np.linalg.norm(projector, axis=0)
    return norm_projector


def get_projector(structure: Structure, trans = True, rot = True, molecule_sets: list[dict] | None = None, print_removal: bool = True) -> np.ndarray:
    projector_raw = get_projector_raw(structure, trans=trans, rot=rot, molecule_sets=molecule_sets, print_removal=print_removal)
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