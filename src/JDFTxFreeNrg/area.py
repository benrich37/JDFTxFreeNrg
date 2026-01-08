import numpy as np
from pymatgen.core.structure import Structure
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R



def get_box_area(rs: list[float], centers: list[np.ndarray], n_rotations: int = 50):
    # Generate uniformly distributed rotations using the Fibonacci sphere method
    idcs = np.arange(0, n_rotations, dtype=float) + 0.5
    phi = np.arccos(1 - 2*idcs/n_rotations)
    theta = np.pi * (1 + 5**0.5) * idcs
    rotations = [R.from_euler('zy', [t, p]) for t, p in zip(theta, phi)]
    # Compute box area for each rotation
    centerss = np.array([R.apply(c) for R in rotations for c in centers])
    box_areas = np.array([_get_box_area(rs, centers_rot) for centers_rot in centerss])
    # Return the minimum box area found
    return np.min(box_areas)

def _get_box_area(rs: list[float], centers: list[np.ndarray]) -> float:
    # Calculate the bounding box dimensions
    min_coords = np.min(centers - np.array(rs)[:, None], axis=0)
    max_coords = np.max(centers + np.array(rs)[:, None], axis=0)
    lengths = max_coords - min_coords
    # Calculate the surface area of the bounding box
    area = 2 * (lengths[0]*lengths[1] + lengths[1]*lengths[2] + lengths[0]*lengths[2])
    return area


def get_mc_spheres_area(rs: list[float], centers: list[np.ndarray], n_points: int = 100000, n_points_per_atom: int | None = None) -> float:
    """ Estimate the surface area of the structure using Monte Carlo sampling.

    Args:
        structure (Structure): pymatgen Structure
        n_points (int): number of random points to sample

    Returns:
        float: estimated surface area in Å^2
    """
    natoms = len(rs)
    if n_points_per_atom is None:
        n_points_per_atom = int(n_points / natoms)
    a_tot = 0.0
    for iAtom in range(natoms):
        r = rs[iAtom]
        center = centers[iAtom]
        other_rs = rs[:iAtom] + rs[iAtom+1:]
        other_centers = centers[:iAtom] + centers[iAtom+1:]
        n_outside = 0
        for _ in range(n_points_per_atom):
            # Generate random point on sphere surface
            theta = np.arccos(1 - 2 * np.random.rand())
            phi = 2 * np.pi * np.random.rand()
            x = center[0] + r * np.sin(theta) * np.cos(phi)
            y = center[1] + r * np.sin(theta) * np.sin(phi)
            z = center[2] + r * np.cos(theta)
            point = np.array([x, y, z])
            # Check if point is outside all other spheres
            dists = np.linalg.norm(point - np.array(other_centers), axis=1)
            if np.all(dists >= np.array(other_rs)):
                n_outside += 1
        a_tot += (n_outside / n_points_per_atom) * (4 * np.pi * r**2)
    return a_tot


def get_sr_spheres_area(rs: list[float], centers: list[np.ndarray], n_points: int = 100000, n_points_per_atom: int | None = None) -> float:
    # https://ljmartin.github.io/blog/21_sasa_in_numpy.html
    """ Estimate the surface area of the structure using Shrake-Rupley Algorithm.

    Args:
        structure (Structure): pymatgen Structure
        n_points (int): number of random points to sample

    Returns:
        float: estimated surface area in Å^2
    """
    natoms = len(rs)
    if n_points_per_atom is None:
        n_points_per_atom = int(n_points / natoms)
    idcs = np.arange(0, n_points_per_atom, dtype=float) + 0.5
    phi = np.arccos(1 - 2*idcs/n_points_per_atom)
    theta = np.pi * (1 + 5**0.5) * idcs
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);
    pts = np.vstack([x,y,z]).T
    sp = np.tile(pts, (centers.shape[0],1))
    sp = sp * ( np.repeat(rs, n_points_per_atom)[:,None]) #note the small buffer, 1e-5
    sp += np.repeat(centers, n_points_per_atom,axis=0)
    fraction_outside = ((cdist(sp, centers)-(rs)).min(1)>0).reshape(-1, n_points_per_atom).mean(1)
    return fraction_outside * (4*np.pi*(rs+1.4)**2)



def get_biotite_spheres_area(structure: Structure) -> float:
    """ Estimate the surface area using Biotite's method.

    Args:
        structure (Structure): pymatgen Structure

    Returns:
        float: estimated surface area in Å^2
    """
    import biotite.structure as struc
    _array = []
    for i, site in enumerate(structure.sites):
        atom = struc.Atom(
            structure.cart_coords[i],
            element=site.specie.element.symbol
        )
        _array.append(atom)
    array = struc.array(_array)
    sasa = struc.sasa(array, vdw_radii="Single")
    return sasa