import numpy as np
from pymatgen.core import Structure
from pymatgen.io.jdftx.inputs import JDFTXInfile
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from pathlib import Path
import json
import warnings
# Band-aid until I fix a UserWarning about `JDFTXOutfile` structure's having partially filled site properties
warnings.filterwarnings('ignore', category=UserWarning)


def get_pyvol_spheres_volume(
        rs: list[float], centers: list[np.ndarray], ncubes: int = 1000000, grid_spacing: float | None = None) -> float:
    from pyvolgrid import volume_from_spheres
    minx, miny, minz = [min([c[i] - r for c, r in zip(centers, rs)]) for i in range(3)]
    maxx, maxy, maxz = [max([c[i] + r for c, r in zip(centers, rs)]) for i in range(3)]
    spans = [maxx - minx, maxy - miny, maxz - minz]
    vol_tot = spans[0] * spans[1] * spans[2]
    if grid_spacing is not None:
        dstep = grid_spacing
    else:
        dstep = (vol_tot / float(ncubes)) ** (1/3)
    return volume_from_spheres(centers, rs, grid_spacing=dstep)
    
    
def get_pyvista_spheres_volume(
        rs: list[float], centers: list[np.ndarray], nslices: int = 900):
    """ Returns the mesh-integrated volume of union of spheres

    Args:
        rs (list[float]): Radii of each sphere
        centers (list[np.ndarray]): Center of each sphere
        ncubes (int): Approximate number of cubes used in integration

    Returns:
        float: Mesh-integrated volume
    """
    import pyvista as pv
    # dangle = int(nslices/((len(rs)) * 2))
    dangle = max(int(nslices/(2)), 10)
    print(f"Using {dangle} theta/phi resolution for PyVista {len(rs)} spheres")
    spheres = pv.Sphere(radius=rs[0], center=centers[0], theta_resolution=dangle, phi_resolution=dangle)
    spheres = spheres.triangulate()
    for r, c in zip(rs[1:], centers[1:]):
        sphere = pv.Sphere(radius=r, center=c, theta_resolution=dangle, phi_resolution=dangle)
        sphere = sphere.triangulate()
        spheres = spheres.boolean_union(sphere)
        spheres = spheres.triangulate()
    vol = spheres.volume
    return vol

def get_mesh_spheres_volume(
        rs: list[float], centers: list[np.ndarray], ncubes: int = 1000000, grid_spacing: float | None = None) -> float:
    """ Returns the mesh-integrated volume of union of spheres

    Args:
        rs (list[float]): Radii of each sphere
        centers (list[np.ndarray]): Center of each sphere
        ncubes (int): Approximate number of cubes used in integration

    Returns:
        float: Mesh-integrated volume
    """
    minx, miny, minz = [min([c[i] - r for c, r in zip(centers, rs)]) for i in range(3)]
    maxx, maxy, maxz = [max([c[i] + r for c, r in zip(centers, rs)]) for i in range(3)]
    spans = [maxx - minx, maxy - miny, maxz - minz]
    vol_tot = spans[0] * spans[1] * spans[2]
    if grid_spacing is not None:
        dstep = grid_spacing
    else:
        dstep = (vol_tot / float(ncubes)) ** (1/3)
    nx, ny, nz = [int(np.round(sp / dstep)) for sp in spans]
    # print(f"Using {np.prod([nx, ny, nz])} cubes for volume integration (approx {ncubes} requested)")
    x, y, z = np.meshgrid(
        np.linspace(minx, maxx, nx),
        np.linspace(miny, maxy, ny),
        np.linspace(minz, maxz, nz),
    )
    x_flat, y_flat, z_flat = x.flatten(), y.flatten(), z.flatten()
    points = np.vstack((x_flat, y_flat, z_flat)).T
    distancess = [np.linalg.norm(points - c, axis=1) for c in centers]
    inside_any_sphere = np.zeros(len(points), dtype=bool)
    for r, distances in zip(rs, distancess):
        inside_sphere = distances <= r
        inside_any_sphere = inside_any_sphere | inside_sphere
    vol = np.sum(inside_any_sphere)
    dV = ((maxx - minx)/nx) * ((maxy - miny)/ny) * ((maxz - minz)/nz)
    return vol * dV

def get_monte_carlo_spheres_volume(
        rs: list[float], centers: list[np.ndarray], npoints: int = 1000000) -> float:
    """ Returns the Monte-Carlo-integrated volume of union of spheres

    Args:
        rs (list[float]): Radii of each sphere
        centers (list[np.ndarray]): Center of each sphere
        npoints (int): Number of samples used in integration

    Returns:
        float: Monte-Carlo-integrated volume
    """
    rs = np.array(rs)
    centers = np.array(centers)
    min_coords = np.min([c - r for c, r in zip(centers, rs)], axis=0)
    max_coords = np.max([c + r for c, r in zip(centers, rs)], axis=0)
    cube_vol = np.prod(max_coords - min_coords)
    def f(x):
        dminr = np.linalg.norm(x - centers, axis=1) - rs
        return float(np.any(dminr <= 0))*cube_vol
    points = np.random.uniform(min_coords, max_coords, size=(npoints, 3))
    results = np.array([f(point) for point in points])
    mean = np.mean(results)
    sem = np.std(results) / np.sqrt(npoints)
    return mean, sem

def get_vdw_volume(structure: Structure, npoints: int | None = None, grid_spacing: float | None = None, method="MC") -> float:
    """ Returns the van der waals volume of a structure 

    Args:
        structure (Structure): Structure to evaluate vdw volume of
        npoints (int): Number of samples used in monte-carlo integration

    Returns:
        float: Monte-Carlo-integrated van der waals volume in A^3
    """
    if (npoints is None) and (grid_spacing is None):
        raise ValueError("Must specify either npoints or grid_spacing for volume calculation")
    rs = [site.specie.van_der_waals_radius for site in structure.sites]
    centers = [site.coords for site in structure.sites]
    if method.lower() == "mc":
        vol, _ = get_monte_carlo_spheres_volume(rs, centers, npoints=npoints)
    elif method.lower() == "mesh":
        vol = get_mesh_spheres_volume(rs, centers, grid_spacing=grid_spacing)
    elif method.lower() == "pyvol":
        vol = get_pyvol_spheres_volume(rs, centers, grid_spacing=grid_spacing)
    else:
        raise ValueError(f"Unknown method {method} for vdw volume calculation (choose 'MC', 'Mesh', or 'PyVol')")
    return vol


class StructureVolume(Structure):

    cache: dict | None = None
    structure: Structure
    method: str = "MC"
    npoint_default: int = 1000000
    # TODO: benchmark the actual grid spacing instead of the npoints so we know where this value stands
    grid_spacing_default: float = 0.01

    def set_cache_parent(self, cache_parent: Path | None = None):
        if cache_parent is not None:
            self.cache_dir = cache_parent / "vdw_radii_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.load_cache()
        else:
            self.cache_dir = None
            
    @classmethod
    def from_structure(cls, structure: Structure, cache_parent: Path | None = None, method: str = "MC"):
        struct_vol = cls.from_sites(structure.sites)
        struct_vol.set_cache_parent(cache_parent)
        struct_vol.method = method
        return struct_vol
    
    @classmethod
    def from_calc_dir(cls, calc_dir: Path, use_in: bool = False, method: str = "MC"):
        infile = calc_dir / "in"
        outfile = calc_dir / "out"
        if infile.exists() and use_in:
            structure = JDFTXInfile.from_file(infile).to_pmg_structure()
        else:
            structure = JDFTXOutfile.from_file(outfile).structure
        # structure = JDFTXOutfile.from_file(calc_dir / "out").structure
        struct_vol = cls.from_sites(structure.sites)
        struct_vol.set_cache_parent(calc_dir)
        struct_vol.method = method
        return struct_vol
    
    @classmethod
    def from_outfile_path(cls, outfile_path: Path, method: str = "MC"):
        return cls.from_calc_dir(outfile_path.parent, use_in=False, method=method)

    def clear_cache(self):
        self.cache = {}
        self.backup_cache()

    def backup_cache(self):
        if self.cache_dir is not None and self.cache is not None:
            cache_file = self.cache_dir / "vdw_volumes_cache.json"
            with open(cache_file, 'w') as f:
                json.dump(self.cache, f, indent=4)

    def load_cache(self):
        if self.cache_dir is not None:
            cache_file = self.cache_dir / "vdw_volumes_cache.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
                if self.method not in self.cache:
                    self.cache[self.method] = {}
            else:
                self.cache = {self.method: {}}
        else:
            self.cache = None

    def get_idcs_key(self, idcs: list[int]) -> str:
        idcs = sorted(idcs)
        idx_key = "_".join([str(i) for i in idcs])
        return idx_key
    
    def get_npoints_key(self, npoints: int) -> str:
        return f"n: {npoints}"
    
    def get_grid_spacing_key(self, grid_spacing: float) -> str:
        return f"gs: {grid_spacing:.6f}"
    
    def parse_npoints_key(self, npoints_key: str) -> int:
        return int(npoints_key.split(":")[1].strip())
    
    def parse_grid_spacing_key(self, grid_spacing_key: str) -> float:
        return float(grid_spacing_key.split(":")[1].strip())
    
    def is_npoints_key(self, key: str) -> bool:
        return key.startswith("n:")
    
    def is_grid_spacing_key(self, key: str) -> bool:
        return key.startswith("gs:")

    def compute_volume(self, idcs: list[int], npoints: int | None = None, grid_spacing: float | None = None,) -> float:
        idcs = sorted(idcs)
        substructure = Structure.from_sites(
            [self[i] for i in range(len(self.sites)) if i in idcs])
        vol = get_vdw_volume(substructure, npoints=npoints, method=self.method, grid_spacing=grid_spacing)
        if self.cache is not None:
            if self.method not in self.cache:
                self.cache[self.method] = {}
            idx_key = self.get_idcs_key(idcs)
            if idx_key not in self.cache[self.method]:
                self.cache[self.method][idx_key] = {}
            self.cache[self.method][idx_key][self.get_npoints_key(npoints)] = vol
            self.backup_cache()
        return vol
    
    # TODO: _get_volume_idcs_mc and _get_volume_idcs_gs have a lot of repeated code, refactor
    def _get_volume_idcs_mc(self, idcs: list[int], npoints: int | None = None, grid_spacing: float | None = None) -> float:
        # Assues idx key is in cache and method is "MC"
        idx_key = self.get_idcs_key(idcs)
        if npoints is None:
            if len(self.cache[self.method][idx_key]) == 0:
                return self.compute_volume(idcs, npoints=self.npoint_default)
            else:
                npoints_keys = [k for k in self.cache[self.method][idx_key].keys() if self.is_npoints_key(k)]
                npointss = [self.parse_npoints_key(k) for k in npoints_keys]
                max_npoints = max(npointss)
                return self.cache[self.method][idx_key][self.get_npoints_key(max_npoints)]
        elif self.get_npoints_key(npoints) in self.cache[self.method][idx_key]:
            return self.cache[self.method][idx_key][self.get_npoints_key(npoints)]
        else:
            return self.compute_volume(idcs, npoints=npoints)
        
    def _get_volume_idcs_gs(self, idcs: list[int], npoints: int | None = None, grid_spacing: float | None = None) -> float:
        # Assumes idx key is in cache and method is not "MC"
        idx_key = self.get_idcs_key(idcs)
        if grid_spacing is None:
            if len(self.cache[self.method][idx_key]) == 0:
                return self.compute_volume(idcs, grid_spacing=self.grid_spacing_default)
            else:
                grid_spacing_keys = [k for k in self.cache[self.method][idx_key].keys() if self.is_grid_spacing_key(k)]
                grid_spacings = [self.parse_grid_spacing_key(k) for k in grid_spacing_keys]
                min_grid_spacing = min(grid_spacings)
                return self.cache[self.method][idx_key][self.get_grid_spacing_key(min_grid_spacing)]
        elif self.get_grid_spacing_key(grid_spacing) in self.cache[self.method][idx_key]:
            return self.cache[self.method][idx_key][self.get_grid_spacing_key(grid_spacing)]
        else:
            return self.compute_volume(idcs, grid_spacing=grid_spacing)
    
    def _get_volume_idcs(self, idcs: list[int], npoints: int | None = None, grid_spacing: float | None = None) -> float:
        # Assumes idx key is in cache
        if self.method == "MC":
            return self._get_volume_idcs_mc(idcs, npoints=npoints)
        else:
            return self._get_volume_idcs_gs(idcs, grid_spacing=grid_spacing)

    def get_volume(self, idcs: list[int] | None = None, npoints: int | None = None, grid_spacing: float | None = None) -> float:
        if npoints is not None:
            npoints = int(npoints)
        if idcs is None:
            idcs = list(range(len(self.sites)))
        idcs = sorted(idcs)
        if self.cache is not None:
            if self.method not in self.cache:
                self.cache[self.method] = {}
            idx_key = self.get_idcs_key(idcs)
            if idx_key in self.cache[self.method]:
                return self._get_volume_idcs(idcs, npoints=npoints, grid_spacing=grid_spacing)
            else:
                return self.compute_volume(idcs, npoints=npoints, grid_spacing=grid_spacing)