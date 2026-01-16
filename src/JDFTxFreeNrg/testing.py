import numpy as np
from JDFTxFreeNrg.volume import get_mesh_spheres_volume, get_monte_carlo_spheres_volume, get_pyvol_spheres_volume
from JDFTxFreeNrg.solv_entropy import get_vcav, eff_volume, _get_solv_entropy_trans
import matplotlib.pyplot as plt
import timeit
from pathlib import Path
import zipfile
import requests
import inspect
from os import remove
import pickle

def anl_spheres_area(rs: list[float], centers: list[np.ndarray]) -> float:
    assert len(rs) == len(centers), "Number of radii must match number of centers."
    if len(rs) == 1:
        return anl_sphere_area(rs[0])
    elif len(rs) == 2:
        return anl_2sphere_union_area(rs[0], rs[1], np.linalg.norm(centers[0]-centers[1]))
    else:
        raise NotImplementedError("Analytical area calculation only implemented for up to 2 spheres.")
    
def anl_sphere_area(r: float) -> float:
    """Return surface area of a sphere.

    Args:
        r (float): Radius of the sphere.

    Returns:
        float: Surface area of the sphere.
    """
    return 4 * np.pi * r**2

def anl_2sphere_intersection_area(r1: float, r2: float, l: float):
    if l >= r1 + r2:
        return 0.0
    # Break down the distance between centers into four parts:
    ## l = A + B + C + D
    ## A = distance from center 1 to closest point on sphere 2 (= l - r2)
    ## B = cap height for cap from sphere 2
    ## C = cap height for cap from sphere 1
    ## D = distance from center 2 to closest point on sphere 1 (= l - r1)
    # In the case where the smaller sphere is completely inside the larger sphere, this equation doesn't work
    # and the intersection is just the area of the smaller sphere
    r1, r2 = max(r1, r2), min(r1, r2)
    if l + r2 <= r1:
        return anl_sphere_volume(r2)
    # We can solve A + B = m, as m being the distance from first sphere where the two laguerre powers are equal,
    # which equates to solving x**2 - r1**2 = (x-l)**2 - r2**2, giving x = ((r1**2) - (r2**2) + (l**2))/(2*l).
    m = ((r1**2) - (r2**2) + (l**2))/(2*l)
    A = l - r2
    D = l - r1
    B = m - A
    C = l - (A + B + D)
    Acap1 = 2 * np.pi * r1 * C
    Acap2 = 2 * np.pi * r2 * B
    Aintersection = Acap1 + Acap2
    return Aintersection


def anl_2sphere_union_area(r1: float, r2: float, l: float):
    return anl_sphere_area(r1) + anl_sphere_area(r2) - anl_2sphere_intersection_area(r1, r2, l)

def anl_spheres_volume(rs: list[float], centers: list[np.ndarray]) -> float:
    assert len(rs) == len(centers), "Number of radii must match number of centers."
    if len(rs) == 1:
        return anl_sphere_volume(rs[0])
    elif len(rs) == 2:
        return anl_2sphere_union_volume(rs[0], rs[1], np.linalg.norm(centers[0]-centers[1]))
    else:
        raise NotImplementedError("Analytical volume calculation only implemented for up to 2 spheres.")

def anl_sphere_volume(r: float) -> float:
    """Return volume of a sphere.

    Args:
        r (float): Radius of the sphere.

    Returns:
        float: Volume of the sphere.
    """
    return (4/3) * np.pi * r**3

def anl_2sphere_intersection_volume(r1: float, r2: float, l: float):
    if l >= r1 + r2:
        return 0.0
    # Break down the distance between centers into four parts:
    ## l = A + B + C + D
    ## A = distance from center 1 to closest point on sphere 2 (= l - r2)
    ## B = cap height for cap from sphere 2
    ## C = cap height for cap from sphere 1
    ## D = distance from center 2 to closest point on sphere 1 (= l - r1)
    # In the case where the smaller sphere is completely inside the larger sphere, this equation doesn't work
    # and the intersection is just the vol of the smaller sphere
    r1, r2 = max(r1, r2), min(r1, r2)
    if l + r2 <= r1:
        return anl_sphere_volume(r2)
    # We can solve A + B = m, as m being the distance from first sphere where the two laguerre powers are equal,
    # which equates to solving x**2 - r1**2 = (x-l)**2 - r2**2, giving x = ((r1**2) - (r2**2) + (l**2))/(2*l).
    m = ((r1**2) - (r2**2) + (l**2))/(2*l)
    A = l - r2
    D = l - r1
    B = m - A
    C = l - (A + B + D)
    # Now compute cap volumes
    Vcap1 = np.pi*(C**2)*(r1 - (C/3))
    Vcap2 = np.pi*(B**2)*(r2 - (B/3))
    # And use cap volumes to get intersection volume (negative if no intersection, so clamp to zero)
    Vintersection = max(0.0, Vcap1 + Vcap2)
    return Vintersection

def anl_2sphere_union_volume(r1: float, r2: float, l: float):
    """Return volume of a sphere.

    Args:
        r (float): Radius of the sphere.

    Returns:
        float: Volume of the sphere.
    """
    # Start off with total volume of two spheres
    vol = anl_sphere_volume(r1) + anl_sphere_volume(r2)
    # Use intersection volume to subtract overlap
    vol -= anl_2sphere_intersection_volume(r1, r2, l)
    return vol

def time_mesh_spheres_volume(rs: list[float], centers: list[np.ndarray], ncubes: int = 1000000, runs=5):
    def time_fn():
        get_mesh_spheres_volume(rs, centers, ncubes=ncubes)
    code_snippet = f'time_fn()'
    return timeit.timeit(code_snippet, globals={"time_fn": time_fn}, number=runs)

def time_mc_spheres_volume(rs: list[float], centers: list[np.ndarray], npoints: int = 1000000, runs=5):
    def time_fn():
        get_monte_carlo_spheres_volume(rs, centers, npoints=npoints)
    code_snippet = f'time_fn()'
    return timeit.timeit(code_snippet, globals={"time_fn": time_fn}, number=runs)

def time_pyvol_spheres_volume(rs: list[float], centers: list[np.ndarray], ncubes: int = 1000000, runs=5):
    def time_fn():
        get_pyvol_spheres_volume(rs, centers, ncubes=ncubes)
    code_snippet = f'time_fn()'
    return timeit.timeit(code_snippet, globals={"time_fn": time_fn}, number=runs)

def get_mesh_volume_samples(rs: list[float], centers: list[np.ndarray], nsampless: list[int], time_only: bool = False):
    v_meshs = []
    t_meshs = []
    for ns in nsampless:
        t_mesh = time_mesh_spheres_volume(rs, centers, ncubes=int(ns), runs=1)
        t_meshs.append(t_mesh)
        if not time_only:
            v_mesh = get_mesh_spheres_volume(rs, centers, ncubes=int(ns))
            v_meshs.append(v_mesh)
    return v_meshs, t_meshs

def get_mc_volume_samples(rs: list[float], centers: list[np.ndarray], nsampless: list[int], zscore: float = 1.96, time_only: bool = False):
    v_mcs = []
    t_mcs = []
    unc_mcs = []
    for ns in nsampless:
        t_mc = time_mc_spheres_volume(rs, centers, npoints=int(ns), runs=1)
        t_mcs.append(t_mc)
        if not time_only:
            v_mc, sem_mc = get_monte_carlo_spheres_volume(rs, centers, npoints=int(ns))
            v_mcs.append(v_mc)
            unc_mcs.append(sem_mc*zscore)
    return v_mcs, t_mcs, unc_mcs

def get_pyvol_volume_samples(rs: list[float], centers: list[np.ndarray], ncubess: list[float], time_only: bool = False):
    v_pvs = []
    t_pvs = []
    for nc in ncubess:
        t_pv = time_pyvol_spheres_volume(rs, centers, ncubes=int(nc), runs=1)
        t_pvs.append(t_pv)
        if not time_only:
            v_pv = get_pyvol_spheres_volume(rs, centers, ncubes=nc)
            v_pvs.append(v_pv)
    return v_pvs, t_pvs


def get_mc_generic_samples(
        rss: list[float],
        centerss: list[np.ndarray],
        treat_func: callable,
        err_func: callable,
        nsampless: list[int],
        time_only: bool = False,
        zscore: float = 1.96
        ):
    n_collections = len(rss)
    assert n_collections == len(centerss), "Number of radius collections must match number of center collections."
    out_generics = []
    out_unc_generics = []
    t_generics = []
    for ns in nsampless:
        t_generic = 0.0
        for i in range(n_collections):
            t_generic += time_mc_spheres_volume(rss[i], centerss[i], npoints=int(ns), runs=1)
        t_generics.append(t_generic)
        if not time_only:
            v_mcs = []
            unc_mcs = []
            for i in range(n_collections):
                v_mc, sem_mc = get_monte_carlo_spheres_volume(rss[i], centerss[i], npoints=int(ns))
                v_mcs.append(v_mc)
                unc_mcs.append(sem_mc*zscore)
            v_generic = treat_func(v_mcs)
            out_generics.append(v_generic)
            out_unc_generics.append(err_func(unc_mcs))
    return out_generics, t_generics, out_unc_generics


def get_mesh_generic_samples(
        rss: list[float],
        centerss: list[np.ndarray],
        treat_func: callable,
        nsampless: list[int],
        time_only: bool = False):
    n_collections = len(rss)
    assert n_collections == len(centerss), "Number of radius collections must match number of center collections."
    out_generics = []
    t_generics = []
    for ns in nsampless:
        t_generic = 0.0
        for i in range(n_collections):
            t_generic += time_mesh_spheres_volume(rss[i], centerss[i], ncubes=int(ns), runs=1)
        t_generics.append(t_generic)
        if not time_only:
            vss = [get_mesh_spheres_volume(rss[i], centerss[i], ncubes=int(ns)) for i in range(n_collections)]
            v_generic = treat_func(vss)
            out_generics.append(v_generic)
    return out_generics, t_generics

def get_pyvol_generic_samples(
        rss: list[float],
        centerss: list[np.ndarray],
        treat_func: callable,
        nsampless: list[int],
        time_only: bool = False):
    n_collections = len(rss)
    assert n_collections == len(centerss), "Number of radius collections must match number of center collections."
    out_generics = []
    t_generics = []
    for ns in nsampless:
        t_generic = 0.0
        for i in range(n_collections):
            t_generic += time_pyvol_spheres_volume(rss[i], centerss[i], ncubes=int(ns), runs=1)
        t_generics.append(t_generic)
        if not time_only:
            vss = [get_pyvol_spheres_volume(rss[i], centerss[i], ncubes=int(ns)) for i in range(n_collections)]
            v_generic = treat_func(vss)
            out_generics.append(v_generic)
    return out_generics, t_generics





def plot_volume_accuracy(
        vol_true: float, 
        labels: list[str],
        vol_numerics: dict[str, list[float]],
        time_numerics: dict[str, list[float]],
        dev_numerics: dict[str, list[float]] = None,
        colors: dict[str, str] = None,
        standalone_vol_labels: list[str] | None = None,
        computed_label: str = 'Computed volume'
        ):
    if colors is None:
        colors = {label: color for label, color in zip(labels, ['red', 'blue', 'orange', 'purple', 'cyan'])}
    if dev_numerics is None:
        dev_numerics = {}
    err_numerics = {label: np.array([((v - vol_true) / vol_true) for v in vol_numerics[label]])*100 for label in labels}
    fig, ax = plt.subplots(nrows=2 if standalone_vol_labels is None else 3, sharex=True)
    ax[0].axhline(y=0.0, color='green')
    for label in labels:
        ax[0].plot(time_numerics[label], err_numerics[label], marker='o', label=label, color=colors[label])
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel(r'Signed % error')
    ax[0].legend()
    for label in labels:
        if label in dev_numerics:
            ax[1].scatter(time_numerics[label], vol_numerics[label], marker='o', label=label, color=colors[label])
            ax[1].errorbar(time_numerics[label], vol_numerics[label], yerr=dev_numerics[label], color=colors[label], fmt='o')
            ax[1].plot(time_numerics[label], vol_numerics[label], color=colors[label], zorder=3)
        else:
            ax[1].plot(time_numerics[label], vol_numerics[label], marker='o', label=label, color=colors[label])
    ax[1].axhline(y=vol_true, color='green', linestyle='--', label='Analytical')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel(computed_label)
    ax[1].legend()
    if standalone_vol_labels is not None:
        for label in standalone_vol_labels:
            assert label in labels, f"Standalone volume label '{label}' not found in labels."
            if label in dev_numerics:
                ax[2].scatter(time_numerics[label], vol_numerics[label], marker='o', label=label, color=colors[label])
                ax[2].errorbar(time_numerics[label], vol_numerics[label], yerr=dev_numerics[label], color=colors[label], fmt='o')
                ax[2].plot(time_numerics[label], vol_numerics[label], color=colors[label], zorder=3)
            else:
                ax[2].plot(time_numerics[label], vol_numerics[label], marker='o', label=label, color=colors[label])
        ax[2].axhline(y=vol_true, color='green', linestyle='--', label='Analytical')
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel(computed_label)
        ax[2].legend()
        ax[2].ticklabel_format(useOffset=False)
    # plt.show()
    fig.tight_layout()
    return fig, ax

def gen_random_spheres(num_spheres: int, min_r: float = 0.8, max_r: float = 1.2, min_dist: float = 1.2, max_dist: float = 1.5):
    rs = np.random.random(num_spheres) * (max_r - min_r) + min_r
    centers = [np.zeros(3)]
    for _ in range(num_spheres-1):
        _dist = np.random.random() * (max_dist - min_dist) + min_dist
        _center = np.random.random(3)
        _center *= _dist/np.linalg.norm(_center)
        centers.append(_center)
    return rs.tolist(), centers

def ensure_random_spheres(rs: list[float] | None = None, centers: list[np.ndarray] | None = None, num_spheres: int = 5, min_r: float = 0.5, max_r: float = 1.5):
    _rs, _centers = gen_random_spheres(num_spheres, min_r, max_r)
    if rs is None:
        rs = _rs
    if centers is None:
        centers = _centers
    return rs, centers






def test_generic(
        rss: list[list[float]], centerss: list[list[np.ndarray]], nsampless: list[int], 
        treat_func: callable, err_func: callable,
        mesh_sample_scale: float = 150., pyvol_sample_scale = 3000.,
        computed_label: str = 'Computed volume',
        standalone_vol_labels: list[str] = ["Mesh (PyVol)"],
        ):
    v_anl = treat_func([anl_spheres_volume(rss[i], centerss[i]) for i in range(len(rss))])
    v_mcs, t_mcs, dev_mcs = get_mc_generic_samples(
        rss,
        centerss,
        treat_func,
        err_func,
        np.array(nsampless),
        )
    v_meshs, t_meshs = get_mesh_generic_samples(
        rss,
        centerss,
        treat_func,
        np.array(nsampless)*mesh_sample_scale,
        )
    v_pyvols, t_pyvols = get_pyvol_generic_samples(
        rss,
        centerss,
        treat_func,
        np.array(nsampless)*pyvol_sample_scale,
        )
    fig, ax = plot_volume_accuracy(
        v_anl,
        [
            "Monte Carlo", "Mesh", "Mesh (PyVol)", 
        ],
        {
            "Monte Carlo": v_mcs,
            "Mesh": v_meshs,
            "Mesh (PyVol)": v_pyvols,
        },
        {
            "Monte Carlo": t_mcs,
            "Mesh": t_meshs,
            "Mesh (PyVol)": t_pyvols,
        },
        {"Monte Carlo": dev_mcs},
        standalone_vol_labels=standalone_vol_labels,
        computed_label=computed_label,
        )
    return fig, ax

def test_single_sphere_volume(r: float | None = None, center: np.ndarray | None = None, nsampless: list[int] | None = None, mesh_sample_scale: float = 150., pyvol_sample_scale = 5000):
    rs, centers = ensure_random_spheres(None if r is None else [r], None if center is None else [center], num_spheres=1)
    r = rs[0]
    center = centers[0]
    if nsampless is None:
        nsampless = [1e3, 1e4, 1e5]
    def treat_func(vs: list[float]) -> float:
        return vs[0]
    def err_func(uncs: list[float]) -> float:
        return uncs[0]
    fig, ax = test_generic(
        [rs],
        [centers],
        nsampless,
        treat_func,
        err_func,
        mesh_sample_scale,
        pyvol_sample_scale,
    )
    fig.suptitle('Single Sphere of r={:.2f} Volume Accuracy'.format(r))
    fig.tight_layout()
    plt.show()

def test_double_sphere_volume(rs: list[float] | None = None, centers: list[np.ndarray] | None = None, nsampless: list[int] | None = None, mesh_sample_scale: float = 150., pyvol_sample_scale = 3000.):
    rs, centers = ensure_random_spheres(rs, centers, num_spheres=2)
    if nsampless is None:
        nsampless = [1e3, 1e4, 1e5]
    def treat_func(vs: list[float]) -> float:
        return vs[0]
    def err_func(uncs: list[float]) -> float:
        return uncs[0]
    fig, ax = test_generic(
        [rs],
        [centers],
        nsampless,
        treat_func,
        err_func,
        mesh_sample_scale,
        pyvol_sample_scale,
    )
    fig.suptitle('Double Sphere of r1={:.2f}, r2={:.2f}, d={:.2f} Volume Accuracy'.format(rs[0], rs[1], np.linalg.norm(centers[0]-centers[1])))
    fig.tight_layout()
    plt.show()

def test_vcav(solvent_rs: list[float] | None = None, solvent_centers: list[np.ndarray] | None = None, solute_rs: list[float] | None = None, solute_centers: list[np.ndarray] | None = None, nsampless: list[int] | None = None, mesh_sample_scale: float = 150., pyvol_sample_scale = 3000., na_solute: int = 2, na_solvent: int = 2, scale_free_volume: float = 2.0):
    solvent_rs, solvent_centers = ensure_random_spheres(solvent_rs, solvent_centers, num_spheres=na_solvent)
    solute_rs, solute_centers = ensure_random_spheres(solute_rs, solute_centers, num_spheres=na_solute)
    # Solvent molecules with greater free space will have errors effectively muted out
    # Scale needs to be >1 to ensure positive free volume
    avg_volume_per_molecule = anl_spheres_volume(solvent_rs, solvent_centers) * scale_free_volume
    if nsampless is None:
        nsampless = [1e3, 1e4, 3e4, 5e4]
    def treat_func(vs: list[float]) -> float:
        v_solv = vs[0]
        v_solute = vs[1]
        vfree = avg_volume_per_molecule - v_solv
        vcav = get_vcav(v_solute, vfree)
        return vcav
    def err_func(uncs: list[float]) -> float:
        return np.nan
    fig, ax = test_generic(
        [solvent_rs, solute_rs],
        [solvent_centers, solute_centers],
        nsampless,
        treat_func,
        err_func,
        mesh_sample_scale,
        pyvol_sample_scale,
    )
    plt.show()


def test_eff_volume(solvent_rs: list[float] | None = None, solvent_centers: list[np.ndarray] | None = None, solute_rs: list[float] | None = None, solute_centers: list[np.ndarray] | None = None, nsampless: list[int] | None = None, mesh_sample_scale: float = 150., pyvol_sample_scale = 3000., na_solute: int = 2, na_solvent: int = 2, scale_free_volume: float = 2.0):
    solvent_rs, solvent_centers = ensure_random_spheres(solvent_rs, solvent_centers, num_spheres=na_solvent)
    solute_rs, solute_centers = ensure_random_spheres(solute_rs, solute_centers, num_spheres=na_solute)
    # Solvent molecules with greater free space will have errors effectively muted out
    # Scale needs to be >1 to ensure positive free volume
    avg_volume_per_molecule = anl_spheres_volume(solvent_rs, solvent_centers) * scale_free_volume
    if nsampless is None:
        nsampless = [1e3, 1e4, 3e4, 5e4]
    def treat_func(vs: list[float]) -> float:
        v_solv = vs[0]
        v_solute = vs[1]
        vfree = avg_volume_per_molecule - v_solv
        veff = eff_volume(v_solute, v_solv, vfree)
        return veff
    def err_func(uncs: list[float]) -> float:
        return np.nan
    fig, ax = test_generic(
        [solvent_rs, solute_rs],
        [solvent_centers, solute_centers],
        nsampless,
        treat_func,
        err_func,
        mesh_sample_scale,
        pyvol_sample_scale,
    )
    plt.show()


def test_solve_entropy_trans(
        solvent_rs: list[float] | None = None, solvent_centers: list[np.ndarray] | None = None, solute_rs: list[float] | None = None, solute_centers: list[np.ndarray] | None = None, 
        nsampless: list[int] | None = None, mesh_sample_scale: float = 150., pyvol_sample_scale = 3000., 
        na_solute: int = 2, na_solvent: int = 2, scale_free_volume: float = 2.0, T: float = 300., m_solute: float = 18.0):
    solvent_rs, solvent_centers = ensure_random_spheres(solvent_rs, solvent_centers, num_spheres=na_solvent)
    solute_rs, solute_centers = ensure_random_spheres(solute_rs, solute_centers, num_spheres=na_solute)
    # Solvent molecules with greater free space will have errors effectively muted out
    # Scale needs to be >1 to ensure positive free volume
    avg_volume_per_molecule = anl_spheres_volume(solvent_rs, solvent_centers) * scale_free_volume
    if nsampless is None:
        nsampless = [1e3, 1e4, 3e4, 5e4]
    def treat_func(vs: list[float]) -> float:
        v_solv = vs[0]
        v_solute = vs[1]
        vfree = avg_volume_per_molecule - v_solv
        veff = eff_volume(v_solute, v_solv, vfree)
        solv_entr_trans = _get_solv_entropy_trans(m_solute, T, veff, d=3) * T
        return solv_entr_trans
    def err_func(uncs: list[float]) -> float:
        return np.nan
    fig, ax = test_generic(
        [solvent_rs, solute_rs],
        [solvent_centers, solute_centers],
        nsampless,
        treat_func,
        err_func,
        mesh_sample_scale,
        pyvol_sample_scale,
        computed_label = "Computed T*S\n(eV)"
    )
    plt.show()


def test_relative_solve_entropy_trans(
        solvent_rs: list[float] | None = None, solvent_centers: list[np.ndarray] | None = None, 
        solute_rs1: list[float] | None = None, solute_centers1: list[np.ndarray] | None = None, 
        solute_rs2: list[float] | None = None, solute_centers2: list[np.ndarray] | None = None, 
        nsampless: list[int] | None = None, mesh_sample_scale: float = 150., pyvol_sample_scale = 5000., 
        na_solute1: int = 2, na_solute2: int = 2, na_solvent: int = 2, scale_free_volume: float = 2.0, T: float = 300., m_solute: float = 18.0,
        standalone_vol_labels: list[str] = ["Mesh (PyVol)", "Mesh"]
        ):
    solvent_rs, solvent_centers = ensure_random_spheres(solvent_rs, solvent_centers, num_spheres=na_solvent)
    solute_rs1, solute_centers1 = ensure_random_spheres(solute_rs1, solute_centers1, num_spheres=na_solute1)
    solute_rs2, solute_centers2 = ensure_random_spheres(solute_rs2, solute_centers2, num_spheres=na_solute2)
    # Solvent molecules with greater free space will have errors effectively muted out
    # Scale needs to be >1 to ensure positive free volume
    avg_volume_per_molecule = anl_spheres_volume(solvent_rs, solvent_centers) * scale_free_volume
    if nsampless is None:
        nsampless = [1e4, 2e4, 3e4, 5e4]
    def treat_func(vs: list[float]) -> float:
        v_solv = vs[0]
        v_solute1 = vs[1]
        v_solute2 = vs[2]
        vfree = avg_volume_per_molecule - v_solv
        veff1 = eff_volume(v_solute1, v_solv, vfree)
        solv_entr_trans1 = _get_solv_entropy_trans(m_solute, T, veff1, d=3)
        veff2 = eff_volume(v_solute2, v_solv, vfree)
        solv_entr_trans2 = _get_solv_entropy_trans(m_solute, T, veff2, d=3)
        return (solv_entr_trans1 - solv_entr_trans2)*T
    def err_func(uncs: list[float]) -> float:
        return np.nan
    fig, ax = test_generic(
        [solvent_rs, solute_rs1, solute_rs2],
        [solvent_centers, solute_centers1, solute_centers2],
        nsampless,
        treat_func,
        err_func,
        mesh_sample_scale,
        pyvol_sample_scale,
        computed_label = r"Computed T*$\Delta$S" + "\n(eV)",
        standalone_vol_labels=standalone_vol_labels,
    )
    plt.show()

def getcwd2():
    return Path(inspect.stack()[-1].filename).parent

def download_freesolv_data(version='0.51'):
    data_path = Path(getcwd2()) / 'data'
    data_path.mkdir(parents=True, exist_ok=True)
    zip_file_path = data_path / f'FreeSolv-{version}.zip'
    r = requests.get(
        f'https://escholarship.org/content/qt6sd403pz/supp/FreeSolv-{version}.zip', 
        allow_redirects=True, 
        # I do not like having to pretend to be a browser, but the server seems to block non-browser user agents
        # headers={'User-Agent': 'JDFTxFreeNrg/1.0'}
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        )
    with open(zip_file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    with open(zip_file_path, 'rb') as f:
        with zipfile.ZipFile(f) as zf:
            zf.extractall(path=data_path)
    remove(zip_file_path)

def get_freesolv_pickle(version='0.51', rel_path = Path("database.pickle")):
    data_path = Path(getcwd2()) / 'data' / f'FreeSolv-{version}'
    pickle_file_path = data_path / rel_path
    if not pickle_file_path.exists():
        download_freesolv_data(version=version)
    return pickle_file_path

def get_freesolv_database(version='0.51'):
    pickle_file_path = get_freesolv_pickle(version=version, rel_path = Path("database.pickle"))
    with open(pickle_file_path, 'rb') as f:
        database = pickle.load(f, encoding='latin1')
    return database
# nsampless = [1e3, 2e3, 3e3]
# test_single_sphere_volume(nsampless=nsampless)
# test_double_sphere_volume(nsampless=nsampless)
# test_vcav(nsampless=nsampless)
# test_eff_volume(nsampless=nsampless)
# test_solve_entropy_trans(nsampless=nsampless)
# test_relative_solve_entropy_trans(nsampless=nsampless)


