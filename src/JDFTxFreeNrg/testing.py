import numpy as np
# import time
from time import time
from JDFTxFreeNrg.solv_entropy import get_monte_carlo_spheres_volume, get_mesh_spheres_volume

def anl_sphere_volume(r: float) -> float:
    """Return volume of a sphere.

    Args:
        r (float): Radius of the sphere.

    Returns:
        float: Volume of the sphere.
    """
    return (4/3) * np.pi * r**3

def anl_2sphere_union_volume(r1: float, r2: float, l: float):
    """Return volume of a sphere.

    Args:
        r (float): Radius of the sphere.

    Returns:
        float: Volume of the sphere.
    """
    m = ((r2**2) - (r1**2) + (l**2))/(2*l)
    h1 = r1 + l - m
    h2 = r2 - m
    V11 = np.pi*(h1**2)*(r1-(h1/3))
    V21 = np.pi*(h2**2)*(r2-(h2/3))
    V2 = anl_sphere_volume(r2)
    return V11 + V2 - V21

def get_mesh_volume_samples(rs: list[float], centers: list[np.ndarray], nsampless: list[int]):
    v_meshs = []
    t_meshs = []
    for ns in nsampless:
        start = time()
        v_mesh = get_mesh_spheres_volume(rs, centers, ncubes=int(ns))
        end = time()
        v_meshs.append(v_mesh)
        t_meshs.append(end - start)
    return v_meshs, t_meshs

def get_mc_volume_samples(rs: list[float], centers: list[np.ndarray], nsampless: list[int], zscore: float = 1.96):
    v_mcs = []
    t_mcs = []
    unc_mcs = []
    for ns in nsampless:
        start = time()
        v_mc, sem_mc = get_monte_carlo_spheres_volume(rs, centers, npoints=int(ns))
        end = time()
        v_mcs.append(v_mc)
        t_mcs.append(end - start)
        # unc_mcs.append(sem_mc*v_mc)
        unc_mcs.append(sem_mc*zscore)
    return v_mcs, t_mcs, unc_mcs


from JDFTxFreeNrg.testing import anl_sphere_volume, anl_2sphere_union_volume, get_mc_volume_samples, get_mesh_volume_samples
# from JDFTxFreeNrg.solv_entropy import get_monte_carlo_spheres_volume, get_mesh_spheres_volume
import numpy as np
import matplotlib.pyplot as plt

def plot_volume_accuracy(
        vol_true: float, vol_mcs: list[float], vol_mesh: list[float], nsamples: list[float], time_mcs: list[float], 
        time_mesh: list[float], dev_mcs: list[float]):
    err_mcs = np.array([((v - vol_true) / vol_true) for v in vol_mcs])*100
    err_mesh = np.array([((v - vol_true) / vol_true) for v in vol_mesh])*100
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].axhline(y=0.0, color='green')
    ax[0].plot(time_mesh, err_mesh, marker='o', label='Mesh', color="red")
    ax[0].plot(time_mcs, err_mcs, marker='o', label='Monte Carlo', color="blue")
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel(r'Signed % error')
    ax[0].legend()
    ax[1].plot(time_mesh, vol_mesh, marker='o', label='Mesh', color="red")
    ax[1].scatter(time_mcs, vol_mcs, marker='o', label='Monte Carlo', color="blue")
    ax[1].errorbar(time_mcs, vol_mcs, yerr=dev_mcs, color="cyan")
    ax[1].plot(time_mcs, vol_mcs, color="blue", zorder=3)
    
    ax[1].axhline(y=vol_true, color='green', linestyle='--', label='Analytical')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Computed volume')
    ax[1].legend()
    # plt.show()
    return fig, ax

def test_single_sphere_volume(r: float | None = None, center: np.ndarray | None = None, nsampless: list[int] | None = None, mesh_sample_scale: float = 150.):
    if r is None:
        r = np.random.random() + 0.5
    v_anl = anl_sphere_volume(r)
    rs = [r]
    if center is None:
        center = np.random.rand(3) * 10.0
    centers = [center]
    if nsampless is None:
        nsampless = [1e3, 1e4, 1e5]
    v_meshs, t_meshs = get_mesh_volume_samples(rs, centers, np.array(nsampless)*mesh_sample_scale)
    v_mcs, t_mcs, dev_mcs = get_mc_volume_samples(rs, centers, nsampless)
    fig, ax = plot_volume_accuracy(v_anl, v_mcs, v_meshs, nsampless, t_mcs, t_meshs, dev_mcs)
    fig.suptitle('Single Sphere of r={:.2f} Volume Accuracy'.format(r))
    plt.show()


def test_double_sphere_volume(rs: list[float] | None = None, centers: list[np.ndarray] | None = None, nsampless: list[int] | None = None, mesh_sample_scale: float = 150.):
    if rs is None:
        rs = [np.random.random() + 0.5, np.random.random() + 0.5]
    if centers is None:
        centers = [np.zeros(3), np.ones(3)*0.5+np.random.random(3)*0.1]
    v_anl = anl_2sphere_union_volume(rs[0], rs[1], np.linalg.norm(centers[0]-centers[1]))
    # nsampless = [1e3, 1e4, 1e5, 2e5, 3e5]
    nsampless = [1e3, 1e4, 1e5, 1e6]
    v_meshs, t_meshs = get_mesh_volume_samples(rs, centers, np.array(nsampless)*mesh_sample_scale)
    v_mcs, t_mcs, dev_mcs = get_mc_volume_samples(rs, centers, nsampless)
    fig, ax = plot_volume_accuracy(v_anl, v_mcs, v_meshs, nsampless, t_mcs, t_meshs, dev_mcs)
    fig.suptitle('Double Sphere of r1={:.2f}, r2={:.2f}, d={:.2f} Volume Accuracy'.format(rs[0], rs[1], np.linalg.norm(centers[0]-centers[1])))
    plt.show()