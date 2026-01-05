from JDFTxFreeNrg.testing import anl_sphere_volume, anl_2sphere_union_volume, get_mc_volume_samples, get_mesh_volume_samples, anl_3sphere_union_volume
# from JDFTxFreeNrg.solv_entropy import get_monte_carlo_spheres_volume, get_mesh_spheres_volume
import numpy as np
import matplotlib.pyplot as plt

# def plot_volume_accuracy(
#         vol_true: float, vol_mcs: list[float], vol_mesh: list[float], nsamples: list[float], time_mcs: list[float], 
#         time_mesh: list[float], dev_mcs: list[float]):
#     err_mcs = np.array([((v - vol_true) / vol_true) for v in vol_mcs])*100
#     err_mesh = np.array([((v - vol_true) / vol_true) for v in vol_mesh])*100
#     fig, ax = plt.subplots(nrows=2, sharex=True)
#     ax[0].axhline(y=0.0, color='green')
#     ax[0].plot(time_mesh, err_mesh, marker='o', label='Mesh', color="red")
#     ax[0].plot(time_mcs, err_mcs, marker='o', label='Monte Carlo', color="blue")
#     ax[0].set_xlabel('Time (s)')
#     ax[0].set_ylabel(r'Signed % error')
#     ax[0].legend()
#     ax[1].plot(time_mesh, vol_mesh, marker='o', label='Mesh', color="red")
#     ax[1].scatter(time_mcs, vol_mcs, marker='o', label='Monte Carlo', color="blue")
#     ax[1].errorbar(time_mcs, vol_mcs, yerr=dev_mcs, color="cyan")
#     ax[1].plot(time_mcs, vol_mcs, color="blue", zorder=3)
    
#     ax[1].axhline(y=vol_true, color='green', linestyle='--', label='Analytical')
#     ax[1].set_xlabel('Time (s)')
#     ax[1].set_ylabel('Computed volume')
#     ax[1].legend()
#     # plt.show()
#     return fig, ax


def plot_volume_accuracy(
        vol_true: float, 
        labels: list[str],
        vol_numerics: dict[str, list[float]],
        time_numerics: dict[str, list[float]],
        dev_numerics: dict[str, list[float]] = None,
        colors: dict[str, str] = None
        ):
    if colors is None:
        colors = {label: color for label, color in zip(labels, ['red', 'blue', 'green', 'orange', 'purple', 'cyan'])}
    if dev_numerics is None:
        dev_numerics = {}
    err_numerics = {label: np.array([((v - vol_true) / vol_true) for v in vol_numerics[label]])*100 for label in labels}
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].axhline(y=0.0, color='green')
    for label in labels:
        ax[0].plot(time_numerics[label], err_numerics[label], marker='o', label=label, color=colors[label])
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel(r'Signed % error')
    ax[0].legend()
    for label in labels:
        if label in dev_numerics:
            ax[1].scatter(time_numerics[label], vol_numerics[label], marker='o', label=label, color=colors[label])
            ax[1].errorbar(time_numerics[label], vol_numerics[label], yerr=dev_numerics[label], color=colors[label], fmt='o', label=label)
            ax[1].plot(time_numerics[label], vol_numerics[label], color=colors[label], zorder=3)
        else:
            ax[1].plot(time_numerics[label], vol_numerics[label], marker='o', label=label, color=colors[label])
    ax[1].axhline(y=vol_true, color='green', linestyle='--', label='Analytical')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Computed volume')
    ax[1].legend()
    # plt.show()
    return fig, ax

# def test_single_sphere_volume():
#     r = np.random.random() + 0.5
#     v_anl = anl_sphere_volume(r)
#     rs = [r]
#     centers = [np.random.rand(3) * 10.0]
#     # nsampless = [1e3, 1e4, 1e5, 2e5, 3e5]
#     nsampless = [1e3, 1e4, 1e5]
#     # nsampless = [1e2, 2e2, 3e2, 4e2, 1e3, 3e3, 6e3, 1e4]
#     # nsampless = list(np.arange(1e4, 1e5, 2e4))
#     v_meshs, t_meshs = get_mesh_volume_samples(rs, centers, np.array(nsampless)*200)
#     v_mcs, t_mcs, dev_mcs = get_mc_volume_samples(rs, centers, nsampless)
#     fig, ax = plot_volume_accuracy(v_anl, v_mcs, v_meshs, nsampless, t_mcs, t_meshs, dev_mcs)
#     fig.suptitle('Single Sphere of r={:.2f} Volume Accuracy'.format(r))
#     plt.show()


# def test_double_sphere_volume():
#     rs = [np.random.random() + 0.5, np.random.random() + 0.5]
#     centers = [np.zeros(3), np.ones(3)*0.5+np.random.random(3)*0.1]
#     v_anl = anl_2sphere_union_volume(rs[0], rs[1], np.linalg.norm(centers[0]-centers[1]))
#     # nsampless = [1e3, 1e4, 1e5, 2e5, 3e5]
#     nsampless = [1e3, 1e4, 1e5, 1e6]
#     v_meshs, t_meshs = get_mesh_volume_samples(rs, centers, np.array(nsampless)*200)
#     v_mcs, t_mcs, dev_mcs = get_mc_volume_samples(rs, centers, nsampless)
#     fig, ax = plot_volume_accuracy(v_anl, v_mcs, v_meshs, nsampless, t_mcs, t_meshs, dev_mcs)
#     fig.suptitle('Double Sphere of r1={:.2f}, r2={:.2f}, d={:.2f} Volume Accuracy'.format(rs[0], rs[1], np.linalg.norm(centers[0]-centers[1])))
#     plt.show()

# def test_triple_sphere_volume():
#     rs = [np.random.random()*0.1 + 0.5, np.random.random()*0.1 + 0.5, np.random.random()*0.1 + 0.5]
#     centers = [np.zeros(3), np.array([1.,0.,0.]), np.array([0.5, np.sqrt(1-(0.5**2)), 0.0])]
#     v_anl = anl_3sphere_union_volume(
#         rs[0], rs[1], rs[2],
#         centers[0],
#         centers[1],
#         centers[2],
#     )
#     nsampless = [1e3, 1e4, 1e5, 2e5, 3e5]
#     v_meshs, t_meshs = get_mesh_volume_samples(rs, centers, np.array(nsampless)*200)
#     v_mcs, t_mcs, dev_mcs = get_mc_volume_samples(rs, centers, nsampless)
#     fig, ax = plot_volume_accuracy(v_anl, v_mcs, v_meshs, nsampless, t_mcs, t_meshs, dev_mcs)
#     fig.suptitle('Triple Sphere of r1={:.2f}, r2={:.2f}, r3={:.2f} Volume Accuracy'.format(rs[0], rs[1], rs[2]))
#     plt.show()




# test_single_sphere_volume()
# test_double_sphere_volume()
# test_triple_sphere_volume()


