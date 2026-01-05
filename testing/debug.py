from JDFTxFreeNrg.testing import *

def test_triple_sphere_volume(rs: list[float] | None = None, centers: list[np.ndarray] | None = None, nsampless: list[int] | None = None, mesh_sample_scale: float = 200.):
    if rs is None:
        rs = [np.random.random()*0.1 + 0.5, np.random.random()*0.1 + 0.5, np.random.random()*0.1 + 0.5]
    if centers is None:
        centers = [np.zeros(3), np.array([1.,0.,0.]), np.array([0.5, np.sqrt(1-(0.5**2)), 0.0])]
    v_anl = anl_3sphere_union_volume(
        rs[0], rs[1], rs[2],
        centers[0],
        centers[1],
        centers[2],
    )
    if nsampless is None:
        nsampless = [1e3, 1e4, 1e5, 2e5]
    v_meshs, t_meshs = get_mesh_volume_samples(rs, centers, np.array(nsampless)*mesh_sample_scale)
    v_mcs, t_mcs, dev_mcs = get_mc_volume_samples(rs, centers, nsampless)
    fig, ax = plot_volume_accuracy(v_anl, v_mcs, v_meshs, nsampless, t_mcs, t_meshs, dev_mcs)
    fig.suptitle('Triple Sphere of r1={:.2f}, r2={:.2f}, r3={:.2f} Volume Accuracy'.format(rs[0], rs[1], rs[2]))
    plt.show()