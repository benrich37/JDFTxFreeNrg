import pytest
from JDFTxFreeNrg.qrrho import get_qrrho_vib_entropies, get_qrrho_vib_enthalpies
from JDFTxFreeNrg.standard import get_enthalpy_vib, get_entropy_vib, k_ev
from JDFTxFreeNrg.hessian import pert_along_vec, pert_along_vib_mode
import numpy as np
from pymatgen.core import Structure


@pytest.mark.parametrize(
        ("natoms", "disp"),
        [
            (5, 0.1),
            (10, 0.05),
        ]
)
def test_pert_along_vec(natoms: int, disp: float):
    coords = np.random.random((natoms, 3))
    struc = Structure(np.eye(3)*10., list(["H" for _ in range(natoms)]), coords, coords_are_cartesian=True)
    vec = np.random.random((natoms, 3))*5
    pert_struc = pert_along_vec(struc, vec, disp)
    _test_pert_along_vec(struc, pert_struc, vec, disp, natoms)


@pytest.mark.parametrize(
        ("disp"),
        [
            0.1,
            0.05,
        ]
)
def test_pert_along_vib_mode(disp: float):
    # Constructing test modes that are purely Cartesian displacements for ensured orthogonality for easy checking
    test_evecs = np.zeros((3,3,3))
    for i_cart in range(3):
        test_evecs[i_cart, :, i_cart] += 1.0
    struc = Structure(np.eye(3)*10., list(["H" for _ in range(3)]), np.random.random((3, 3)), coords_are_cartesian=True)
    for i_cart in range(3):
        vec = test_evecs[i_cart]
        pert_struc = pert_along_vib_mode(struc, test_evecs, i_cart, disp)
        _test_pert_along_vec(struc, pert_struc, vec, disp, 3)
        


def _test_pert_along_vec(struc: Structure, pert_struc: Structure, vec: np.ndarray, disp: float, natoms: int):
    coords = struc.cart_coords
    pert_coords = pert_struc.cart_coords
    dvec = pert_coords - coords
    dvec_len = np.linalg.norm(dvec)
    assert np.shape(pert_coords) == (natoms, 3)
    # Check magnitude is correct
    assert np.isclose(np.linalg.norm(dvec), disp), f"Expected displacement {disp}, got {dvec_len}"
    # Check displacement direction is correct
    dvec_normed = dvec / dvec_len
    vec_norm = vec / np.linalg.norm(vec)
    assert np.allclose(dvec_normed, vec_norm)



@pytest.mark.parametrize(
        ("freq", "T", "enthalpy_expected"),
        [
            (0.001, 300.0, k_ev * 300.0 / 2),
            (0.001, 150.0, k_ev * 150.0 / 2),
        ]
)
def test_expected_qrrho_enthalpy(freq: float, T: float, enthalpy_expected: float):
    enthalpy = get_qrrho_vib_enthalpies(freq, T=T)
    assert pytest.approx(enthalpy, rel=1e-5) == enthalpy_expected