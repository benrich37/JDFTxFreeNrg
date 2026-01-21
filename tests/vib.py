import pytest
from JDFTxFreeNrg.qrrho import get_qrrho_vib_entropies, get_qrrho_vib_enthalpies
from JDFTxFreeNrg.standard import get_enthalpy_vib, get_entropy_vib, k_ev
from JDFTxFreeNrg.hessian import pert_along_vec
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
    _test_pert_along_vec(struc, vec, disp, natoms)

def _test_pert_along_vec(struc: Structure, vec: np.ndarray, disp: float, natoms: int):
    coords = struc.cart_coords
    pert_struc = pert_along_vec(struc, vec, disp)
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