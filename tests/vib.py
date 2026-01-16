import pytest
from JDFTxFreeNrg.qrrho import get_qrrho_vib_entropies, get_qrrho_vib_enthalpies
from JDFTxFreeNrg.standard import get_enthalpy_vib, get_entropy_vib, k_ev


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