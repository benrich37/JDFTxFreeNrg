from JDFTxFreeNrg.projection import gen_ortho_axes_R3, project_out_vector_from_vector
import numpy as np
import pytest


def _assert_orthogonal_unit_vectors(vtests: list[np.ndarray], vrefs: list[np.ndarray], atol: float = 1e-6):
    for i_test, vtest in enumerate(vtests):
        assert vtest.ndim == 1, f"Test vector {i_test} is not 1D (ndim {vtest.ndim})"
        assert np.isclose(np.linalg.norm(vtest), 1.0, atol=atol), f"Test vector {i_test} is not unit length (norm {np.linalg.norm(vtest)})"
        for i_ref, vref in enumerate(vrefs):
            dotprod = np.dot(vtest, vref)
            assert np.isclose(dotprod, 0.0, atol=atol), f"Test vector {i_test} is not orthogonal to reference vector {i_ref} (dot product {dotprod})"


@pytest.mark.parametrize(
        ("d"),
        [2, 3, 5, 10
        ]
)
def test_project_out_vector_from_vector(d: int):
    v1 = np.random.random((d,))
    v2 = np.random.random((d,))
    v2_proj = project_out_vector_from_vector(v2, v1)
    _assert_orthogonal_unit_vectors([v2_proj], [v1])
    assert np.isclose(np.dot(v2_proj, v1), 0.0), f"Projection failed to be orthogonal in dimension {d} (dot product {np.dot(v2_proj, v1)})"
    assert v2_proj.shape == (d,), f"Projected vector has incorrect shape in dimension {d}"
    assert np.isclose(1.0, np.linalg.norm(v2_proj)), f"Projected vector is not normalized in dimension {d}"


def test_gen_ortho_axes_R3():
    v1 = np.random.random((3,))
    v2, v3 = gen_ortho_axes_R3(v1)
    _assert_orthogonal_unit_vectors([v2, v3], [v1])
    _assert_orthogonal_unit_vectors([v2], [v3])