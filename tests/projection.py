from JDFTxFreeNrg.projection import gen_ortho_axes_R3, project_out_vector_from_vector, resolve_axis, ref_axs
from JDFTxFreeNrg.testing import gen_random_structure
from typing import Any
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


@pytest.mark.parametrize(
        ("axis", "natoms"),
        [
            ("x", 5),
            ("y", 10),
            ("z", 3),
            ([0, 1], 4),
            (np.array([1.0, 0.0, 0.0]), 6),
            ({"axis": "x", "ortho": True}, 7),
            ({"axis": [2, 3], "ortho": False}, 8),
        ]
)
def test_resolve_axis_data_type(axis: list[int] | str | np.ndarray | dict, natoms: int):
    structure = gen_random_structure(natoms)
    axes = resolve_axis(structure, axis)
    for i, ax in enumerate(axes):
        assert ax.ndim == 1, f"Resolved axis {i} is not 1D (ndim {ax.ndim})"
        assert isinstance(ax, np.ndarray), f"Resolved axis {i} is not a numpy array (type {type(ax)})"


@pytest.mark.parametrize(
        ("natoms"),
        [
            5,
            10,
            3,
        ]
)
def test_resolve_axis_named_axes(natoms: int):
    structure = gen_random_structure(natoms)
    for axis_name in ref_axs.keys():
        axis = ref_axs[axis_name]
        axis_out = resolve_axis(structure, axis_name)
        assert len(axis_out) == 1, f"Expected single axis output for named axis {axis_name}, got {len(axis_out)}"
        ax = axis_out[0]
        assert np.allclose(ax, axis), f"Resolved axis for named axis {axis_name} does not match reference axis."


@pytest.mark.parametrize(
        ("natoms"),
        [
            5,
            10,
            3,
        ]
)
def test_resolve_axis_pair_indices(natoms: int):
    structure = gen_random_structure(natoms)
    for i1 in range(natoms):
        for _i2 in range(natoms-1):
            i2 = (_i2 + i1 + 1) % natoms
            axis_indices = [i1, i2]
            p0 = structure.cart_coords[axis_indices[0]]
            p1 = structure.cart_coords[axis_indices[1]]
            expected_axis = p1 - p0
            axis_out = resolve_axis(structure, axis_indices)
            assert len(axis_out) == 1, f"Expected single axis output for axis indices {axis_indices}, got {len(axis_out)}"
            ax = axis_out[0]
            assert np.allclose(ax, expected_axis), f"Resolved axis for indices {axis_indices} does not match expected axis."

@pytest.mark.parametrize(
        ("axis", "natoms"),
        [
            ("x", 5),
            ("y", 10),
            ("z", 3),
            ([0, 1], 4),
            (np.array([1.0, 0.0, 0.0]), 6),
        ]
)
def test_resolve_axis_generate_ortho_set(axis: list[int] | str | np.ndarray | dict, natoms: int):
    structure = gen_random_structure(natoms)
    reg_axis = resolve_axis(structure, axis)[0]
    orthog_axes = resolve_axis(structure, {"axis": axis, "ortho": True})
    assert len(orthog_axes) == 2, f"Expected 2 orthogonal axes generated from axis {axis}, got {len(orthog_axes)}"
    _assert_orthogonal_unit_vectors(orthog_axes, [reg_axis])
    redundant_axes = resolve_axis(structure, {"axis": axis, "ortho": False})
    assert len(redundant_axes) == 1, f"Expected 1 axis generated from axis {axis} with ortho=False, got {len(redundant_axes)}"
    assert np.allclose(redundant_axes[0], reg_axis), f"Axis generated from axis {axis} with ortho=False does not match regular resolved axis."


@pytest.mark.parametrize(
        ("axis", "exception"),
        [
            ({"ortho": True}, ValueError),
            (np.ones(5), ValueError),
            ("invalid_axis_name", ValueError),
            ({"axis": 5, "ortho": True}, TypeError),
            (["a", "b"], TypeError),
            (["x", 1], TypeError),
        ]
)
def test_resolve_axis_invalid_inputs(axis: Any, exception: Exception):
    structure = gen_random_structure(5)
    with pytest.raises(exception):
        resolve_axis(structure, axis)