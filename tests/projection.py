from JDFTxFreeNrg.projection import gen_ortho_axes_R3, project_out_vector_from_vector, resolve_axis, ref_axs, resolve_axes, get_projector, get_rotation_vector, get_center_of_mass, resolve_center
from JDFTxFreeNrg.testing import gen_random_structure
from typing import Any
import copy
from pymatgen.core.structure import Structure
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

def test_resolve_axes():
    structure = gen_random_structure(5)
    molecule_sets = [
        {
            "name": "molecule1",
            "axes": [
                "x",
                [0, 1],
                np.array([0.0, 1.0, 0.0]),
                {"axis": "y", "ortho": True},
                {"axis": [2, 3], "ortho": False},
            ]
        }
    ]
    resolve_axes(structure, molecule_sets)
    axes = molecule_sets[0]["axes"]
    assert len(axes) == 6, f"Expected 6 resolved axes in molecule set, got {len(axes)}"
    for i, ax in enumerate(axes):
        assert ax.ndim == 1, f"Resolved axis {i} is not 1D (ndim {ax.ndim})"
        assert isinstance(ax, np.ndarray), f"Resolved axis {i} is not a numpy array (type {type(ax)})"

def test_get_projector_does_not_mutate_molecule_sets():
    molecule_sets = [
        {"indices": "all", "trans": True, "rot": False},
        {"indices": [0, 1, 2], "trans": False, "rot": True, "axes": [[0, 1], "z", {"axis": "x", "ortho": True}]},
    ]
    original = copy.deepcopy(molecule_sets)
    small = gen_random_structure(4)
    large = gen_random_structure(7)
    get_projector(small, molecule_sets=molecule_sets)
    assert molecule_sets == original, "get_projector modified the caller's molecule_sets"
    # Reusing the same list on a structure with a different atom count must match a fresh list
    P_reused = get_projector(large, molecule_sets=molecule_sets)
    P_fresh = get_projector(large, molecule_sets=copy.deepcopy(original))
    assert np.allclose(P_reused @ P_reused.T, P_fresh @ P_fresh.T), "Reused molecule_sets gave a different projector"


def _rigid_rotation_residuals(structure, idcs: list[int], P: np.ndarray) -> list[float]:
    # Norm of each true rigid rotation of atoms idcs (about their COM) left outside the projector P
    sub = Structure.from_sites([structure[i] for i in sorted(set(idcs))])
    com = get_center_of_mass(sub)
    residuals = []
    for ax in np.eye(3):
        d = np.zeros((len(structure), 3))
        for i in set(idcs):
            d[i] = np.cross(ax, structure.cart_coords[i] - com)
        d = d.flatten() / np.linalg.norm(d)
        residuals.append(np.linalg.norm(d - P @ (P.T @ d)))
    return residuals


@pytest.mark.parametrize(
        ("idcs"),
        [
            [1, 3, 4],
            [4, 1, 3],
            [3, 4, 1],
            [5, 0, 2, 4],
            [4, 1, 3, 1],
        ]
)
def test_rotation_projector_index_order(idcs: list[int]):
    structure = gen_random_structure(6)
    P = get_projector(structure, molecule_sets=[{"indices": idcs, "rot": True}])
    P_sorted = get_projector(structure, molecule_sets=[{"indices": sorted(set(idcs)), "rot": True}])
    assert np.allclose(P @ P.T, P_sorted @ P_sorted.T), f"Rotation projector depends on index order for {idcs}"
    residuals = _rigid_rotation_residuals(structure, idcs, P)
    assert np.allclose(residuals, 0.0, atol=1e-8), f"Rigid rotations of {idcs} not captured by projector (residuals {residuals})"


def test_get_rotation_vector_subset():
    structure = gen_random_structure(6)
    idcs = [4, 1, 3]
    P = get_projector(structure, molecule_sets=[{"indices": idcs, "rot": True, "axes": ["x", "y", "z"]}])
    for ax in ref_axs.values():
        vec = get_rotation_vector(ax, structure, mol_indices=idcs)
        vec = vec / np.linalg.norm(vec)
        assert np.isclose(np.linalg.norm(vec - P @ (P.T @ vec)), 0.0, atol=1e-8), "get_rotation_vector is not a rigid rotation of the subset"
        assert np.allclose(vec.reshape(-1, 3)[[0, 2, 5]], 0.0), "get_rotation_vector moved atoms outside the subset"


def _rotation_residuals_about_point(structure, idcs: list[int], point: np.ndarray, P: np.ndarray) -> list[float]:
    # Norm of each rigid rotation of atoms idcs about point left outside the projector P
    residuals = []
    for ax in np.eye(3):
        d = np.zeros((len(structure), 3))
        for i in set(idcs):
            d[i] = np.cross(ax, structure.cart_coords[i] - point)
        d = d.flatten() / np.linalg.norm(d)
        residuals.append(np.linalg.norm(d - P @ (P.T @ d)))
    return residuals


@pytest.mark.parametrize(
        ("center", "expected_idcs"),
        [
            (3, [3]),
            (np.int64(4), [4]),
            ([1, 4], [1, 4]),
            ([0, 5], [0, 5]),  # atoms outside the rotating set are allowed as a center
        ]
)
def test_rotation_projector_center_indices(center, expected_idcs: list[int]):
    structure = gen_random_structure(6)
    idcs = [1, 3, 4]
    point = np.mean(structure.cart_coords[expected_idcs], axis=0)
    P = get_projector(structure, molecule_sets=[{"indices": idcs, "rot": True, "center": center, "axes": ["x", "y", "z"]}])
    P_coords = get_projector(structure, molecule_sets=[{"indices": idcs, "rot": True, "center": point, "axes": ["x", "y", "z"]}])
    assert np.allclose(P @ P.T, P_coords @ P_coords.T), f"Center {center} does not match explicit coordinates {point}"
    residuals = _rotation_residuals_about_point(structure, idcs, point, P)
    assert np.allclose(residuals, 0.0, atol=1e-8), f"Rotations about center {center} not captured (residuals {residuals})"


@pytest.mark.parametrize(
        ("center", "expected"),
        [
            (None, None),
            ([], None),
            ([0.0, 0.0, 1.5], np.array([0.0, 0.0, 1.5])),
            ([0, 0, 1.5], np.array([0.0, 0.0, 1.5])),
            ((1.0, 2.0, 3.0), np.array([1.0, 2.0, 3.0])),
            (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])),
        ]
)
def test_resolve_center_coordinates(center, expected):
    structure = gen_random_structure(4)
    resolved = resolve_center(structure, center)
    if expected is None:
        assert resolved is None
    else:
        assert np.allclose(resolved, expected)


@pytest.mark.parametrize(
        ("center", "exception"),
        [
            ("com", TypeError),
            (1.5, TypeError),
            ([1.0, 2.0], ValueError),
            (np.zeros((3, 1)), ValueError),
        ]
)
def test_resolve_center_invalid(center, exception):
    structure = gen_random_structure(4)
    with pytest.raises(exception):
        resolve_center(structure, center)
