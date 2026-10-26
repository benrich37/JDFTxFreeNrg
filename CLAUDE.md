# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python library for free-energy analysis of (solvated, surface) JDFTx DFT calculations. Three main areas, each implementing equations from specific papers (see README.md for exact equation numbers):

- **Hessian projection / vibrational frequencies**: read the Hessian JDFTx dumps from a vibrational calculation, project out chosen translations/rotations of sub-molecules to remove spurious imaginary modes, and diagonalize.
- **Vibrational thermochemistry**: QRRHO entropies (Grimme 2012) and enthalpies (Li & Head-Gordon 2014), plus the ordinary harmonic-oscillator / rigid-rotor / ideal-gas terms.
- **Solvation entropies** (Garza 2019): effective cavity volume for translational entropy, the restricted-rotor correction, and Van der Waals volumes from Monte Carlo integration over atomic spheres.

## Environment and commands

- Install for development: `pip install -e .` (Python ≥3.10). `pyproject.toml` has no `[build-system]` and no `__init__.py` in `src/JDFTxFreeNrg/`, so the package is imported as a namespace package (`from JDFTxFreeNrg.hessian import ...`).
- **pymatgen is not a declared dependency** (it is commented out in `pyproject.toml` on purpose, to speed up reinstalls). The code needs `pymatgen.io.jdftx` (`JDFTXInfile`, `JDFTXOutfile`), which comes from the author's fork `pymatgen@git+https://github.com/benrich37/pymatgen.git` or a recent upstream pymatgen. scipy and matplotlib are also imported but not declared.
- Optional extras: `pyvolgrid` (an alternative volume integrator, `volume.get_pyvol_spheres_volume`) and `qrotor` (only used by `qrotor.py`).
- Tests: the files in `tests/` are **not** named `test_*.py`, so a plain `pytest` collects nothing. Pass them by path:
  - `python -m pytest tests/projection.py tests/vib.py`
  - Single test: `python -m pytest tests/vib.py::test_pert_along_vec`
- No linter or formatter is configured (the ruff/pyright sections in `pyproject.toml` are commented out).
- Most usage and benchmarking lives in Jupyter notebooks (`examples/`, `benchmarking/`). The example JDFTx calculation directories are in `examples/data/`. `benchmarking/data/` and all `*.json` files are gitignored.

## Architecture

### JDFTx calculation directory convention
Functions that take a `calc_dir` expect JDFTx's file layout: `in` (input, parsed by `JDFTXInfile`, the default source of the structure since `use_in=True`), `out` (parsed by `JDFTXOutfile`), and `K` or `<prefix>.K` (the Hessian as raw complex128 binary). `hessian.read_K` reshapes `K` to `(3*nFree, 3*nFree)`, where nFree counts only atoms whose `selective_dynamics` flags allow movement.

### Hessian pipeline (`hessian.py` → `projection.py` → `_orthog.py`)
1. `read_K(..., expand_to_full=True)` puts the free-atom Hessian into the full `3N×3N` space, with zero rows and columns for fixed atoms. **Projection always runs in the full space.**
2. `get_projected_K` builds a projector from `molecule_sets` (`projection.get_projector`) and applies `(I − PP†) K (I − PP†)ᵀ`, or `PP† K PP†ᵀ` with `reverse=True`.
3. `reduce_K` and `get_reduced_structure` go back to free atoms only. The result is mass-weighted (`get_omegaSq`) and diagonalized (`solve_vib_modes`), and eigenvalues (Hartree) become cm⁻¹ through `get_freqs` and `freq_nrg_to_cm`. Imaginary frequencies come out as complex values with a nonzero imaginary part.

`_orthog.py` removes near-parallel projector vectors (`remove_parallel_vectors_loop`, cutoff 1e-4) so that overlapping translation/rotation sets don't make the projector rank-deficient.

### `molecule_sets`: the central user-facing data structure
A list of dicts passed through `hessian`, `combo`, and `projection`. Each dict describes a group of atoms whose rigid motions get projected out (for the Hessian) and counted as trans/rot free-energy terms (in `combo`):
- `indices` (or `idcs`): list of ints or `"all"`
- `trans` / `rot`: bool; at least one must be True
- `axes`: optional list. Each entry is `"x"/"y"/"z"`, a 2-int list of atom indices (the axis between them), a length-3 array, or `{"axis": ..., "ortho": bool}`. With `ortho=True` that entry expands to the two axes orthogonal to the given one. `None` means all three directions (inertia-tensor axes for rotations). The number of axes sets the dimensionality `d` of the trans/rot terms.
- `center`: optional rotation center (atom index, list of indices, or a 3-vector)

`resolve_idcss` and `resolve_axes` **mutate the dicts in place**, replacing user specs with resolved index lists and unit vectors, so a `molecule_sets` list should not be reused across calls that expect raw input (see the open bug under Known state). `combo.py` has its own `resolve_axis`/`resolve_axes`, which wrap the result differently (`append` rather than `+=`) than the ones in `projection.py`.

### Free-energy assembly (`combo.py`)
`get_free_energy(solute_calc_dir, vib_calc_dir, T, solvent_calc_dir=None, ...)` is the top-level entry point. It returns A (or G with `mu`) in eV. It:
- takes electronic `F` from the last slice of `solute_calc_dir/out`,
- prepends whole-structure trans and rot sets to `molecule_sets` when `free_trans`/`free_rot` are set,
- sums vibrational (`qrrho.py`, or plain HO when `qrrho=False`), translational, and rotational contributions. With a `solvent_calc_dir`, trans/rot use the Garza solvation forms in `solv_entropy.py`, which need `StructureVolume` volumes. Without one, they use ideal-gas forms from `standard.py`.

### Units
Energies and free energies are in **eV**, and entropies in eV/K (`standard.k_ev`). Frequencies are in cm⁻¹ at the API boundary and Hartree internally. Temperatures are in K, and `T` defaults to 300 in most functions. Volumes are in Å³, from pymatgen Cartesian coordinates.

### Volumes (`volume.py`)
`StructureVolume` subclasses pymatgen `Structure` and caches computed VdW volumes as JSON under a `vdw_radii_cache/` directory next to the calculation, keyed by atom indices and sample count/grid spacing. The method is `"MC"` (Monte Carlo, the default) or grid-based. `testing.py` holds analytic sphere volume/area references and benchmarking helpers (it is a library module, not a test file; its `test_*` functions are benchmarks that use matplotlib, and some download FreeSolv data).

### Other modules
- `glogwrite.py`: writes Gaussian-format `.log` files so modes can be visualized in standard viewers (`hessian.write_Gaussian_vib_log_from_calc_dir`).
- `freq_fitting.py`: fits frequencies and rotational barriers from JDFTx scan trajectories.
- `hindered_rotor.py`, `qrotor.py`: hindered-rotor thermochemistry (analytic, and numerical through the external `qrotor` package).
- `area.py`, `_common.py`: surface areas and small linear-algebra helpers.

## Known state

- As of this writing, `tests/projection.py::test_resolve_axis_pair_indices` fails because it expects the un-normalized `p1 - p0`, while `resolve_axis` returns unit vectors.
- `combo.get_free_energy_vib` passes `proj_rot=` and `proj_trans=` to `hessian.get_freqs_cm_from_calc_dir`, which does not accept those keywords.
- **Open bug: the public frequency functions mutate the caller's `molecule_sets`.** `get_freqs_cm_from_calc_dir` → `get_omegaSqEigs_from_calc_dir` → `get_projected_K` → `get_projector` → `get_projector_raw` → `_get_projector_raw`, which calls `resolve_idcss` and `resolve_axes` on the caller's list with no copy anywhere on that path. After one call, every `"indices": "all"` is overwritten with `list(range(n_atoms))` *of that first structure*. If the same list object is then used for a structure with a different atom count, the projected-out rigid motions cover the wrong atoms. The frequencies, and the vib/trans/rot terms built from them, come out silently wrong; nothing errors.
  - Observed 2026-10-26 in `research/Patty/anl/plot_cur_nrgs.py`: one module-level `molecule_sets` was used for H2 and then reused for NH3, CH4 and CO2. NH3's qRRHO vib term came out 0.866 eV instead of the correct 0.913 eV.
  - Shallow copies don't protect callers. `research.projects.no3_ads_gen.energy_caching(_v2).collect_data_free_species` does `mol_sets.copy()` before its own `resolve_idcss` call, but the dicts inside are the same objects.
  - Atom-pair axes (`[i, j]` entries in `axes`) have the same problem as `"all"`: they become the unit vector between those atoms *in the first structure*, and that fixed vector is reused for later structures. Explicit integer `indices`, named axes (`"x"/"y"/"z"`, via `ref_axs`) and explicit 3-vectors resolve the same way for any structure, so reusing them gives correct results, though they are still replaced in the caller's dicts.
  - Intended fix (not applied yet): `molecule_sets = copy.deepcopy(molecule_sets)` at the top of `projection.get_projector_raw`, the single point every projection caller goes through. In the `research` repo, change both `collect_data_free_species` copies to `copy.deepcopy`. Until then, callers should build a fresh `molecule_sets` for each structure, as `research/NO3_ads_gen/anl/mnc/cache_all_molecules.py` does.
