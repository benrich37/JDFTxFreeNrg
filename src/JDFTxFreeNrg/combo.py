
from pathlib import Path
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from JDFTxFreeNrg.hessian import get_freqs_cm_from_calc_dir, print_freqs
from JDFTxFreeNrg.solv_entropy import get_vfree, get_solv_entropy_rot, get_solv_entropy_trans, StructureVolume
from JDFTxFreeNrg.standard import get_enthalpy_rot, get_enthalpy_trans, get_entropy_rot, get_ideal_gas_vol, get_entropy_trans, k_ev
from JDFTxFreeNrg.qrrho import get_qrrho_vib_enthalpies, get_qrrho_vib_entropies


def clean_structure(structure: Structure) -> Structure:
    structure.remove_site_property("group_names")
    structure.remove_site_property("velocities")
    structure.remove_site_property("constraint_vectors")
    structure.remove_site_property("constraint_types")

ref_axs = {
    "x": np.array([1,0,0], dtype=np.float64),
    "y": np.array([0,1,0], dtype=np.float64),
    "z": np.array([0,0,1], dtype=np.float64)
}

def resolve_axis(structure: Structure, axis: str | list[int] | np.ndarray) -> np.ndarray:
    if isinstance(axis, np.ndarray):
        return axis
    elif isinstance(axis, str):
        return ref_axs[axis]
    elif isinstance(axis, list):
        p0 = structure.cart_coords[axis[0]]
        p1 = structure.cart_coords[axis[1]]
        return p1 - p0
    
def resolve_axes(structure: Structure, molecule_sets: list[dict]) -> None:
    for mset in molecule_sets:
        if "axes" in mset:
            axes = mset["axes"]
            resolved_axes = []
            for axis in axes:
                resolved_axes.append(resolve_axis(structure, axis))
            mset["axes"] = resolved_axes
        

def get_free_energy_vib(vib_calc_dir: Path, T: float, qrrho: bool = True, freq0: float = 100., alpha: float = 4.0, molecule_sets: list[dict] | None = None, bad_proj_method_len: int = 0, no_proj: bool = False, structure: Structure | None = None, verbose: bool = False,
                        integration_method: str = "MC", integration_kwargs: dict | None = None) -> tuple[float, float, float]:
    if molecule_sets is None:
        molecule_sets = []
    if structure is None:
        structure = StructureVolume.from_calc_dir(vib_calc_dir)
    resolve_axes(structure, molecule_sets)
    freqs = get_freqs_cm_from_calc_dir(vib_calc_dir, molecule_sets=None if no_proj else molecule_sets, proj_rot=False, proj_trans=False, trim_zero = (bad_proj_method_len == 0))[bad_proj_method_len:].real
    freqs = np.array([f for f in freqs if f > 1e-6])
    if verbose:
        print_freqs(freqs, zero_thresh=None)
    if qrrho:
        E_v = np.nansum(get_qrrho_vib_enthalpies(freqs, T, freq0=freq0, alpha=alpha)).real
        S_v = np.nansum(get_qrrho_vib_entropies(freqs, T, freq0=freq0, alpha=alpha)).real
    else:
        E_v = np.nansum(get_qrrho_vib_enthalpies(freqs, T, freq0=0, alpha=alpha)).real
        S_v = np.nansum(get_qrrho_vib_entropies(freqs, T, freq0=0, alpha=alpha)).real
    return E_v, S_v

def get_free_energy_tr(solute_calc_dir: Path, T: float, solvent_calc_dir: Path | None = None, solvent_molarity: float = 55.5, P_ref: float = 1., molecule_sets: list[dict] | None = None, verbose: bool = False,
                       integration_method: str = "MC", integration_kwargs: dict | None = None
                       ) -> tuple[float, float]:
    if integration_kwargs is None:
        integration_kwargs = {}
    if molecule_sets is None:
        molecule_sets = []
    structure = StructureVolume.from_calc_dir(solute_calc_dir)
    clean_structure(structure)
    E_t = 0.0
    S_t = 0.0
    vs = None
    vf = None
    if solvent_calc_dir is not None:
        solv_structure = StructureVolume.from_calc_dir(solvent_calc_dir)
        vs = solv_structure.get_volume(**integration_kwargs)
        vf = get_vfree(vs, solvent_molarity)
    for mset in molecule_sets:
        idcs = mset["indices"]
        substructure = Structure.from_sites([structure.sites[i] for i in idcs])
        if mset["trans"]:
            axes = mset.get("axes", None)
            d = 3 if axes is None else len(axes)
            E_t += get_enthalpy_trans(T, d=d)
            if solvent_calc_dir is not None:
                vm = structure.get_volume(idcs=idcs, **integration_kwargs)
                S_t += get_solv_entropy_trans(substructure, vm, vs, vf, T, d=d)
            else:
                vol = get_ideal_gas_vol(P_ref, T)
                mass = sum([site.specie.atomic_mass for site in substructure])
                S_t += get_entropy_trans(mass, T, vol, d=d)
    ntrans = sum([1 for mset in molecule_sets if mset.get("trans", False)])
    S_t -= (ntrans - 1) * k_ev
    return E_t, S_t

def get_free_energy_rot(solute_calc_dir: Path, T: float, solvent_calc_dir: Path | None = None, solvent_molarity: float = 55.5, P_ref: float = 1., molecule_sets: list[dict] | None = None, verbose: bool = False,
                        integration_method: str = "MC", integration_kwargs: dict | None = None
                        ) -> tuple[float, float]:
    if integration_kwargs is None:
        integration_kwargs = {}
    if molecule_sets is None:
        molecule_sets = []
    structure = StructureVolume.from_calc_dir(solute_calc_dir, method=integration_method)
    clean_structure(structure)
    E_r = 0.0
    S_r = 0.0
    vs = None
    vf = None
    if solvent_calc_dir is not None:
        solv_structure = StructureVolume.from_calc_dir(solvent_calc_dir, method=integration_method)
        vs = solv_structure.get_volume(**integration_kwargs)
        vf = get_vfree(vs, solvent_molarity)
    for mset in molecule_sets:
        idcs = mset["indices"]
        substructure = Structure.from_sites([structure.sites[i] for i in idcs])
        if mset["rot"]:
            axes = mset.get("axes", None)
            d = 3 if axes is None else len(axes)
            E_r += get_enthalpy_rot(substructure, T, d=d)
            # Don't yet have a partition function for a 1D rigid rotor
            if d == 3:
                if solvent_calc_dir is not None:
                    vm = structure.get_volume(idcs=idcs, **integration_kwargs)
                    S_r += get_solv_entropy_rot(substructure, vm, vf, T)
                else:
                    S_r += get_entropy_rot(substructure, T)
            else:
                if verbose:
                    print(f"Skipping rigid rotor entropy for d < 3 ({mset})")
    return E_r, S_r




def get_free_energy(
        solute_calc_dir: Path, vib_calc_dir: Path, T: float, solvent_calc_dir: Path | None = None, solvent_molarity: float = 55.5, P_ref: float = 1., M_ref: float = 1,
        qrrho: bool = True, freq0: float = 100., alpha: float = 4.0, apply_ssc: bool = True, verbose: bool = False, molecule_sets: list[dict] | None = None,
        free_trans: bool = True, free_rot: bool = True, mu: float | bool | None = None, skip_elec: bool = False, bad_proj_method_len: int = 0, no_proj: bool = False,
        integration_method: str = "MC", integration_kwargs: dict | None = None
        ) -> float:
    if molecule_sets is None:
        molecule_sets = []
    if skip_elec:
        structure = StructureVolume.from_calc_dir(solute_calc_dir, method=integration_method)
        A_e = 0.0
    else:
        structure = StructureVolume.from_calc_dir(solute_calc_dir, method=integration_method)
        outfile = JDFTXOutfile.from_file(solute_calc_dir / "out")
        A_e = outfile.slices[-1].ecomponents["F"]
    clean_structure(structure)
    if free_trans:
        molecule_sets = [{"indices": list(range(len(structure.sites))), "trans": True, "rot": False}] + molecule_sets
    if free_rot:
        molecule_sets = [{"indices": list(range(len(structure.sites))), "trans": False, "rot": True}] + molecule_sets
    E_v, S_v = get_free_energy_vib(
        vib_calc_dir, T, qrrho=qrrho, freq0=freq0, alpha=alpha, molecule_sets=molecule_sets, bad_proj_method_len=bad_proj_method_len, no_proj=no_proj, structure=structure,
    )
    E_t, S_t = get_free_energy_tr(
        solute_calc_dir, T, solvent_calc_dir=solvent_calc_dir, solvent_molarity=solvent_molarity, P_ref=P_ref, molecule_sets=molecule_sets, verbose=verbose,
        integration_method=integration_method, integration_kwargs=integration_kwargs
    )
    E_r, S_r = get_free_energy_rot(
        solute_calc_dir, T, solvent_calc_dir=solvent_calc_dir, solvent_molarity=solvent_molarity, P_ref=P_ref, molecule_sets=molecule_sets, verbose=verbose,
        integration_method=integration_method, integration_kwargs=integration_kwargs
    )
    A_v = E_v - T * S_v
    total = A_e + A_v + E_t + E_r - T * (S_t + S_r)
    if verbose:
        print(f"A_e: {A_e:.4f} eV")
        print(f"A_v: {E_v:.4f} - {T*S_v:.4f} = {(E_v - T*S_v):.4f} eV")
        print(f"A_t: {E_t:.4f} - {T*S_t:.4f} = {(E_t - T*S_t):.4f} eV")
        print(f"A_r: {E_r:.4f} - {T*S_r:.4f} = {(E_r - T*S_r):.4f} eV")
        print(f"-------------------")
        print(f"A: {total:.4f} eV")
    if mu is not None:
        nelec = outfile.total_electrons
        if isinstance(mu, float):
            total -= mu * nelec
        elif mu is True:
            ferm = outfile.slices[-1].mu
            total -= ferm * nelec
        if verbose:
            print(f"-------------------")
            print(f"G: {total:.4f} eV")
    return total