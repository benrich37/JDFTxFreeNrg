import json
import numpy as np
import scipy.constants as const
from JDFTxFreeNrg.standard import get_q_rot, check_is_linear, get_entropy_trans, _get_entropy_rot
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from pymatgen.io.jdftx.inputs import JDFTXInfile
from pymatgen.core.structure import Structure
from pathlib import Path
import pyvista as pv
from JDFTxFreeNrg.volume import StructureVolume, get_vdw_volume

# liter is 0.1 m^3, A is 1e-10 m
molarity_to_part_per_A3 = const.Avogadro/(((0.1**3))/((1e-10)**3))
A3_to_liters = ((1e-10)/(0.1))**3  # Å^3 to liters
eV_to_J = const.eV
J_to_eV = 1 / eV_to_J




def solute_hopping_probability(vol_solute: float, vol_solvent: float, vol_free: float) -> float:
    """ Returns probability of solute hopping to an adjacent cavity

    Args:
        vol_solute (float): Volume of solute
        vol_solvent (float): Volume of solvent
        vol_free (float): Volume of free space per solvent

    Returns:
        float: Probability of solute hopping to an adjacent cavity
    """
    vm23 = vol_solute ** (2/3)
    vs23 = vol_solvent ** (2/3)
    vf23 = vol_free ** (2/3)
    prob = max(0, (vf23 - vm23))/(vf23 + vs23)
    return prob

def get_vcav(vol_solute: float, vol_free: float) -> float:
    """ Returns volume of single cavity
    
    Args:
        vol_solute (float): Volume of solute
        vol_free (float): Volume of free space per solvent

    Returns:
        float: Cavity volume
    """
    return (vol_solute**(1/3) + vol_free**(1/3))**3

def num_hopping_cavs(vol_solute: float, vol_solvent: float, vol_free: float) -> float:
    """ Returns effective number of accessible adjacent cavities

    Args:
        vol_solute (float): Volume of solute
        vol_solvent (float): Volume of solvent
        vol_free (float): Volume of free space per solvent

    Returns:
        float: Effective number of accessible adjacent cavities
    """
    v_cav = get_vcav(vol_solute, vol_free)
    r_cav = (3*v_cav*(4*np.pi))**(1/3)
    nx = 4*((4*np.pi/3)**(2/3))*((r_cav**2)/((vol_free**(2/3)) + (vol_solvent**(2/3))))
    return nx

def eff_num_cavities(vol_solute: float, vol_solvent: float, vol_free: float) -> float:
    """ Returns effective number of cavities accessible to solute
    
    Args:
        vol_solute (float): Volume of solute
        vol_solvent (float): Volume of solvent
        vol_free (float): Volume of free space per solvent
        
    Returns:
        float: Effective number of cavities accessible to solute
    """
    nx = num_hopping_cavs(vol_solute, vol_solvent, vol_free)
    x = solute_hopping_probability(vol_solute, vol_solvent, vol_free)
    return 1 + nx*((1/(1-x)) - 1)

def eff_volume(vol_solute: float, vol_solvent: float, vol_free: float) -> float:
    """ Returns effective volume available to a solute in solvent 
    
    Args:
        vol_solute (float): Volume of solute
        vol_solvent (float): Volume of solvent
        vol_free (float): Volume of free space per solvent
    
    Returns:
        float: Effective evailable volume for a solute in solvent
    """
    nc = eff_num_cavities(vol_solute, vol_solvent, vol_free)
    vc = get_vcav(vol_solute, vol_free)
    return nc*vc

def get_vfree(vol_solvent: float, molarity: float) -> float:
    """ Returns free volume per solvent molecule
    
    Args:
        vol_solvent: solvent volume in Å^3
        molarity: molarity in mol/L
        
    Returns:
        float: free volume per solvent molecule in Å^3
    """
    solvent_density = molarity * molarity_to_part_per_A3  # molecules per A^3
    avg_volume_per_molecule = 1 / solvent_density
    vfree = avg_volume_per_molecule - vol_solvent
    return vfree

def get_radius_of_gyration(structure: Structure) -> float:
    """ Returns radius of gyration of a structure

    Args:
        structure (Structure): Structure to evaluate radius of gyration of
    
    Returns:
        float: Radius of gyration in Å
    """
    coords = np.array([site.coords for site in structure.sites])
    return _get_radius_of_gyration(coords)

# Passing to a helper function for easier testing
def _get_radius_of_gyration(coords: np.ndarray) -> float:
    center_of_mass = np.mean(coords, axis=0)
    squared_distances = np.sum((coords - center_of_mass) ** 2, axis=1)
    rg = np.sqrt(np.mean(squared_distances))
    return rg

def get_theta0(rg: float, rf: float) -> float:
    """ Returns angular range of free rotation for restricted rotor model

    Args:
        rg (float): radius of gyration in Å
        rf (float): effective free radius in Å (4/3 pi * vf)^(1/3)

    Returns:
        float: angular range in radians
    """
    return 2 * np.arccos(rg/(np.sqrt(rg**2 + rf**2)))

def get_entropy_conditionally_restricted_rotor(
        structure: Structure, rg: float, rf: float, rm: float, T: float
        ) -> float:
    """ Returns the rotational entropy of a restricted (if rf < rg - rm) or free rigid rotor

    Args:
        structure (Structure): pymatgen Structure
        rg (float): radius of gyration in Å
        rf (float): effective free radius in Å (4/3 pi * vf)^(1/3)
        rm (float): effective solute radius in Å (4/3 pi * vm)^(1/3)
        T (float): temperature in K

    Returns:
        float: rotational entropy in eV/K
    """
    qr = get_q_rot(structure, T)
    if rf <= (rg - rm):
        print("modifying qr for hindered rotor")
        # TODO: Decide if symmetry number should be removed depending on theta0
        # sym = PointGroupAnalyzer(Molecule.from_sites(structure.sites)).get_rotational_symmetry_number()
        theta0 = get_theta0(rg, rf)
        qr *= (theta0/np.pi)**2
    return _get_entropy_rot(qr, linear = check_is_linear(structure))

def get_entropy_change_tr_from_rotation(structure: Structure, vm: float, vf: float, T: float) -> float:
    """ Returns the change in translational entropy due to loss of volume due to radius of gyration

    Args:
        structure (Structure): pymatgen Structure
        vf
    
    """
    vcav = get_vcav(vm, vf)
    rg = get_radius_of_gyration(structure)
    rc = (3*vcav/(4*np.pi))**(1/3)
    deltaV = np.pi*(4/3)*(rc - rg)**3
    rcut = (vf**(1/3)) * (3/(4*np.pi))**(2/3)
    if (rc - rg) < rcut:
        print("using something other than rc - rg ")
        deltaV = (4/3)*np.pi*(rcut**3)
    mass = sum([site.specie.atomic_mass for site in structure])
    return get_entropy_trans(mass, T, deltaV) - get_entropy_trans(mass, T, vcav)



def get_solv_entropy_rot(
        structure: Structure, vm: float, vf: float, T: float
    ) -> float:
    """ Returns rotational entropy of solute in solvent, accounting for restricted rotation and loss of translational entropy

    Args:
        structure (Structure): pymatgen Structure
        vm (float): solute volume in Å^3
        vf (float): free volume in Å^3
        T (float): temperature in K

    Returns:
        float: rotational entropy in eV/K
    """
    rg = get_radius_of_gyration(structure)
    rm = (3*vm/(4*np.pi))**(1/3)
    rf = (3*vf/(4*np.pi))**(1/3)
    S_rot = get_entropy_conditionally_restricted_rotor(structure, rg, rf, rm, T)
    S_rot += get_entropy_change_tr_from_rotation(structure, vm, vf, T)
    return S_rot

def get_solv_entropy_trans(
        structure: Structure, vm: float, vs: float, vf: float, T: float, d: int = 3
    ) -> float:
    """ Returns translational entropy of solute in solvent

    Args:
        structure (Structure): pymatgen Structure
        vm (float): solute volume in Å^3
        vs (float): solvent volume in Å^3
        vf (float): free volume in Å^3
        T (float): temperature in K

    Returns
        float: entropy in eV/K
    """
    veff = eff_volume(vm, vs, vf)
    mass = sum([site.specie.atomic_mass for site in structure])
    return _get_solv_entropy_trans(mass, T, veff, d=d)

def _get_solv_entropy_trans(
        mass, T, veff, d
    ) -> float:
    return get_entropy_trans(mass, T, veff, d=d)

def get_standard_state_correction(T: float, P: float = 101325, M: float = 1) -> float:
    """ Returns standard state correction for entropy from 1 mol/L to ideal gas at given P and T

    Args:
        T (float): temperature in K
        P (float): pressure in Pa
        M (float): molarity in mol/L

    Returns:
        float: Standard state correction in eV/K
    """
    ideal_gas_molarity = (P / (const.R * T))*(1/1000)  # mol/L
    standard_state_correction = const.k * np.log(ideal_gas_molarity / M) * J_to_eV  # eV/K
    return standard_state_correction



