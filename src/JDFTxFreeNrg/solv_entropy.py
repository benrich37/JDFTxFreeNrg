import json
import numpy as np
import scipy.constants as const
from JDFTxFreeNrg.standard import get_q_rot, check_is_linear, get_entropy_trans, _get_entropy_rot
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from pymatgen.io.jdftx.inputs import JDFTXInfile
from pymatgen.core.structure import Structure
from pathlib import Path
from JDFTxFreeNrg.volume import StructureVolume, get_vdw_volume

# liter is 0.1 m^3, A is 1e-10 m
molarity_to_part_per_A3 = const.Avogadro/(((0.1**3))/((1e-10)**3))
A3_to_liters = ((1e-10)/(0.1))**3  # Å^3 to liters
k_ev = const.k / const.eV




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

# TODO: While this argument signature is the most efficient since these terms are used in other expressions, they may not be immediately user friendly.
# We possibly should make this version of the function a private function, and make a more user-friendly public function
# that computed vm, vs, and vf from more intuitive inputs.
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
    mass = structure.composition.weight
    return _get_solv_entropy_trans(mass, T, veff, d=d)

def _get_solv_entropy_trans(
        mass, T, veff, d
    ) -> float:
    return get_entropy_trans(mass, T, veff, d=d)

def get_standard_state_correction(T: float = 300., P: float = 1., M: float = 1.) -> float:
    """ Returns standard state correction for entropy from 1 mol/L to ideal gas at given P and T

    Args:
        T (float): temperature in K
        P (float): pressure in atm
        M (float): molarity in mol/L

    Returns:
        float: Standard state correction in eV/K
    """
    ideal_gas_molarity = ((P * 101325.) / (const.R * T))*(1/1000)  # mol/L
    standard_state_correction = k_ev * np.log(ideal_gas_molarity / M)  # eV/K
    return standard_state_correction

def get_Gc_f(y, R):
    fA = -np.log(1-y) + R*(3/(1-y))
    fB = (R**2)*((3*y/(1-y)) + (9/2)*((y/(1-y))**2))
    return fA + fB

def get_Gc_y(ep_r):
    return (3/(4*np.pi))*((ep_r - 1)/(ep_r + 2))

def get_Gc_R(vol_solute: float, vol_solvent: float):
    return (vol_solute/vol_solvent)**(1/3)

def get_Gc(y, R, T: float = 300.):
    return k_ev*T*get_Gc_f(y, R)

def get_solv_entropy_cav_scp_ep(vol_solute: float, vol_solvent: float, ep_r: float, T: float = 300.):
    """ Returns the cavitation entropy from SCP model in common limit of large Gc/T"""
    return get_Gc(
        get_Gc_y(ep_r),
        get_Gc_R(vol_solute, vol_solvent),
        T = T)/T

def get_Gc_dy_dT_atP(alpha, y):
    return - alpha * y

def get_Gc_df_dy(y: float, R: float):
    numA = -((R**2) * (6*y + 3))
    numB = 3 * R * (y - 1)
    numC = -(y-1)**2
    denom = (y - 1)**3
    return (numA + numB + numC) / denom

def get_dGc_dT_atP(vol_solute: float, vol_solvent: float, ep_r: float, alpha: float, T: float = 300.):
    y = get_Gc_y(ep_r)
    R = get_Gc_R(vol_solute, vol_solvent)
    Gc = get_Gc(y, R, T=T)
    df_fy = get_Gc_df_dy(y, R)
    dy_dT = get_Gc_dy_dT_atP(alpha, y)
    return (Gc / T) + k_ev * T * df_fy * dy_dT

def get_solv_entropy_cav_scp_epalpha(vol_solute: float, vol_solvent: float, ep_r: float, alpha: float, T: float = 300.):
    return - get_dGc_dT_atP(vol_solute, vol_solvent, ep_r, alpha, T=T)

def get_solv_entropy_cav_scp(
        vol_solute: float, vol_solvent: float, ep_r: float, alpha: float | None = None, T: float = 300.
    ):
    """ Returns cavitation entropy from SCP model limit (alpha is None) or full model (alpha provided)

    Args:
        vol_solute (float): solute volume in Å^3
        vol_solvent (float): solvent volume in Å^3
        ep_r (float): Dialectric constant of the solvent
        alpha (float | None): Isobaric volumetric thermal expansion coefficient of the solvent
        T (float): temperature in K
    
    Returns:
        float: cavitation entropy in eV/K
    """
    if alpha is None:
        return get_solv_entropy_cav_scp_ep(vol_solute, vol_solvent, ep_r, T=T)
    else:
        return get_solv_entropy_cav_scp_epalpha(vol_solute, vol_solvent, ep_r, alpha, T=T)
    

def get_Sc0(vol_solute: float, vol_solvent: float, solvent_molarity: float):
    return (k_ev * vol_solute / vol_solvent) * np.log(1 - (solvent_molarity * A3_to_liters * vol_solvent))

def get_mol_geometry_factor(sa_solute: float, sa_solvent: float, box_sa_solute: float, box_sa_solvent: float):
    phi_solv = sa_solvent / box_sa_solvent
    phi_solute = sa_solute / box_sa_solute
    return (sa_solute * phi_solv) / (sa_solvent * phi_solute)



def get_solv_entropy_cav_acc_factor_approx():
    return get_Sc0(vol_solute, vol_solvent, solvent_molarity) - get_mol_geometry_factor()*(get_deltaS_gas_sol() + k_ev * omega * 5.365)