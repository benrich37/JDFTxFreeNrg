from pymatgen.analysis.quasirrho import get_avg_mom_inertia as _get_avg_mom_inertia
from pymatgen.core.structure import Structure, Molecule
from scipy import constants as const
import numpy as np
import math
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

freq_to_vt = 100 * const.speed_of_light * const.Planck / const.k


def get_avg_mom_inertia(structure: Structure):
    molecule = Molecule.from_sites(structure.sites)
    return _get_avg_mom_inertia(molecule)


def get_ideal_gas_vol(P: float, T: float) -> float:
    """Return volume of ideal gas at given P and T.

    Args:
        P (float): Pressure in atm
        T (float): Temperature in K

    Returns:
        float: Volume in Å^3
    """
    si_P = P * 101325  # Pa
    si_vol = const.k * T / si_P  # m^3
    return si_vol * 1e30  # Å^3


def check_is_linear(structure: Structure) -> bool:
    """Return whether a molecule is linear.

    Args:
        structure (Structure): Pymatgen Structure object

    Returns:
        bool: True if linear, False otherwise
    """
    mol = Molecule.from_sites(structure.sites)
    coords = mol.cart_coords
    v0 = coords[1] - coords[0]
    linear = True
    for coord in coords[1:]:
        theta = abs(np.dot(coord - coords[0], v0) / np.linalg.norm(coord - coords[0]) / np.linalg.norm(v0))
        if not math.isclose(theta, 1, abs_tol=1e-4):
            linear = False
    return linear

def get_sym_number(structure: Structure) -> int:
    """Return the rotational symmetry number of a molecule.
    
    Args:
        structure (Structure): Pymatgen Structure object
    
    Returns:
        int: Rotational symmetry number
    """
    molecule = Molecule.from_sites(structure.sites)
    pga = PointGroupAnalyzer(molecule)
    sym = pga.get_rotational_symmetry_number()
    return sym

"""
TRANS
"""

def get_q_trans(mass: float, T: float, vol: float, d: int = 3) -> float:
    """Return the translational partition function.

    Args:
        mass (float): Mass in amu
        T (float): Temperature in K
        vol (float): in Å^d (cubic Angstroms for d=3)
        d (int): dimensionality of free translations

    Returns: 
        float: Translational partition function (unitless)
    """
    si_vol = vol * ((1e-10)**d)  # m^3
    si_mass = mass * (1/(1000*const.Avogadro))
    return si_vol * (
        2 * np.pi * si_mass * const.k * T / const.h**2
    )**(d/2)

def get_entropy_trans(mass: float, T: float, vol: float, d: int = 3) -> float:
    """Returns the translational entropy in J/K.

    Args:
        mass (float): Mass in amu
        T (float): Temperature in K
        vol (float): in Å^d (cubic Angstroms for d=3)
        d (int): dimensionality of free translations

    Returns: 
        float: Entropy in J/K

    Note - this includes the +k term in the Sackur-Tetrode equation, so if you are adding
    in entropies from other sources, you may need to subtract that out.
    """
    q = get_q_trans(mass, T, vol, d=d)
    return const.k * (np.log(q) + 5/2)

def get_enthalpy_trans(T: float, d=3):
    """ Returns the translational enthalpy in J.
    Args:
        T (float): Temperature in K
        d (int): dimensionality of free translations

    Returns: 
        float: Enthalpy in J
    """
    return (d/2) * const.k * T

"""
ROT
"""

def get_q_rot(structure: Structure, T: float = 300.0, linear: bool | None = None) -> float:
    """Return the rotational partition function.
    
    Args:
        structure (Structure): Pymatgen Structure (for getting moments of inertia and checking linearity and symmetry)
        T (float): temperature in K
        linear (bool | None): Whether the molecule is linear. If None, will check automatically
        
    Returns: 
        float: Rotational partition function (unitless)
    """
    if linear is None:
        linear = check_is_linear(structure)
    sym = get_sym_number(structure)
    _, (i1, i2, i3) = get_avg_mom_inertia(structure)
    qr = (np.sqrt(np.pi*np.prod(np.array([i1, i2, i3])))/sym) * (
        (8 * np.pi**2 * const.k * T / const.Planck**2)**(3/2)
    )
    if linear:
        qr = (8 * np.pi**2 * max(i1, i2, i3) * const.k * T / (sym * const.Planck**2))
    else:
        qr = (np.sqrt(np.pi*np.prod(np.array([i1, i2, i3])))/sym) * (
        (8 * np.pi**2 * const.k * T / const.Planck**2)**(3/2)
    )
    return qr

def _get_entropy_rot(qr: float, linear: bool = False) -> float:
    """Return the rotational entropy in J/K.

    Args:
        qr (float): Rotational partition function (unitless)
        linear (bool): Whether the molecule is linear

    Returns: 
        float: Entropy in J/K
    """
    return const.k * (np.log(qr) + 1 + 0.5*(not linear))


def get_entropy_rot(structure: Structure,  T: float):
    """ Return the rotational entropy in J/K.

    Args:
        structure (Structure): pymatgen Structure (for getting moments of inertia and checking linearity and symmetry)
        T (float): temperature in K

    Returns:
        float: Entropy in J/K
    """
    qr = get_q_rot(structure, T)
    return _get_entropy_rot(qr, linear = check_is_linear(structure))



def get_enthalpy_rot(structure: Structure,  T: float, d: int = 3):
    """ Return the rotational enthalpy in J.

    Args:
        structure (Structure): pymatgen Structure (for checking linearity)
        T (float): temperature in K
        d (int): dimensionality of free rotations

    Returns:
        float: Enthalpy in J
    """
    if check_is_linear(structure):
        d = min(2, d)
    return (d/2) * const.k * T

"""
VIB
"""

def __get_ho_vib_enthalpies(vt_over_T: float | np.ndarray, T: float) -> float:
    """ Return the HO vibrational enthalpy in K.

    Args:
        vt_over_T (float): Vibrational temperature over T
        T (float): temperature in K

    Returns:
        float: enthalpy in K (multiply by k to get energy)
    """
    return T * vt_over_T * (
        0.5 + (np.exp(-vt_over_T) / (1 - np.exp( - vt_over_T)))
        )

def get_enthalpy_vib(freq: float | np.ndarray, T: float) -> float:
    """ Return the HO vibrational enthalpy in J.

    Args:
        freq (float): Vibrational frequency in cm^-1
        T (float): temperature in K

    Returns:
        float: enthalpy in J
    """
    vt_over_T = freq * freq_to_vt / T
    return __get_ho_vib_enthalpies(vt_over_T, T) * const.k


def _get_ho_vib_entropies(vt_over_t: float | np.ndarray) -> float:
    """ Return the HO vibrational entropy in unitless.

    Args:
        vt: Vibrational temperature

    returns entropy in unitless (multiply by k to get J/K)
    """
    return vt_over_t / ((np.exp(vt_over_t) - 1)) - np.log(1 - np.exp(-vt_over_t))

def get_entropy_vib(freq: float | np.ndarray, T: float) -> float:
    """ Return the HO vibrational entropy in J/K.

    Args:
        freq (float): Vibrational temperature
        T (float): temperature in K

    Returns:
        float: entropy in J/K
    """
    vt_over_t = freq * freq_to_vt / T
    return _get_ho_vib_entropies(vt_over_t) * const.k