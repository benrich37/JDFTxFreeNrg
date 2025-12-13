import numpy as np
from scipy import constants as const
from JDFTxFreeNrg.standard import __get_ho_vib_enthalpies, _get_ho_vib_entropies

"""
Quasi-RRHO vibrational free energy calculations.

In general, functions with no underscores take frequencies in cm^-1 as input, and return as final energies in eV or eV/K.
Functions with underscores are "internal" functions and skip multiplying by the boltzmann constant for efficiency and expect different frequency forms.
Single underscore for vibrational temperatueres in K, and double underscores for vibrational temperature over T.
"""

eV_to_J = const.eV
kb_J = const.k # [J/K]
R_SI = const.R
Na = const.Avogadro
jpmol_to_evppart = 1/(eV_to_J * Na)

light_speed = const.speed_of_light * 100  # [cm/s]
h_J = const.h  # [J.s]
freq_to_vt = light_speed * h_J / (kb_J)
kb_and_j_to_eV = kb_J / (eV_to_J)


"""
HARMONIC OSCILLATOR FUNCTIONS
"""

"""
ENTHALPY
"""

def _get_ho_vib_enthalpies_zpe(vt: float) -> float:
    """ Returns the zero-point HO vibrational enthalpy in K

    Args:
        vt (float): Vibrational temperature in K
        T (float): temperature in K

    Returns:
        float: Zero-point vibrational enthalpy in K (multiply by k to get J)
    """
    return vt / 2

def __get_ho_vib_enthalpies_no_zpe(vt_over_T: float, T: float) -> float:
    """ Returns the excitation HO vibrational enthalpy

    Args:
        vt_over_T (float): Vibrational temperature over T
        T (float): temperature in K

    Returns:
        float: Excitation HO vibrational enthalpy in K (multiply by k to get J)
    """
    return T * (vt_over_T)*(np.exp(-vt_over_T)/(1 - np.exp(-vt_over_T)))

"""
RIGID ROTOR FUNCTIONS
"""

"""
ENTHALPY
"""

def _get_rr_vib_enthalpies(vts: np.ndarray, T: float = 300.0) -> float:
    """ Returns the rigid rotor vibrational enthalpy

    Args:
        vts (np.ndarray): Vibrational temperatures (or any array with length of number of vibrational modes)

    Returns:
        np.ndarray: Vibrational enthalpies in K (multiply by k to get J)
    """
    # np.ones(...) for compatibility with array input
    return 0.5 * T * np.ones(np.shape(vts))

_rr_vib_entrop_conv1 = h_J / (8 * np.pi**2 * light_speed)
_rr_vib_entrop_conv2 = 8 * np.pi**3 * kb_J / h_J**2
_rr_vib_entrop_conv12 = _rr_vib_entrop_conv1 * _rr_vib_entrop_conv2

"""
ENTROPY
"""

def get_rr_vib_entropies(freq: float, T: float = 300) -> float:
    """ Returns the rigid rotor entropy

    Args:
        freqs (float): A frequency in cm^-1
        T (float): temperature in K

    Returns:
        float: Rigid rotor entropy in K (mulitply by k to get J)
    """
    vt_over_t = (freq * freq_to_vt) / T
    return 0.5*(1+np.log(_rr_vib_entrop_conv12 / vt_over_t))

def _get_rr_vib_entropies(vt_over_t: float) -> float:
    """ Returns the rigid rotor entropy

    Args:
        vt_over_t (float): Vibrational temperature over temperature
        T (float): temperature in K

    Returns:
        float: Rigid rotor entropy in K (mulitply by k to get J)
    """
    return 0.5*(1+np.log(_rr_vib_entrop_conv12 / vt_over_t))

"""
MISC
"""

def get_rrho_mixing(freq: float | np.ndarray, freq0: float = 200, alpha: float = 4) -> float:
    """ Return the mixing percentage

    Args:
        freq (float): A frequency in cm^-1
        freq0 (float): threshold frequency in cm^-1
        alpha (float): parameter controlling the sharpness of the transition

    Returns:
        float: Mixing percentage of standard HO contribution
    """
    # Freq can also be in vibrational temperature units, as long as freq0 is converted the same way
    return 1 / (1 + ((freq0/freq)**alpha))

"""
QRRHO
"""

def get_qrrho_vib_enthalpies(freqs: np.ndarray | float, T: float = 300., sep_zpe: bool = True, freq0: float = 100, alpha: float = 4.) -> float:
    """ Return the QRHHO vibrational enthalpy in eV

    Args:
        freqs (float): A frequency in cm^-1
        T (float): temperature in K
        freq0 (float): threshold frequency in cm^-1
        alpha (float): parameter controlling the sharpness of the transition

    Returns:
        float: QRRHO vibrational enthalpy in eV
    """
    mixs = get_rrho_mixing(freqs, freq0, alpha)
    vts = freqs * freq_to_vt
    vt_over_Ts = vts / T
    Es_rr = _get_rr_vib_enthalpies(vts, T)
    As_rr = (1 - mixs)*(Es_rr)
    As_ho = np.zeros_like(freqs)
    if sep_zpe:
        As_ho += __get_ho_vib_enthalpies_no_zpe(vt_over_Ts, T)
        As_ho *= mixs
        As_ho += _get_ho_vib_enthalpies_zpe(vts)
    else:
        As_ho += __get_ho_vib_enthalpies(vt_over_Ts, T=T)
        As_ho *= mixs
    return (As_rr + As_ho) * kb_and_j_to_eV

def get_qrrho_vib_entropies(freqs: np.ndarray | float, T: float = 300.0, freq0: float = 100., alpha: float = 4) -> float:
    """ Return the QRHHO vibrational entropy in eV/K

    Args:
        freqs (float): A frequency in cm^-1
        T (float): temperature in K
        freq0 (float): threshold frequency in cm^-1
        alpha (float): parameter controlling the sharpness of the transition

    Returns:
        float: QRRHO vibrational entropy in eV/K
    """
    mixs = get_rrho_mixing(freqs, freq0, alpha)
    vts = freqs * freq_to_vt
    vt_over_Ts = vts / T
    Ss_ho = _get_ho_vib_entropies(vt_over_Ts)
    Ss_rr = _get_rr_vib_entropies(vt_over_Ts)
    return (((1 - mixs) * Ss_rr) + (mixs * Ss_ho)) * kb_and_j_to_eV

def get_qrrho_vib_free_energies(freqs: np.ndarray | float, T: float = 300.0, freq0: float = 100., alpha: float = 4, sep_zpe: bool = True) -> float:
    """ Return the QRHHO vibrational Helmholtz free energy in eV

    Args:
        freqs (float): A frequency in cm^-1
        T (float): temperature in K
        freq0 (float): threshold frequency in cm^-1
        alpha (float): parameter controlling the sharpness of the transition

    Returns:
        float: QRRHO vibrational Helmholtz free energy in eV
    """
    # Set freq0 to 0 for pure HO
    mixs = get_rrho_mixing(freqs, freq0, alpha)
    vts = freqs * freq_to_vt
    vt_over_Ts = vts / T
    Ss_rr = _get_rr_vib_entropies(vt_over_Ts)
    Ss_ho = _get_ho_vib_entropies(vt_over_Ts)
    Es_rr = _get_rr_vib_enthalpies(vts, T)
    As_rr = (1 - mixs)*(Es_rr - T*Ss_rr)
    As_ho = -T*Ss_ho
    if sep_zpe:
        As_ho += __get_ho_vib_enthalpies_no_zpe(vt_over_Ts, T)
        As_ho *= mixs
        As_ho += _get_ho_vib_enthalpies_zpe(vts)
    else:
        As_ho += __get_ho_vib_enthalpies(vt_over_Ts, T=T)
        As_ho *= mixs
    return (As_rr + As_ho) * kb_and_j_to_eV