import numpy as np
import scipy.constants as const
from scipy.special import i0, i1
from scipy.special import iv
# def i0(x):
#     return iv(0, x)
# def i1(x):
#     return iv(1, x)
from JDFTxFreeNrg.standard import get_entropy_vib, get_enthalpy_vib

def _get_nu_r(n, W_r, I):
    # W_r and I must be in SI units
    return (n*0.5/np.pi)*np.sqrt(0.5*W_r/I)

def get_nu_r(n, W_r, I):
    # W_r in eV, I in amu*angstrom^2
    return _get_nu_r(n, W_r * const.eV, I*const.value("atomic mass unit-kilogram relationship") * 1e-20)

def get_T_r(n, W_r, I, T: float = 300.):
    # W_r in eV, I in amu*angstrom^2
    nu_r = get_nu_r(n, W_r, I)
    return get_T_x(nu_r, T=T)

def get_T_x(nu_x, T: float = 300.):
    return const.k * T / (const.h * nu_x)

def get_r_r(n, W_r, I):
    r_r = W_r*const.eV / (const.h*get_nu_r(n, W_r, I))
    # print(f"r_r: {r_r}")
    return r_r

def _get_P_rot(r_r, T_r):
    return np.sqrt(np.pi*r_r/T_r)*np.exp(-r_r/(2*T_r))*(i0(r_r/(2*T_r)))

def get_P_rot(n, W_r, I, T: float = 300.):
    r_r = get_r_r(n, W_r, I)
    T_r = get_T_r(n, W_r, I, T=T)
    return _get_P_rot(r_r, T_r)

def _get_f_rot(r_r, T_r):
    return _get_P_rot(r_r, T_r) * np.exp(1/(T_r*(2+(16*r_r))))

# This is numerically equivalent
def _get_f_rot_alt(r_r, T_r, nu_r, T: float = 300.):
    return _get_P_rot(r_r, T_r) * np.exp(_get_pade_delta_E_zpe(nu_r, r_r) / (const.k * T))

def _get_pade_delta_E_zpe(nu_r, r_r):
    return const.h * nu_r / (2 + 16*r_r)

def get_f_rot(n, W_r, I, T: float = 300., alt: bool = False):
    r_r = get_r_r(n, W_r, I)
    # print(f"r_r: {r_r}")
    T_r = get_T_r(n, W_r, I, T=T)
    # print(f"T_r: {T_r}")
    # if alt:
    #     nu_r = get_nu_r(n, W_r, I)
    #     return _get_f_rot_alt(r_r, T_r, nu_r, T=T)
    return _get_f_rot(r_r, T_r)

def __get_q_sHO(T_x):
    return np.exp(-0.5/T_x) / (1 - np.exp(-1/T_x))

def _get_q_sHO(nu_x, T: float = 300.):
    return __get_q_sHO(get_T_x(nu_x, T=T))

def get_q_SHO_r(n, W_r, I, T: float = 300.):
    nu_r = get_nu_r(n, W_r, I)
    # print(f"nu_r: {nu_r}")
    f_rot = get_f_rot(n, W_r, I, T=T)
    # print(f"f_rot: {f_rot}")
    q_SHO = _get_q_sHO(nu_r, T=T)
    # print(f"q_SHO: {q_SHO}")
    return q_SHO * f_rot
    # return _get_q_sHO(get_nu_r(n, W_r, I), T=T) * get_f_rot(n, W_r, I, T=T)

def get_HR_A_rot(n, W_r, I, symmetry: int, T: float = 300.):
    q_rot = get_q_SHO_r(n, W_r, I, T=T)
    # print(f"q_rot: {q_rot}")
    return -1 * const.k * T * np.log(q_rot/symmetry) / const.eV


def _get_entropy_sHO(T_x, T:float=300.):
    x = 1/T_x
    return const.k * ((x / (np.exp(x) - 1)) - np.log(1 - np.exp(-x)))

def _get_enthalpy_sHO(T_x, T: float = 300.):
    x = 1/T_x
    return const.k * T * (x / (np.exp(x) - 1))

def get_HR_S_rot(n, W_r, I, symmetry, T: float = 300.):
    nu_r = get_nu_r(n, W_r, I)
    T_r = get_T_r(n, W_r, I, T=T)
    r_r = get_r_r(n, W_r, I)
    S_rot = get_entropy_vib(nu_r / (const.c * 100), T)
    # S_rot = _get_entropy_sHO(T_r) / const.eV
    S_rot += _get_delta_S_x(r_r, T_r) / const.eV
    S_rot -= const.k * np.log(symmetry) / const.eV
    return S_rot

def _get_delta_S_x(r_x, T_x):
    x = r_x/(2*T_x)
    return -const.k * (0.5 + (x*i1(x)/i0(x)) - np.log(i0(x)*np.sqrt(2*np.pi*x)))

def get_HR_E_rot(n, W_r, I, T: float = 300.):
    nu_r = get_nu_r(n, W_r, I)
    T_r = get_T_r(n, W_r, I, T=T)
    r_r = get_r_r(n, W_r, I)
    # E_rot = get_enthalpy_vib(nu_r / (const.c * 100), T)
    E_rot = (_get_enthalpy_sHO(T_r, T=T) + _get_delta_E_zpe_x(nu_r, r_r)) / const.eV
    E_rot += _get_delta_E_x(r_r, T_r, T=T) / const.eV
    # E_rot += _get_delta_E_zpe_x(nu_r, r_r) / const.eV
    return E_rot

def _get_delta_E_x(r_x, T_x, T:float=300.):
    x = r_x/(2*T_x)
    return -const.k*T*(0.5 + 1/(T_x*(2+(16*r_x))) - x*(1-i1(x)/i0(x)))

def get_HR_A_rot_alt(n, W_r, I, symmetry: int, T: float = 300.):
    E_rot = get_HR_E_rot(n, W_r, I, T=T)
    S_rot = get_HR_S_rot(n, W_r, I, symmetry, T=T)
    return E_rot - T * S_rot

def get_HR_A_rot_nozpe(n, W_r, I, symmetry: int, T: float = 300.):
    T_r = get_T_r(n, W_r, I, T=T)
    return const.k * T * np.log(1 - np.exp(-1/T_r)) / const.eV

def _get_delta_E_zpe_x(nu_x, r_x):
    # return const.h * nu_x / (2 + 16*r_x)
    return const.h * nu_x / 2.