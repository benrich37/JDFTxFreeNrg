import numpy as np
import scipy.constants as const
from scipy.special import i0, i1
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

def _get_HO_enthalpy(T_x, T: float = 300.):
    x = 1/T_x
    return const.k * T * (x / (np.exp(x) - 1))

def get_HR_rot_entropy(n, W_r, I, symmetry, T: float = 300.):
    nu_r = get_nu_r(n, W_r, I)
    T_r = get_T_r(n, W_r, I, T=T)
    r_r = get_r_r(n, W_r, I)
    S_rot = get_entropy_vib(nu_r / (const.c * 100), T)
    S_rot += _get_delta_S_x(r_r, T_r) / const.eV
    # sym is assumed to be equal to n in the derivations, this is double counting
    # S_rot -= const.k * np.log(symmetry) / const.eV
    return S_rot

def _get_delta_S_x(r_x, T_x):
    x = r_x/(2*T_x)
    return -const.k * (0.5 + (x*i1(x)/i0(x)) - np.log(i0(x)*np.sqrt(2*np.pi*x)))

def get_HR_rot_enthalpy(n, W_r, I, T: float = 300.):
    nu_r = get_nu_r(n, W_r, I)
    T_r = get_T_r(n, W_r, I, T=T)
    r_r = get_r_r(n, W_r, I)
    E_rot = (_get_HO_enthalpy(T_r, T=T) + _get_delta_E_zpe_x(nu_r, r_r)) / const.eV
    E_rot += _get_delta_E_x(r_r, T_r, T=T) / const.eV
    return E_rot

def _get_delta_E_x(r_x, T_x, T:float=300.):
    x = r_x/(2*T_x)
    return -const.k*T*(0.5 + 1/(T_x*(2+(16*r_x))) - x*(1-i1(x)/i0(x)))

def get_HR_rot_free_energy(n, W_r, I, symmetry: int, T: float = 300.):
    E_rot = get_HR_rot_enthalpy(n, W_r, I, T=T)
    S_rot = get_HR_rot_entropy(n, W_r, I, symmetry, T=T)
    return E_rot - T * S_rot

def _get_delta_E_zpe_x(nu_x, r_x):
    # return const.h * nu_x / (2 + 16*r_x)
    return const.h * nu_x / 2.

