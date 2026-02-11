# This should probably be a private module
from qrotor import System
import numpy as np
import scipy.constants as const
import contextlib
import os

def load_man(
        energies: list[float], # in eV
        angles: list[float], # in degrees
        ) -> System:
    system = System()
    positions = np.radians(angles)
    potentials = np.array(energies) * 1000
    system.grid = np.array(positions)
    system.gridsize = len(positions)
    system.potential_values = np.array(potentials)
    return system


def get_system(energies: list[float], angles:list[float], inertia: float, searched_E: int = 400) -> System:
    inertia_SI = inertia * const.physical_constants['atomic mass constant'][0] * 1e-20
    system_B = (const.physical_constants['reduced Planck constant'][0]**2) / (2 * inertia_SI) * (1000 / const.eV)
    system = load_man(energies, angles)
    system.B = system_B
    if searched_E is not None:
        system.searched_E = searched_E
    return system


k_B = const.physical_constants['Boltzmann constant in eV/K'][0]


def get_Z_hr(eigenvalues, T: float = 300.):
    beta = 1 / (k_B * T)
    Z = np.sum(np.exp(-eigenvalues * beta))
    return Z
        

def noprint(f):  
    """Silences print statements"""
    def wrap(*args, **kwargs):  
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                return f(*args, **kwargs)
    return wrap  

@noprint
def _get_helmholtz(system, T: float = 300., gridsize: int | None = None, sym: int = 1):
    if gridsize is None:
        gridsize = system.searched_E + 100
    system.solve(gridsize)
    eigenvalues = system.eigenvalues / 1000.
    return __get_helmholtz(eigenvalues, T, sym=sym)

@noprint
def get_eigenvalues(system, gridsize: int | None = None):
    if gridsize is None:
        gridsize = system.searched_E + 100
    system.solve(gridsize)
    eigenvalues = system.eigenvalues / 1000.
    return eigenvalues

def __get_helmholtz(eigenvalues, T: float = 300., sym: int = 1):
    Z = get_Z_hr(eigenvalues, T) / sym
    return -const.k * T * np.log(Z) / const.eV


def get_helmholtz(system = None,energies=None, degrees=None, inertia = None, T: float = 300., gridsize: int | None = None, searched_E: int = 400, sym: int = 1):
    if system is None:
        system = get_system(energies, degrees, inertia, searched_E=searched_E)
    return _get_helmholtz(system, T=T, gridsize=gridsize, sym=sym)

# Use finite difference to approximate dlnZ/dT, then get entropy and enthalpy from that. Requires making 4 partition functions instead of just 1.
def get_entropy_and_enthalpy(system = None,energies=None, degrees=None, inertia = None, T: float = 300., gridsize: int | None = None, searched_E: int = 400, sym: int = 1):
    if system is None:
        system = get_system(energies, degrees, inertia, searched_E=searched_E)
    system.solve(gridsize)
    eigenvalues = system.eigenvalues / 1000.
    S = get_fd_entropy(eigenvalues, T)
    E = get_fd_enthalpy(eigenvalues, T)
    return S, E


## These should probably be in a separate module

def get_fd_dlnZdT(eigenvalues: np.ndarray, T: float, step=0.1):
    Ts = np.array([T - step, T + step])
    Zs = np.array([get_Z_hr(eigenvalues, _T) for _T in Ts])
    log_Zs = np.log(Zs)
    dlnZdT = (log_Zs[1] - log_Zs[0]) / (2 * step)
    return dlnZdT

def get_fd_entropy(eigenvalues: np.ndarray, T: float, step=0.1):
    dlnZdT = get_fd_dlnZdT(eigenvalues, T, step=step)
    lnZ = np.log(get_Z_hr(eigenvalues, T))
    S = const.k * (lnZ + T * dlnZdT)
    return S / const.eV

def get_fd_enthalpy(eigenvalues: np.ndarray, T: float, step=0.1):
    dlnZdT = get_fd_dlnZdT(eigenvalues, T, step=step)
    E = const.k * T**2 * dlnZdT
    return E / const.eV

def get_fd_Cv(eigenvalues: np.ndarray, T: float, step=0.1):
    Ts = np.array([T - step, T + step])
    Es = np.array([get_fd_enthalpy(eigenvalues, _T, step=step) for _T in Ts])
    Cv = (Es[1] - Es[0]) / (2 * step)
    return Cv

def get_fd_helmholtz(eigenvalues: np.ndarray, T: float, step=0.1):
    S = get_fd_entropy(eigenvalues, T, step=step)
    E = get_fd_enthalpy(eigenvalues, T, step=step)
    A = E - T * S
    return A