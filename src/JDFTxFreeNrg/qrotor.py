# This should probably be a private module
from qrotor import System
import numpy as np
import scipy.constants as const

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
    return 


def get_system(energies: list[float], angles:list[float], inertia: float, searched_E: int = 400) -> System:
    inertia_SI = inertia * const.physical_constants['atomic mass constant'][0] * 1e-20
    system_B = (const.physical_constants['reduced Planck constant'][0]**2) / (2 * inertia_SI) * (1000 / const.eV)
    system = load_man(energies, angles)
    system.B = system_B
    if searched_E is not None:
        system.searched_E = searched_E
    return system