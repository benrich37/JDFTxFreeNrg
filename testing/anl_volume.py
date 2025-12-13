import numpy as np

def anl_sphere_volume(r: float) -> float:
    """Return volume of a sphere.

    Args:
        r (float): Radius of the sphere.

    Returns:
        float: Volume of the sphere.
    """
    return (4/3) * np.pi * r**3

def anl_2sphere_union_volume(r1: float, r2: float, l: float):
    """Return volume of a sphere.

    Args:
        r (float): Radius of the sphere.

    Returns:
        float: Volume of the sphere.
    """
    m = ((r2**2) - (r1**2) + (l**2))/(2*l)
    h1 = r1 + l - m
    h2 = r2 - m
    V11 = np.pi*(h1**2)*(r1-(h1/3))
    V21 = np.pi*(h2**2)*(r2-(h2/3))
    V2 = anl_sphere_volume(r2)
    return V11 + V2 - V21

