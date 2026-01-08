import numpy as np
from pymatgen.core.structure import Structure

# TODO: Implement the function the surface area of the bounding box for a VdW surface
def get_box_area(structure: Structure):
    raise NotImplementedError("Function get_box_area is not yet implemented.")

def get_mesh_spheres_area(structure: Structure):
    # 1. Generate the 3D mesh based on min/max coordinates of atomic positions plus some padding
    # 2. Identify each voxel 
    raise NotImplementedError("Function get_mesh_spheres_area is not yet implemented.")