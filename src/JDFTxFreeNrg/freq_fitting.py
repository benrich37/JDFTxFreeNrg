from pymatgen.io.jdftx.outputs import JDFTXOutfile
from pathlib import Path
import numpy as np
from JDFTxFreeNrg.hessian import freq_nrg_to_cm
from ase.units import Hartree, Bohr
from scipy import constants as const
import matplotlib.pyplot as plt


def get_freq_from_scan_outfile(
        outfile_path: Path | str, i_min: int | None = None, i_max: int | None = None, 
        target_disp: float = 0.005, plot_fit: bool = True, conv_disp: bool = True, anh: bool = True
        ) -> float:
    """Extract frequency from JDFTX scan outfile by fitting quadratic to energy vs displacement.
    
    Args:
        outfile_path (Path | str): Path to JDFTX scan outfile.
        i_min (int | None): Minimum index for fitting. If None, determined from target_disp.
        i_max (int | None): Maximum index for fitting. If None, determined from target_disp.
        target_disp (float): Target Cartesian displacement in Bohr to determine fitting range.
        plot_fit (bool): Whether to save a plot the fit to the outfile_path's parent.
    
    Returns:
        float: Frequency in cm^-1.
    """
    # Assumes the scan moves linearly along the reaction coordinate
    outfile = JDFTXOutfile.from_file(outfile_path)
    traj = outfile.trajectory
    disps, energies, unweighted_disps = get_mass_weighted_displacements(traj, sort=True, add_unweighted=True)
    if conv_disp:
        target_disp = np.sqrt(2*(target_disp / Bohr)**2)
    if None in (i_min, i_max):
        i0 = np.argmin(energies)
        if i_min is None:
            i_min = get_target_imin(unweighted_disps, target_disp, i0)
        if i_max is None:
            i_max = get_target_imax(unweighted_disps, target_disp, i0)
        print(f"Determined fit range indices: i_min={i_min}, i_max={i_max} (i0={i0})")
    coefs, r2 = fit_quadratic_to_slice(disps, energies, i_min, i_max, anh=anh)
    freq_cm = fit_coefs_to_freq_cm(coefs)
    if plot_fit:
        plot_scan_fit(Path(outfile_path).parent, disps, unweighted_disps, energies, i_min, i_max, coefs, anh=anh)
    return freq_cm


def fit_coefs_to_freq_cm(coefs: tuple[float, float]) -> float:
    return freq_nrg_to_cm(np.sqrt(coefs[0] * (Bohr**2 / Hartree)))


def plot_scan_fit(save_dir: Path, disps, unweighted_disps, energies, i_min, i_max, coefs, typical_bohr_disp: float = 0.01, anh: bool = True):
    xs = np.linspace(disps[i_min], disps[i_max], 100)
    ys = quadratic(xs, *coefs) if anh else harmonic(xs, *coefs)
    fig, ax = plt.subplots()
    twiny = ax.twiny()
    ax.plot(disps, energies, c="black", zorder=-1)
    twiny.plot([np.nan], [np.nan], c="black", label="Data")
    ax.scatter(disps, energies, c="black", zorder=-1)
    ax.fill_betweenx([0, np.max(energies)], x1=disps[i_min], x2=disps[i_max], color="gray", alpha=0.1)
    twiny.fill_betweenx([0, np.max(energies)], x1=disps[i_min], x2=np.nan, color="gray", alpha=0.3, label="Fitted region")
    twiny.fill_betweenx([0,0], x1=np.nan, x2=np.nan, color="red", alpha=0.3, label="Typical Cartesian displacement range")
    freq_cm = fit_coefs_to_freq_cm(coefs)
    ax.plot(xs, ys, color="green", label=f"Fit freq: {freq_cm:.1f} cm$^{{-1}}$")
    twiny.plot(xs, ys*np.nan, color="green", label=f"Fit freq: {freq_cm:.1f} cm$^{{-1}}$")
    ax.set_ylabel("Energy (eV)")
    ax.set_xlabel("Mass-weighted displacement (Å$\sqrt{{amu}}$)")
    twiny.set_xlabel("Cartesian displacement (Å)")
    max_ang_disp = np.sqrt(2*(typical_bohr_disp / Bohr)**2)
    twiny.fill_betweenx([0, np.max(energies)], x1=-max_ang_disp, x2=max_ang_disp, color="red", alpha=0.1)
    twiny.plot(unweighted_disps, energies, c="black")
    xs = np.linspace(unweighted_disps[i_min], unweighted_disps[i_max], 100)
    twiny.plot(xs, ys, color="green")
    twiny.legend()
    fig.savefig(save_dir / "scan_fit.png", dpi=300)

def get_target_imin(nonweighted_disps: list[float], target_disp: float, i0: int) -> int:
    _idcs = _get_target___idcs(nonweighted_disps, target_disp, i0)
    idcs = [idx for idx in _idcs if idx < i0]
    return min(idcs[0], idcs[1])

def get_target_imax(nonweighted_disps: list[float], target_disp: float, i0: int) -> int:
    _idcs = _get_target___idcs(nonweighted_disps, target_disp, i0)
    idcs = [idx for idx in _idcs if idx > i0]
    return max(idcs[0], idcs[1])

def _get_target___idcs(nonweighted_disps: list[float], target_disp: float, i0: int):
    abs_nonweighted_disps = np.abs(np.array(nonweighted_disps))
    dev_abs_nonweighted_disps = abs_nonweighted_disps - target_disp
    abs_dev_abs_nonweighted_disps = abs(dev_abs_nonweighted_disps)
    _idcs = np.argsort(abs_dev_abs_nonweighted_disps)
    return _idcs



def get_mass_weighted_displacements(traj, sort=True, add_unweighted: bool = False):
    energies = np.array([frame["energy"] for frame in traj.frame_properties])
    energies -= np.min(energies)
    min_idx = np.argmin(energies)
    traj_cart_coords = np.array([frame.cart_coords.flatten() for frame in traj])
    cum_displacements = np.array([np.linalg.norm(tcc.flatten() - traj_cart_coords[min_idx]) for tcc in traj_cart_coords])
    zero_disp_idcs = [i for i, disp in enumerate(cum_displacements) if ((np.isclose(disp, 0.)) and (i != min_idx))]
    traj_cart_coords = np.array([frame.cart_coords for frame in traj])
    cleaned_energies = np.array([energy for i, energy in enumerate(energies) if i not in zero_disp_idcs])
    cleaned_traj_cart_coords = np.array([coords for i, coords in enumerate(traj_cart_coords) if i not in zero_disp_idcs])
    mass_vector = np.array([site.specie.atomic_mass for site in traj[0].sites]) * 1822.888
    cleaned_traj_cart_coords *= np.sqrt(mass_vector)[:, np.newaxis]
    cum_displacements = get_signed_displacement_magnitudes(cleaned_traj_cart_coords, cleaned_energies)
    if sort:
        idcs = np.argsort(cum_displacements)
        cum_displacements = cum_displacements[idcs]
        cleaned_energies = cleaned_energies[idcs]
    if not add_unweighted:
        return cum_displacements, energies
    else:
        unit_conv = 1/np.sqrt(mass_vector)[:, np.newaxis]
        unweighted_cum_displacement = get_signed_displacement_magnitudes(cleaned_traj_cart_coords * unit_conv, cleaned_energies)
        return cum_displacements, cleaned_energies, unweighted_cum_displacement

def get_signed_displacement_magnitudes(cart_coords_list, energies):
    min_idx = np.argmin(energies)
    idcs_lower = list(range(0, min_idx))
    idcs_upper = list(range(min_idx + 1, len(energies)))
    displacement_vecs = np.zeros_like(cart_coords_list)
    for idx in idcs_upper:
        displacement_vecs[idx] = cart_coords_list[idx] - cart_coords_list[idx - 1]
    for idx in reversed(idcs_lower):
        displacement_vecs[idx] = cart_coords_list[idx] - cart_coords_list[idx + 1]
    # displacements_magnitudes = np.linalg.norm(displacement_vecs.reshape(len(cleaned_traj_cart_coords), -1), axis=1)
    displacements_signs = np.zeros_like(energies)
    displacements_signs[idcs_upper] += 1
    displacements_signs[idcs_lower] -= 1
    displacements_magnitudes = np.linalg.norm(displacement_vecs.reshape(len(cart_coords_list), -1), axis=1)
    cum_displacements = np.zeros_like(displacements_magnitudes)
    for idx in idcs_upper:
        cum_displacements[idx] = cum_displacements[idx - 1] + displacements_signs[idx] * displacements_magnitudes[idx]
    for idx in reversed(idcs_lower):
        cum_displacements[idx] = cum_displacements[idx + 1] + displacements_signs[idx] * displacements_magnitudes[idx]
    return cum_displacements

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# def quadratic(x, a, b, c):
#     return a * x**2 + b * x + c

# def fit_quadratic(x, y):
#     coeffs, covariance = curve_fit(quadratic, x, y)
#     return coeffs, covariance

def quadratic(x, a, b, c):
    return a * x**2 + b*x + c

def fit_quadratic(x, y):
    coeffs, covariance = curve_fit(quadratic, x, y)
    return coeffs, covariance

def harmonic(x, a, c):
    return a * x**2 + c

def fit_harmonic(x, y):
    coeffs, covariance = curve_fit(harmonic, x, y)
    return coeffs, covariance

def fit_quadratic_to_slice(x, y, i_min, i_max, anh: bool = True):
    i_min = max(i_min, 0)
    i_max = min(i_max, len(x))
    x_slice = x[i_min:i_max]
    y_slice = y[i_min:i_max]
    if anh:
        coeffs, cov = fit_quadratic(x_slice, y_slice)
        r2 = r2_score(y_slice, quadratic(x_slice, *coeffs))
    else:
        coeffs, cov = fit_harmonic(x_slice, y_slice)
        r2 = r2_score(y_slice, harmonic(x_slice, *coeffs))
    return coeffs, r2

def get_coef_fits(traj, d_min: int = 3, d_max: int = None):
    displacements_magnitudes, energies = get_mass_weighted_displacements(traj)
    mindx = np.argmin(energies)
    if d_max is None:
        d_max = len(energies)
    else:
        d_max = min(d_max, len(energies))
    coeffs_list = []
    r2_list = []
    d_list = []
    for d in range(d_min, d_max):
        coeffs, r2 = fit_quadratic_to_slice(displacements_magnitudes, energies, mindx - d, mindx + d)
        coeffs_list.append(coeffs)
        r2_list.append(r2)
        d_list.append(d)
    return coeffs_list, r2_list, d_list