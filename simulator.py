from __future__ import annotations

import numpy as np
from scipy.linalg import eig

GAMMA_N15 = -2.712_618_04e07


def compute_liouvillian(
    b0: float,
    pb: float,
    kex: float,
    wa_ppm: float,
    wb_ppm: float,
    r2a: float,
    r2b: float,
) -> np.ndarray:
    """
    Compute the Liouvillian matrix for the 1D exchange spectrum.
    """
    pa = 1.0 - pb
    kab = kex * pb
    kba = kex * pa

    wa_rads = GAMMA_N15 * b0 * wa_ppm * 1e-6
    wb_rads = GAMMA_N15 * b0 * wb_ppm * 1e-6

    liouvillian = np.zeros((2, 2), dtype=complex)
    liouvillian[0, 0] = -r2a - kab + 1j * wa_rads
    liouvillian[0, 1] = kba
    liouvillian[1, 0] = kab
    liouvillian[1, 1] = -r2b - kba + 1j * wb_rads

    return liouvillian


def get_spectral_components(
    liouvillian: np.ndarray, pb: float
) -> list[tuple[float, float, float, float]]:
    """
    Compute the spectral components from the Liouvillian matrix.
    """
    eigen_values, eigen_vectors = eig(liouvillian)
    r2_values = -1.0 * eigen_values.real
    w_values = eigen_values.imag

    pa = 1.0 - pb
    m0 = np.array([[pa], [pb]])
    eigen_vectors_inv = np.linalg.inv(eigen_vectors)
    norm_vec = eigen_vectors_inv @ m0

    # "tmp" is a pretty bad name...
    # We detect Ma + Mb
    tmp = eigen_vectors.sum(0) * norm_vec.T
    intensity_values = np.abs(tmp[0, :])
    phase_values = -np.angle(tmp[0, :])

    return list(zip(r2_values, w_values, intensity_values, phase_values))


def lorentzian(
    x: np.ndarray, r2: float, frq: float, intensity: float, phase: float
) -> np.ndarray:
    a = r2**2 + (frq - x) ** 2
    return intensity * (np.cos(phase) * r2 / a + np.sin(phase) * (frq - x) / a)


def calculate_spectrum(
    spectral_width: float, b0: float, spectral_components
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the 1D exchange spectrum.
    """
    x = 0.5 * spectral_width * np.linspace(-1.0, 1.0, 5000)
    x_rads = GAMMA_N15 * b0 * x * 1e-6
    y1 = lorentzian(x_rads, *spectral_components[0])
    y2 = lorentzian(x_rads, *spectral_components[1])
    y = np.asarray(sum((y1, y2)))
    return x, y, y1, y2
