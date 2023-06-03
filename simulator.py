from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig

GAMMA_N15 = -2.712_618_04e07


@dataclass
class Parameters:
    b0: float = 18.7
    pb: float = 0.2
    kex: float = 200.0
    δa: float = -2.0
    δb: float = 2.0
    r2a: float = 10.0
    r2b: float = 10.0


@dataclass
class Component:
    r2: float
    frq: float
    intensity: float
    phase: float


@dataclass
class Coordinates:
    x: np.ndarray
    y: np.ndarray
    y1: np.ndarray
    y2: np.ndarray


def lorentzian(x: np.ndarray, comp: Component) -> np.ndarray:
    a = comp.r2**2 + (comp.frq - x) ** 2
    return comp.intensity * (
        np.cos(comp.phase) * comp.r2 / a + np.sin(comp.phase) * (comp.frq - x) / a
    )


def compute_liouvillian(params: Parameters) -> np.ndarray:
    pa = 1.0 - params.pb
    kab = params.kex * params.pb
    kba = params.kex * pa

    ωa = GAMMA_N15 * params.b0 * params.δa * 1e-6
    ωb = GAMMA_N15 * params.b0 * params.δb * 1e-6

    liouvillian = 1j * np.diag([ωa, ωb])
    liouvillian += np.array([[-kab, +kba], [+kab, -kba]])
    liouvillian += np.diag([-params.r2a, -params.r2b])

    return liouvillian


def get_spectral_components(params: Parameters) -> list[Component]:
    liouvillian = compute_liouvillian(params)

    eigen_values, eigen_vectors = eig(liouvillian)

    r2s = -1.0 * eigen_values.real
    frequencies = eigen_values.imag

    norm_vec = np.linalg.inv(eigen_vectors) @ np.array([[1.0 - params.pb], [params.pb]])
    tmp = eigen_vectors.sum(0) * norm_vec.T
    intensities = np.abs(tmp[0, :])
    phases = -np.angle(tmp[0, :])

    components = [
        Component(*values) for values in zip(r2s, frequencies, intensities, phases)
    ]

    return components


def calculate_spectrum(spectral_width: float, params: Parameters) -> Coordinates:
    x = spectral_width * np.linspace(-0.5, 0.5, 5000)
    x_rads = x * GAMMA_N15 * params.b0 * 1e-6
    components = get_spectral_components(params)
    y1 = lorentzian(x_rads, components[0])
    y2 = lorentzian(x_rads, components[1])
    y = np.asarray(sum((y1, y2)))
    return Coordinates(x, y, y1, y2)
