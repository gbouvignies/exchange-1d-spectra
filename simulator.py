from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import warnings

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from scipy.linalg import eig

# --- Physical Constants ---

# Gyromagnetic ratio for ¹⁵N in rad·s⁻¹·T⁻¹
GAMMA_N15 = -2.712_618_04e07
# Default spectral width for calculations (ppm)
DEFAULT_SPECTRAL_WIDTH = 15.0
# Default number of points for spectrum calculation
DEFAULT_NPOINTS = 5000


@dataclass
class Parameters:
    """Holds the physical parameters for a two-state NMR simulation.
    
    Attributes:
        b0: Magnetic field strength in Tesla
        pb: Population of state B (0 ≤ pb ≤ 1)
        kex: Exchange rate constant in s⁻¹ 
        δa: Chemical shift of state A in ppm
        δb: Chemical shift of state B in ppm
        r2a: Transverse relaxation rate of state A in s⁻¹
        r2b: Transverse relaxation rate of state B in s⁻¹
    """

    b0: float = 18.7
    pb: float = 0.2
    kex: float = 200.0
    δa: float = -2.0
    δb: float = 2.0
    r2a: float = 10.0
    r2b: float = 10.0
    
    def __post_init__(self):
        """Validate parameter values."""
        if not 0 <= self.pb <= 1:
            raise ValueError(f"pb must be between 0 and 1, got {self.pb}")
        if self.b0 <= 0:
            raise ValueError(f"b0 must be positive, got {self.b0}")
        if self.kex < 0:
            raise ValueError(f"kex must be non-negative, got {self.kex}")
        if self.r2a < 0 or self.r2b < 0:
            raise ValueError("Relaxation rates must be non-negative")
    
    @property
    def pa(self) -> float:
        """Population of state A."""
        return 1.0 - self.pb
    
    @property
    def delta_omega_hz(self) -> float:
        """Chemical shift difference in Hz."""
        return abs(self.δb - self.δa) * abs(GAMMA_N15) * self.b0 * 1e-6 / (2 * np.pi)
    
    @property 
    def delta_omega_rads(self) -> float:
        """Chemical shift difference in rad/s."""
        return abs(self.δb - self.δa) * abs(GAMMA_N15) * self.b0 * 1e-6


@dataclass
class Component:
    """Represents a single Lorentzian component of a spectrum.
    
    Attributes:
        r2: Transverse relaxation rate in s⁻¹
        frq: Frequency in rad/s
        intensity: Peak intensity (arbitrary units)
        phase: Phase in radians
    """

    r2: float
    frq: float
    intensity: float
    phase: float


@dataclass
class Coordinates:
    """Holds the X and Y coordinates for plotting.
    
    Attributes:
        x: Frequency axis in ppm
        y: Total spectrum intensity
        y1: Component 1 intensity
        y2: Component 2 intensity
    """

    x: np.ndarray
    y: np.ndarray
    y1: np.ndarray
    y2: np.ndarray


# --- Core Calculation Functions (Unchanged) ---


def lorentzian(x: np.ndarray, comp: Component) -> np.ndarray:
    """Calculates a single Lorentzian/anti-Lorentzian line shape."""
    a = comp.r2**2 + (comp.frq - x) ** 2
    return comp.intensity * (
        np.cos(comp.phase) * comp.r2 / a + np.sin(comp.phase) * (comp.frq - x) / a
    )


def compute_liouvillian(params: Parameters) -> np.ndarray:
    """Computes the Liouvillian matrix for two-site exchange."""
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
    """Gets the spectral components from the Liouvillian matrix."""
    liouvillian = compute_liouvillian(params)
    eigen_values, eigen_vectors = eig(liouvillian)
    r2s = -1.0 * eigen_values.real
    frequencies = eigen_values.imag
    norm_vec = np.linalg.inv(eigen_vectors) @ np.array([[1.0 - params.pb], [params.pb]])
    tmp = eigen_vectors.sum(0) * norm_vec.T
    intensities = np.abs(tmp[0, :])
    phases = -np.angle(tmp[0, :])
    return [Component(*values) for values in zip(r2s, frequencies, intensities, phases)]


def calculate_spectrum(spectral_width: float, params: Parameters, npoints: int = DEFAULT_NPOINTS) -> Coordinates:
    """Calculates the final spectrum from the spectral components.
    
    Args:
        spectral_width: Spectral width in ppm
        params: Parameters object containing simulation parameters
        npoints: Number of points in the spectrum
        
    Returns:
        Coordinates object containing x and y data for plotting
    """
    x = spectral_width * np.linspace(-0.5, 0.5, npoints)
    x_rads = x * GAMMA_N15 * params.b0 * 1e-6
    components = get_spectral_components(params)
    y1 = lorentzian(x_rads, components[0])
    y2 = lorentzian(x_rads, components[1])
    y = y1 + y2
    return Coordinates(x, y, y1, y2)


# --- New High-Level Functions Moved from the Notebook ---


def calculate_el(kd: float, e_tot: float, l_tot: float) -> float:
    """Computes the concentration of the bound complex [EL].
    
    Solves the quadratic equation for a simple binding equilibrium:
    E + L ⇌ EL with dissociation constant Kd = [E][L]/[EL]
    
    Args:
        kd: Dissociation constant (same units as concentrations)
        e_tot: Total enzyme/target concentration
        l_tot: Total ligand concentration
        
    Returns:
        Concentration of bound complex [EL]
        
    Raises:
        ValueError: If any concentration is negative or Kd is non-positive
    """
    if kd <= 0:
        raise ValueError(f"Kd must be positive, got {kd}")
    if e_tot < 0 or l_tot < 0:
        raise ValueError("Concentrations must be non-negative")
    
    sum_ = e_tot + l_tot + kd
    discriminant = sum_**2 - 4 * e_tot * l_tot
    
    if discriminant < 0:
        warnings.warn("Negative discriminant in binding calculation, returning 0")
        return 0.0
        
    return 0.5 * (sum_ - np.sqrt(discriminant))


def plot_two_state_exchange(ax, yscale: bool, **kwargs):
    """
    Calculates and plots the spectrum for a simple two-state exchange.
    This function is designed to be called by `ipywidgets.interactive_output`.

    Args:
        ax: The matplotlib Axes object to plot on.
        yscale: Boolean to control y-axis rescaling.
        **kwargs: Widget values passed as keyword arguments (b0, kex, pb, etc.).
    """
    params = Parameters(**kwargs)
    coordinates = calculate_spectrum(spectral_width=DEFAULT_SPECTRAL_WIDTH, params=params)

    # Store current y-limits if not rescaling
    ylim = ax.get_ylim()

    ax.cla()
    ax.plot(coordinates.x, coordinates.y1, "--", color="C1", lw=1, label="Component 1")
    ax.plot(coordinates.x, coordinates.y2, "--", color="C2", lw=1, label="Component 2")
    ax.plot(coordinates.x, coordinates.y, color="C0", lw=2, label="Total")

    if not yscale:
        ax.set_ylim(*ylim)

    ax.set_xlabel("¹⁵N (ppm)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.grid(True, linestyle=":", alpha=0.5)


def plot_binding_titration(axd: Dict[str, any], **kwargs):
    """
    Calculates and plots a full binding titration series, including spectra
    and the bound fraction curve. Designed for `ipywidgets.interactive_output`.

    Args:
        axd: Dictionary of named matplotlib Axes objects.
        **kwargs: Widget values (b0, lmax_um, kd_um, etc.).
    """
    # Unpack parameters from the widgets
    b0 = kwargs["b0"]
    lmax_um = kwargs["lmax_um"]
    kd_um = kwargs["kd_um"]
    kon = kwargs["kon"]
    δfree = kwargs["δfree"]
    δbound = kwargs["δbound"]
    r2free = kwargs["r2free"]
    r2bound = kwargs["r2bound"]

    # Concentration units and fixed parameters
    unit = 1e-6
    e_tot = 100.0 * unit
    ratios = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2]) / 2.0
    kd = kd_um * unit
    koff = kon * kd
    cmap = plt.get_cmap("cool")

    # Clear old plots
    axd["spectra"].cla()
    axd["fraction"].cla()
    axd["legend"].cla()
    axd["legend"].axis("off")

    pb_list = []

    for index, ratio in enumerate(ratios):
        l_tot_um = ratio * lmax_um
        l_tot = l_tot_um * unit

        e_bound = calculate_el(kd, e_tot, l_tot)
        l_free = l_tot - e_bound

        kex = kon * l_free + koff
        pb = e_bound / e_tot
        pb_list.append(pb)

        spec_params = Parameters(b0, pb, kex, δfree, δbound, r2free, r2bound)
        coordinates = calculate_spectrum(spectral_width=8.0, params=spec_params)

        axd["spectra"].plot(
            coordinates.x,
            coordinates.y + 0.01 * index,
            label=f"$L_{{0}}$ = {l_tot_um:6.1f} μM",
            color=cmap(ratio),
            zorder=100000 - index * 10,
            lw=2,
        )

    # Plot text annotations and labels
    axd["spectra"].text(
        0.05,
        0.90,
        f"$k_{{on}}$ = {kon:6.1e} M⁻¹s⁻¹\n$k_{{off}}$ = {koff:6.1e} s⁻¹",
        transform=axd["spectra"].transAxes,
    )
    axd["spectra"].set_xlabel("¹⁵N (ppm)")
    axd["spectra"].set_ylabel("Intensity (a.u.)")
    axd["spectra"].grid(True, linestyle=":", alpha=0.5)

    # Plot the bound fraction curve
    axd["fraction"].scatter(ratios * lmax_um, pb_list, c=ratios, cmap=cmap)
    axd["fraction"].set_xlabel("$L_{0}$ [μM]")
    axd["fraction"].set_ylabel("Bound fraction")
    axd["fraction"].set_ylim(-0.1, 1.1)
    axd["fraction"].grid(True)

    axd["legend"].legend(*axd["spectra"].get_legend_handles_labels(), loc="lower left")
