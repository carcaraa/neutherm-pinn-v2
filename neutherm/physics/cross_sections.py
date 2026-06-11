"""
Temperature-dependent macroscopic cross sections with Doppler feedback.

This module provides functions to compute neutron cross sections as a function
of temperature, using the square-root Doppler broadening model:

    Σ(T) = Σ(T_ref) * (1 + α * sqrt(T - T_ref))

This is a simplified but widely used parametrization that captures the dominant
effect of thermal motion on resonance absorption (Doppler broadening of the
238U resonances in the epithermal range).

Both NumPy and PyTorch implementations are provided:
- NumPy versions are used in the reference finite-difference solver.
- PyTorch versions are used in the PINN, enabling automatic differentiation
  through the cross-section model.

References
----------
.. [1] Duderstadt & Hamilton, "Nuclear Reactor Analysis" (1976), §4.3.
.. [2] Stacey, "Nuclear Reactor Physics" (2007), §6.2.

Notes
-----
The temperature coefficients α are typically negative for absorption and
fission cross sections in thermal reactors (negative Doppler coefficient),
which is the primary safety feedback mechanism in LWRs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from neutherm.physics.parameters import NeutronicsParams

# -----------------------------------------------------------------------
# TYPE_CHECKING is True only when a type checker (mypy, pyright) runs,
# NOT at runtime. This lets us use 'torch.Tensor' in type hints without
# importing torch at module load time. That way, users who only need the
# NumPy solver don't need torch installed at all.
# -----------------------------------------------------------------------
if TYPE_CHECKING:
    import torch


# =============================================================================
# Data container for NumPy cross-section results
# =============================================================================


@dataclass
class CrossSectionSet:
    """A complete set of two-group cross sections at a given temperature.

    This is a plain container returned by the NumPy evaluation functions.
    All values in [cm^-1] except diffusion coefficients in [cm].

    Attributes
    ----------
    D1 : np.ndarray
        Fast-group diffusion coefficient [cm].
    D2 : np.ndarray
        Thermal-group diffusion coefficient [cm].
    sigma_r1 : np.ndarray
        Fast-group removal cross section [cm^-1].
        (removal = absorption + down-scattering out of group 1)
    sigma_a2 : np.ndarray
        Thermal-group absorption cross section [cm^-1].
    nu_sigma_f1 : np.ndarray
        Fast-group fission neutron production cross section [cm^-1].
        (ν times the fission cross section)
    nu_sigma_f2 : np.ndarray
        Thermal-group fission neutron production cross section [cm^-1].
    sigma_s12 : np.ndarray
        Down-scattering cross section from group 1 → group 2 [cm^-1].
    """

    D1: np.ndarray
    D2: np.ndarray
    sigma_r1: np.ndarray
    sigma_a2: np.ndarray
    nu_sigma_f1: np.ndarray
    nu_sigma_f2: np.ndarray
    sigma_s12: np.ndarray


# =============================================================================
# Doppler correction factors
# =============================================================================
#
# The Doppler effect in nuclear reactors: as temperature rises, the thermal
# motion of fuel nuclei increases, effectively broadening the resonance peaks
# in the neutron cross sections. For 238U (the dominant resonance absorber
# in UO2 fuel), this broadening INCREASES the probability that a neutron is
# captured at energies near a resonance, because the broadened peak "covers"
# a wider energy range.
#
# However, the INTEGRAL of the resonance (area under the peak) is conserved
# (it depends on nuclear structure, not temperature). The net macroscopic
# effect — captured in the α coefficients — is typically:
#   - α < 0 for absorption and fission → cross sections DECREASE with T
#     (this is a simplification; the detailed behavior depends on the
#      competition between fuel temperature and moderator temperature effects)
#   - This gives NEGATIVE temperature reactivity feedback, which is the
#     fundamental passive safety mechanism in LWRs.
#
# The sqrt(T - T_ref) model is a first-order approximation derived from
# the Doppler broadening theory (see Duderstadt & Hamilton §4.3).
# =============================================================================


def _doppler_factor_np(T: np.ndarray, T_ref: float, alpha: float) -> np.ndarray:
    """Compute the Doppler correction factor using NumPy.

    The correction factor multiplies the reference cross section:
        Σ(T) = Σ_ref * factor(T)
        factor(T) = 1 + α * sqrt(T - T_ref)

    Parameters
    ----------
    T : np.ndarray
        Temperature field [K]. Can be scalar or array.
    T_ref : float
        Reference temperature [K] at which cross sections are tabulated.
    alpha : float
        Temperature coefficient [K^{-1/2}]. Typically negative for
        absorption/fission in thermal reactors.

    Returns
    -------
    np.ndarray
        Multiplicative correction factor, same shape as T.
    """
    # Guard against T < T_ref (can happen during early solver iterations
    # or if coolant temperature is below reference). We clamp to zero
    # rather than taking sqrt of a negative number.
    dT = np.maximum(T - T_ref, 0.0)
    return 1.0 + alpha * np.sqrt(dT)


def _doppler_factor_torch(
    T: "torch.Tensor", T_ref: float, alpha: float
) -> "torch.Tensor":
    """Compute the Doppler correction factor using PyTorch (differentiable).

    This is the same formula as the NumPy version, but uses PyTorch ops
    so that gradients can flow through it during PINN training. The key
    difference is the small epsilon (1e-12) added inside the sqrt to
    avoid infinite gradients at T = T_ref (where dT = 0).

    Why epsilon matters:
        d/dT sqrt(T - T_ref) = 1 / (2 * sqrt(T - T_ref))
        At T = T_ref, this is 1/0 = infinity → NaN in backprop.
        Adding epsilon: d/dT sqrt(T - T_ref + ε) ≈ 1/(2*sqrt(ε))
        which is large but finite.

    Parameters
    ----------
    T : torch.Tensor
        Temperature field [K]. Must have requires_grad=True for PINN usage.
    T_ref : float
        Reference temperature [K].
    alpha : float
        Temperature coefficient [K^{-1/2}].

    Returns
    -------
    torch.Tensor
        Multiplicative correction factor.
    """
    # Lazy import: torch is only loaded when this function is actually called.
    # This allows the rest of the package to work without torch installed.
    import torch

    dT = torch.clamp(T - T_ref, min=0.0)
    return 1.0 + alpha * torch.sqrt(dT + 1e-12)


# =============================================================================
# Full cross-section evaluation — NumPy version (for the FD solver)
# =============================================================================


def evaluate_cross_sections_np(
    T: np.ndarray,
    params: NeutronicsParams,
) -> CrossSectionSet:
    """Evaluate all two-group cross sections at given temperatures (NumPy).

    Which cross sections depend on temperature?
    --------------------------------------------
    - D1, D2 (diffusion coefficients): kept CONSTANT.
      Rationale: D = 1/(3*Σ_tr), where Σ_tr is the transport cross section,
      dominated by elastic scattering on hydrogen. Elastic scattering has
      very weak temperature dependence (no resonances in hydrogen).

    - Σ_{r,1} (fast removal): kept CONSTANT.
      Rationale: Σ_r1 = Σ_a1 + Σ_{s,1→2}. Fast absorption (Σ_a1) is small,
      and down-scattering is dominated by hydrogen moderation.

    - Σ_{s,1→2} (down-scattering): kept CONSTANT.
      Same rationale as above.

    - Σ_{a,2} (thermal absorption): TEMPERATURE-DEPENDENT.
      This is the main Doppler effect — 238U resonance absorption.

    - νΣ_{f,1}, νΣ_{f,2} (fission production): TEMPERATURE-DEPENDENT.
      Fission cross sections also have resonance structure, though the
      effect is smaller than for absorption.

    Parameters
    ----------
    T : np.ndarray
        Temperature field [K], shape (N,) or scalar.
    params : NeutronicsParams
        Reference cross-section values and temperature coefficients.

    Returns
    -------
    CrossSectionSet
        All cross sections evaluated at the given temperatures.
    """
    # Ensure T is at least 1D for consistent array operations
    T = np.atleast_1d(np.asarray(T, dtype=np.float64))

    # --- Temperature-INDEPENDENT quantities ---
    # These are broadcast to match the shape of T (constant across the pin)
    D1 = np.full_like(T, params.D1_ref)
    D2 = np.full_like(T, params.D2_ref)
    sigma_r1 = np.full_like(T, params.sigma_r1_ref)
    sigma_s12 = np.full_like(T, params.sigma_s12_ref)

    # --- Temperature-DEPENDENT quantities (Doppler model) ---
    # Each gets multiplied by its own correction factor with its own α
    sigma_a2 = params.sigma_a2_ref * _doppler_factor_np(
        T, params.T_ref, params.alpha_a2
    )
    nu_sigma_f1 = params.nu_sigma_f1_ref * _doppler_factor_np(
        T, params.T_ref, params.alpha_f1
    )
    nu_sigma_f2 = params.nu_sigma_f2_ref * _doppler_factor_np(
        T, params.T_ref, params.alpha_f2
    )

    return CrossSectionSet(
        D1=D1,
        D2=D2,
        sigma_r1=sigma_r1,
        sigma_a2=sigma_a2,
        nu_sigma_f1=nu_sigma_f1,
        nu_sigma_f2=nu_sigma_f2,
        sigma_s12=sigma_s12,
    )


# =============================================================================
# Full cross-section evaluation — PyTorch version (for the PINN)
# =============================================================================


def evaluate_cross_sections_torch(
    T: "torch.Tensor",
    params: NeutronicsParams,
) -> dict[str, "torch.Tensor"]:
    """Evaluate all two-group cross sections at given temperatures (PyTorch).

    This version preserves the computational graph for automatic differentiation,
    which is essential for computing PDE residuals in the PINN.

    The PINN loss function needs to compute terms like:
        d/dr [D(T) * dφ/dr]
    Since D could depend on T (it doesn't in our model, but Σ_a2 does),
    and T depends on φ through the heat equation, the entire chain must
    be differentiable. PyTorch's autograd handles this automatically as
    long as we use torch operations throughout.

    Returns a dict instead of a dataclass for easier unpacking in the
    PINN residual computation.

    Parameters
    ----------
    T : torch.Tensor
        Temperature field [K].
    params : NeutronicsParams
        Reference cross-section values and temperature coefficients.

    Returns
    -------
    dict[str, torch.Tensor]
        Keys: 'D1', 'D2', 'sigma_r1', 'sigma_a2',
              'nu_sigma_f1', 'nu_sigma_f2', 'sigma_s12'.
    """
    # Lazy import — only needed when actually using PyTorch
    import torch

    # Temperature-independent: these create new tensors that DON'T
    # participate in the gradient graph (they're constants).
    D1 = torch.full_like(T, params.D1_ref)
    D2 = torch.full_like(T, params.D2_ref)
    sigma_r1 = torch.full_like(T, params.sigma_r1_ref)
    sigma_s12 = torch.full_like(T, params.sigma_s12_ref)

    # Temperature-dependent: these DO participate in the gradient graph
    # because _doppler_factor_torch uses differentiable torch operations.
    # When we call loss.backward(), gradients will flow through these
    # back to T, and then back to the network weights.
    sigma_a2 = params.sigma_a2_ref * _doppler_factor_torch(
        T, params.T_ref, params.alpha_a2
    )
    nu_sigma_f1 = params.nu_sigma_f1_ref * _doppler_factor_torch(
        T, params.T_ref, params.alpha_f1
    )
    nu_sigma_f2 = params.nu_sigma_f2_ref * _doppler_factor_torch(
        T, params.T_ref, params.alpha_f2
    )

    return {
        "D1": D1,
        "D2": D2,
        "sigma_r1": sigma_r1,
        "sigma_a2": sigma_a2,
        "nu_sigma_f1": nu_sigma_f1,
        "nu_sigma_f2": nu_sigma_f2,
        "sigma_s12": sigma_s12,
    }


# =============================================================================
# Pin cell cross-section assembly (fuel + moderator)
# =============================================================================


def build_pin_cell_xs_np(
    r: np.ndarray,
    T_fuel: np.ndarray,
    r_fuel_cm: float,
    params: NeutronicsParams,
) -> CrossSectionSet:
    """Build cross sections for the full pin cell (fuel + moderator).

    The pin cell has two distinct regions:
    - r <= R_fuel: UO2 fuel with temperature-dependent cross sections
    - r > R_fuel:  Water moderator with fixed cross sections (no fission)

    The transition is sharp (no cladding modeled neutronically — the
    cladding is thin and its effect on neutron transport is small
    compared to the fuel and moderator).

    Parameters
    ----------
    r : np.ndarray
        Full pin cell radial mesh [cm], from 0 to R_cell.
    T_fuel : np.ndarray
        Temperature in the fuel region [K]. Must have the same length
        as the number of points where r <= r_fuel_cm.
    r_fuel_cm : float
        Fuel radius [cm] — boundary between fuel and moderator.
    params : NeutronicsParams
        Cross-section data including moderator values.

    Returns
    -------
    CrossSectionSet
        Cross sections on the full pin cell mesh.
    """
    N = len(r)

    # Identify fuel and moderator regions
    fuel_mask = r <= r_fuel_cm + 1e-10  # Small tolerance for floating point
    n_fuel = np.sum(fuel_mask)

    # Initialize arrays for the full mesh
    D1 = np.zeros(N)
    D2 = np.zeros(N)
    sigma_r1 = np.zeros(N)
    sigma_a2 = np.zeros(N)
    nu_sigma_f1 = np.zeros(N)
    nu_sigma_f2 = np.zeros(N)
    sigma_s12 = np.zeros(N)

    # --- Fuel region: temperature-dependent ---
    if len(T_fuel) < n_fuel:
        raise ValueError(
            f"T_fuel has {len(T_fuel)} points but the fuel region of the "
            f"mesh has {n_fuel} points (r <= {r_fuel_cm} cm). The fuel "
            "temperature must be defined on every fuel mesh point."
        )
    T_on_mesh = T_fuel[:n_fuel]
    fuel_xs = evaluate_cross_sections_np(T_on_mesh, params)

    D1[fuel_mask] = fuel_xs.D1
    D2[fuel_mask] = fuel_xs.D2
    sigma_r1[fuel_mask] = fuel_xs.sigma_r1
    sigma_a2[fuel_mask] = fuel_xs.sigma_a2
    nu_sigma_f1[fuel_mask] = fuel_xs.nu_sigma_f1
    nu_sigma_f2[fuel_mask] = fuel_xs.nu_sigma_f2
    sigma_s12[fuel_mask] = fuel_xs.sigma_s12

    # --- Moderator region: fixed cross sections, no fission ---
    mod_mask = ~fuel_mask
    D1[mod_mask] = params.D1_mod
    D2[mod_mask] = params.D2_mod
    sigma_r1[mod_mask] = params.sigma_r1_mod
    sigma_a2[mod_mask] = params.sigma_a2_mod
    nu_sigma_f1[mod_mask] = 0.0  # No fission in moderator
    nu_sigma_f2[mod_mask] = 0.0
    sigma_s12[mod_mask] = params.sigma_s12_mod

    return CrossSectionSet(
        D1=D1,
        D2=D2,
        sigma_r1=sigma_r1,
        sigma_a2=sigma_a2,
        nu_sigma_f1=nu_sigma_f1,
        nu_sigma_f2=nu_sigma_f2,
        sigma_s12=sigma_s12,
    )