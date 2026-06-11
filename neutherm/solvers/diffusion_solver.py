"""
Two-group neutron diffusion solver in 1D cylindrical geometry.

Solves the eigenvalue problem:
    -1/r d/dr(r D_g dφ_g/dr) + Σ_{r,g} φ_g = (1/k_eff) χ_g Σ_f φ + S_g

using finite differences on a uniform radial mesh and power iteration
to find the dominant eigenvalue k_eff and corresponding flux shape.

The discretization uses the standard central difference scheme for the
cylindrical Laplacian, with special treatment at r=0 (L'Hôpital's rule)
and a reflective (zero-current, dφ/dr = 0) boundary at r = R_cell,
consistent with the Wigner-Seitz pin-cell approximation.

References
----------
.. [1] Duderstadt & Hamilton, "Nuclear Reactor Analysis" (1976), Ch. 5-7.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg as la

from neutherm.physics.cross_sections import CrossSectionSet


def build_diffusion_matrix(
    r: np.ndarray,
    D: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Build the finite-difference matrix for the diffusion operator.

    Discretizes: -1/r * d/dr(r * D * dφ/dr) + Σ * φ

    Supports non-uniform mesh spacing (needed for the composite
    fuel+moderator mesh where the two regions may have different dr).

    Parameters
    ----------
    r : np.ndarray
        Radial mesh points [cm], shape (N,). r[0] should be 0 or close to 0.
    D : np.ndarray
        Diffusion coefficient at each mesh point [cm], shape (N,).
    sigma : np.ndarray
        Removal/absorption cross section at each mesh point [cm^-1], shape (N,).

    Returns
    -------
    np.ndarray
        Tridiagonal matrix A of shape (N, N) such that A @ φ = source.
    """
    N = len(r)
    A = np.zeros((N, N))

    # --- Interior points (i = 1 to N-2) ---
    for i in range(1, N - 1):
        # Local mesh spacings (non-uniform allowed)
        dr_minus = r[i] - r[i - 1]  # spacing to the left
        dr_plus = r[i + 1] - r[i]  # spacing to the right
        dr_avg = (dr_minus + dr_plus) / 2  # average for the control volume

        # Half-point radii and diffusion coefficients
        r_plus = (r[i] + r[i + 1]) / 2
        r_minus = (r[i] + r[i - 1]) / 2
        D_plus = (D[i] + D[i + 1]) / 2
        D_minus = (D[i] + D[i - 1]) / 2

        # Conservative discretization coefficients
        coeff_minus = r_minus * D_minus / (r[i] * dr_minus * dr_avg)
        coeff_plus = r_plus * D_plus / (r[i] * dr_plus * dr_avg)
        coeff_center = coeff_minus + coeff_plus

        A[i, i - 1] = -coeff_minus
        A[i, i] = coeff_center + sigma[i]
        A[i, i + 1] = -coeff_plus

    # --- r = 0: L'Hôpital + symmetry ---
    dr0 = r[1] - r[0] if r[0] == 0 else r[1]
    A[0, 0] = 4.0 * D[0] / dr0**2 + sigma[0]
    A[0, 1] = -4.0 * D[0] / dr0**2

    # --- r = R_cell: reflective BC (zero net current, dφ/dr = 0) ---
    # In the Wigner-Seitz approximation, neighboring pin cells are
    # mirror images of each other, so the net neutron current at the
    # cell boundary is zero. This is a Neumann BC: dφ/dr(R) = 0.
    #
    # Using a ghost point approach: φ_{N} = φ_{N-2} (by symmetry),
    # the second derivative at N-1 becomes:
    #   d²φ/dr² ≈ (φ_{N-2} - 2φ_{N-1} + φ_{N}) / dr² = 2(φ_{N-2} - φ_{N-1}) / dr²
    # But the simpler approach: just use the interior stencil with
    # the right neighbor replaced by the reflection condition.
    i = N - 1
    dr_minus = r[i] - r[i - 1]
    r_minus = (r[i] + r[i - 1]) / 2
    D_minus = (D[i] + D[i - 1]) / 2

    # With dφ/dr = 0 at R, the flux through the right face is zero.
    # Only the left face contributes to leakage.
    coeff_minus = r_minus * D_minus / (r[i] * dr_minus**2)

    A[i, i - 1] = -coeff_minus
    A[i, i] = coeff_minus + sigma[i]  # No right leakage term

    return A


def power_iteration(
    A1: np.ndarray,
    A2: np.ndarray,
    fission_source_matrix: np.ndarray,
    scatter_source: np.ndarray,
    tol: float = 1e-7,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the two-group eigenvalue problem using power iteration.

    The two-group system in matrix form:
        A1 @ φ1 = (1/k) * F1 @ [φ1, φ2]    (fast group)
        A2 @ φ2 = S12 @ φ1                   (thermal group)

    Where:
        A1, A2 = diffusion + removal matrices for each group
        F1 = fission source (both groups contribute to fast fission neutrons)
        S12 = scattering source from group 1 to group 2

    Power iteration algorithm:
        1. Guess φ1, φ2, k
        2. Compute fission source: F = νΣ_f1 * φ1 + νΣ_f2 * φ2
        3. Solve A1 @ φ1_new = (1/k) * F  (with χ1 = 1, all fission → fast)
        4. Solve A2 @ φ2_new = Σ_{s12} * φ1_new
        5. Update k_new = k * (sum(F_new) / sum(F_old))
        6. Normalize fluxes
        7. Repeat until convergence

    Parameters
    ----------
    A1 : np.ndarray
        Diffusion+removal matrix for group 1, shape (N, N).
    A2 : np.ndarray
        Diffusion+absorption matrix for group 2, shape (N, N).
    fission_source_matrix : np.ndarray
        Matrix of shape (2, N) where row 0 = νΣ_f1 and row 1 = νΣ_f2.
    scatter_source : np.ndarray
        Scattering cross section Σ_{s,1→2} at each point, shape (N,).
    tol : float
        Convergence tolerance on k_eff (relative change).
    max_iter : int
        Maximum number of power iterations.

    Returns
    -------
    phi1 : np.ndarray
        Fast-group flux, shape (N,), normalized to max=1.
    phi2 : np.ndarray
        Thermal-group flux, shape (N,), normalized consistently.
    k_eff : float
        Effective multiplication factor.

    Raises
    ------
    RuntimeError
        If power iteration does not converge within max_iter.
    """
    N = A1.shape[0]

    # Initial guess: cosine-shaped flux (physically reasonable for
    # a cylindrical geometry — the fundamental mode is a Bessel J0)
    phi1 = np.cos(np.linspace(0, np.pi / 2, N))
    phi2 = phi1.copy()
    k_eff = 1.0  # Initial guess for eigenvalue

    # Extract fission cross sections for readability
    nu_sigma_f1 = fission_source_matrix[0]  # shape (N,)
    nu_sigma_f2 = fission_source_matrix[1]  # shape (N,)

    for iteration in range(max_iter):
        # Step 1: Compute total fission source
        # All fission neutrons are born in the fast group (χ1 = 1, χ2 = 0)
        fission_source = nu_sigma_f1 * phi1 + nu_sigma_f2 * phi2

        # Step 2: Solve for the fast-group flux
        # A1 @ φ1 = (1/k) * fission_source
        rhs1 = fission_source / k_eff

        # With reflective BC, the RHS at the boundary is just the
        # normal source term (no forcing to zero). The matrix A
        # already encodes dφ/dr = 0 at the outer boundary.
        phi1_new = la.solve(A1, rhs1)

        # Step 3: Solve for the thermal-group flux
        # A2 @ φ2 = Σ_{s,1→2} * φ1_new
        rhs2 = scatter_source * phi1_new
        phi2_new = la.solve(A2, rhs2)

        # Step 4: Compute new fission source and update k_eff
        fission_source_new = nu_sigma_f1 * phi1_new + nu_sigma_f2 * phi2_new

        # k_eff update: ratio of new to old total fission production
        # This is the standard power iteration eigenvalue update
        k_eff_new = k_eff * np.sum(fission_source_new) / np.sum(fission_source)

        # Step 5: Normalize fluxes (prevent overflow/underflow)
        # We normalize φ1 to max = 1; φ2 scales consistently
        norm = np.max(np.abs(phi1_new))
        if norm > 0:
            phi1_new /= norm
            phi2_new /= norm

        # Step 6: Check convergence
        k_change = abs(k_eff_new - k_eff) / abs(k_eff)

        # Update for next iteration
        phi1 = phi1_new
        phi2 = phi2_new
        k_eff = k_eff_new

        if k_change < tol:
            return phi1, phi2, k_eff

    raise RuntimeError(
        f"Power iteration did not converge after {max_iter} iterations. "
        f"Last k_eff change: {k_change:.2e} (tol: {tol:.2e})"
    )


def solve_diffusion(
    r: np.ndarray,
    xs: CrossSectionSet,
    tol: float = 1e-7,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the two-group neutron diffusion eigenvalue problem.

    This is the top-level interface for the diffusion solver. It builds
    the finite-difference matrices and runs power iteration.

    Parameters
    ----------
    r : np.ndarray
        Radial mesh points [cm], shape (N,). Must start at 0.
    xs : CrossSectionSet
        Cross sections at each mesh point (may be temperature-dependent).
    tol : float
        Convergence tolerance on k_eff.
    max_iter : int
        Maximum power iterations.

    Returns
    -------
    phi1 : np.ndarray
        Fast-group flux, normalized to max=1.
    phi2 : np.ndarray
        Thermal-group flux.
    k_eff : float
        Effective multiplication factor.
    """
    # Build the diffusion+removal matrices for each group
    # Group 1: -(1/r) d/dr(r D1 dφ1/dr) + Σ_{r,1} φ1 = (1/k) * fission
    A1 = build_diffusion_matrix(r, xs.D1, xs.sigma_r1)

    # Group 2: -(1/r) d/dr(r D2 dφ2/dr) + Σ_{a,2} φ2 = Σ_{s,1→2} φ1
    A2 = build_diffusion_matrix(r, xs.D2, xs.sigma_a2)

    # Fission source: νΣ_f1 * φ1 + νΣ_f2 * φ2 (both groups contribute)
    fission_matrix = np.array([xs.nu_sigma_f1, xs.nu_sigma_f2])

    # Run power iteration
    phi1, phi2, k_eff = power_iteration(
        A1, A2, fission_matrix, xs.sigma_s12, tol=tol, max_iter=max_iter
    )

    return phi1, phi2, k_eff