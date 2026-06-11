"""
Training dataset generation via parametric sweeps of the coupled solver.

This module generates labeled data for the surrogate model by:
1. Sampling the input parameter space using Latin Hypercube Sampling (LHS)
2. Running the coupled neutronics-thermal solver for each sample
3. Storing the results (flux profiles, temperature, k_eff) in a single file

Latin Hypercube Sampling ensures that the parameter space is covered
uniformly in each dimension, which is more efficient than random sampling
for training surrogate models (fewer samples needed for the same coverage).

The varied parameters are:
- T_coolant: bulk coolant temperature [K]
- r_fuel: fuel pellet radius [m]
- enrichment_factor: multiplier on νΣ_f (proxy for enrichment level)

Each parameter is sampled within a range defined in the config YAML.

Usage
-----
From the command line:
    python -m neutherm.training.dataset --config configs/default.yaml --n-samples 100

From Python:
    from neutherm.training.dataset import generate_dataset
    dataset = generate_dataset(config, n_samples=100)

References
----------
.. [8] McKay, Beckman & Conover (1979). "A comparison of three methods for
       selecting values of input variables in the analysis of output from
       a computer code." Technometrics, 21(2), 239-245.
       (Original LHS paper)
"""

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import qmc  # quasi-Monte Carlo — contains LHS

from neutherm.physics.parameters import ProblemConfig
from neutherm.solvers.coupled_solver import solve_coupled


@dataclass
class ParametricDataset:
    """Container for the generated parametric dataset.

    The dataset consists of N samples. Each sample has:
    - Input parameters (scalars): T_coolant, r_fuel, enrichment_factor
    - Output fields (1D arrays on radial mesh): phi1, phi2, temperature, q_vol
    - Output scalar: k_eff

    All samples share the same *number* of mesh points (set by n_radial /
    n_radial_mod), which is what allows stacking into tensors. The physical
    mesh coordinates differ between samples when r_fuel varies — fields are
    NOT interpolated onto a common physical mesh. The stored r_fuel_mesh /
    r_full_mesh come from the first successful sample and serve only as a
    representative axis for plotting.

    Attributes
    ----------
    params : np.ndarray
        Input parameter matrix, shape (N, 3).
        Columns: [T_coolant, r_fuel, enrichment_factor]
    r_fuel_mesh : np.ndarray
        Common radial mesh for fuel-region fields [cm], shape (n_radial,).
    r_full_mesh : np.ndarray
        Common radial mesh for full pin cell [cm], shape (n_total,).
    phi1 : np.ndarray
        Fast-group flux on the full mesh, shape (N, n_total).
    phi2 : np.ndarray
        Thermal-group flux on the full mesh, shape (N, n_total).
    temperature : np.ndarray
        Fuel temperature profiles, shape (N, n_radial).
    q_volumetric : np.ndarray
        Volumetric heat generation in fuel, shape (N, n_radial).
    k_eff : np.ndarray
        Effective multiplication factors, shape (N,).
    converged : np.ndarray
        Boolean convergence flags, shape (N,).
    param_names : list[str]
        Names of the input parameters, for reference.
    """

    params: np.ndarray
    r_fuel_mesh: np.ndarray
    r_full_mesh: np.ndarray
    phi1: np.ndarray
    phi2: np.ndarray
    temperature: np.ndarray
    q_volumetric: np.ndarray
    k_eff: np.ndarray
    converged: np.ndarray
    param_names: list[str] = field(
        default_factory=lambda: ["T_coolant", "r_fuel", "enrichment_factor"]
    )


def latin_hypercube_sample(
    n_samples: int,
    bounds: list[tuple[float, float]],
    seed: int = 42,
) -> np.ndarray:
    """Generate samples using Latin Hypercube Sampling.

    LHS divides each parameter dimension into n_samples equal intervals,
    then places exactly one sample in each interval per dimension. This
    ensures uniform marginal coverage — no "clumps" or "gaps" in any
    single dimension.

    Compared to pure random sampling:
    - LHS needs ~3-5x fewer samples for the same coverage
    - Especially important when each sample is expensive (running a solver)

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    bounds : list of (low, high) tuples
        Parameter bounds for each dimension.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sample matrix of shape (n_samples, n_dimensions).
        Each column is scaled to its corresponding [low, high] range.
    """
    n_dims = len(bounds)

    # scipy.stats.qmc.LatinHypercube generates samples in [0, 1]^d
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    unit_samples = sampler.random(n=n_samples)  # shape (n_samples, n_dims)

    # Scale from [0, 1] to the actual parameter bounds
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    scaled_samples = qmc.scale(unit_samples, lower, upper)

    return scaled_samples


def generate_dataset(
    config: ProblemConfig,
    n_samples: int | None = None,
    power_level: float = 200.0,
    seed: int = 42,
    verbose: bool = True,
) -> ParametricDataset:
    """Generate the training dataset by running parametric sweeps.

    For each LHS sample, this function:
    1. Modifies the config with the sampled parameters
    2. Runs the coupled solver
    3. Stores the solution fields

    The three varied parameters are:
    - T_coolant: affects the thermal boundary condition and cross sections
    - r_fuel: changes the geometry (fuel volume, surface-to-volume ratio)
    - enrichment_factor: multiplies νΣ_f in both groups (proxy for U-235 content)

    Parameters
    ----------
    config : ProblemConfig
        Base configuration. Sampled parameters override the relevant fields.
    n_samples : int, optional
        Number of samples. Defaults to config.dataset.n_samples.
    power_level : float
        Target linear heat rate [W/cm] for each sample.
    seed : int
        Random seed for LHS reproducibility.
    verbose : bool
        Print progress updates.

    Returns
    -------
    ParametricDataset
        Complete dataset ready for training.
    """
    if n_samples is None:
        n_samples = config.dataset.n_samples

    ds = config.dataset

    # Define the parameter space bounds
    # Each tuple is (lower_bound, upper_bound)
    bounds = [
        ds.vary_T_coolant,           # T_coolant [K]
        ds.vary_r_fuel,              # r_fuel [m]
        ds.vary_enrichment_factor,   # enrichment multiplier (dimensionless)
    ]

    # Generate LHS samples
    if verbose:
        print(f"Generating {n_samples} Latin Hypercube samples...")
        print(f"  T_coolant:    [{bounds[0][0]:.0f}, {bounds[0][1]:.0f}] K")
        print(f"  r_fuel:       [{bounds[1][0]*100:.3f}, {bounds[1][1]*100:.3f}] cm")
        print(f"  enrichment:   [{bounds[2][0]:.2f}, {bounds[2][1]:.2f}] x reference")

    param_samples = latin_hypercube_sample(n_samples, bounds, seed=seed)

    # Pre-allocate output arrays
    # We use the mesh sizes from the base config for consistency.
    # If r_fuel varies, the mesh is rebuilt for each sample, but the
    # number of points stays the same (set by n_radial, n_radial_mod).
    n_radial = config.geometry.n_radial
    n_mod = config.geometry.n_radial_mod
    n_total = n_radial + n_mod

    phi1_all = np.zeros((n_samples, n_total))
    phi2_all = np.zeros((n_samples, n_total))
    temperature_all = np.zeros((n_samples, n_radial))
    q_vol_all = np.zeros((n_samples, n_radial))
    k_eff_all = np.zeros(n_samples)
    converged_all = np.zeros(n_samples, dtype=bool)

    # Reference meshes (from the first sample — will be overwritten
    # but all have the same number of points)
    r_fuel_mesh = None
    r_full_mesh = None

    # Track timing and failures
    t_start = time.time()
    n_failed = 0

    for i in range(n_samples):
        # Extract sampled parameters for this run
        T_coolant_i = param_samples[i, 0]
        r_fuel_i = param_samples[i, 1]
        enrichment_i = param_samples[i, 2]

        # Create a modified config for this sample
        # We use deepcopy to avoid mutating the base config
        cfg_i = copy.deepcopy(config)

        # Apply the sampled parameters
        cfg_i.thermal.T_coolant = T_coolant_i
        cfg_i.geometry.r_fuel = r_fuel_i

        # Scale the fission cross sections by the enrichment factor
        # This is a simplified proxy: increasing enrichment increases
        # the concentration of U-235, which directly increases νΣ_f
        cfg_i.neutronics.nu_sigma_f1_ref *= enrichment_i
        cfg_i.neutronics.nu_sigma_f2_ref *= enrichment_i

        # Run the coupled solver (quietly — no per-sample output)
        try:
            solution = solve_coupled(cfg_i, power_level=power_level, verbose=False)

            phi1_all[i, :] = solution.phi1
            phi2_all[i, :] = solution.phi2
            temperature_all[i, :] = solution.temperature
            q_vol_all[i, :] = solution.q_volumetric
            k_eff_all[i] = solution.k_eff
            converged_all[i] = solution.converged

            # Store the reference meshes from the first successful run
            if r_fuel_mesh is None:
                r_fuel_mesh = solution.r_fuel
                r_full_mesh = solution.r_neutronics

        except Exception as e:
            # If the solver fails (e.g., divergence for extreme parameters),
            # mark as unconverged and continue. The training pipeline will
            # filter these out.
            converged_all[i] = False
            n_failed += 1
            if verbose:
                print(f"  [WARN] Sample {i+1}/{n_samples} failed: {e}")

        # Progress update every 10% or every 50 samples
        if verbose and ((i + 1) % max(1, n_samples // 10) == 0 or (i + 1) == n_samples):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:>{len(str(n_samples))}}/{n_samples}] "
                  f"k_eff={k_eff_all[i]:.4f}  "
                  f"elapsed={elapsed:.1f}s  "
                  f"ETA={eta:.1f}s  "
                  f"({n_failed} failed)")

    # Summary
    n_good = np.sum(converged_all)
    if verbose:
        total_time = time.time() - t_start
        print(f"\nDataset generation complete!")
        print(f"  Total samples:     {n_samples}")
        print(f"  Converged:         {n_good} ({100*n_good/n_samples:.1f}%)")
        print(f"  Failed:            {n_failed}")
        print(f"  Total time:        {total_time:.1f}s")
        print(f"  Time per sample:   {total_time/n_samples:.2f}s")
        print(f"  k_eff range:       [{k_eff_all[converged_all].min():.4f}, "
              f"{k_eff_all[converged_all].max():.4f}]")

    return ParametricDataset(
        params=param_samples,
        r_fuel_mesh=r_fuel_mesh,
        r_full_mesh=r_full_mesh,
        phi1=phi1_all,
        phi2=phi2_all,
        temperature=temperature_all,
        q_volumetric=q_vol_all,
        k_eff=k_eff_all,
        converged=converged_all,
    )


def save_dataset(dataset: ParametricDataset, path: str | Path) -> None:
    """Save the dataset to disk as a compressed NumPy archive (.npz).

    We use .npz (NumPy) instead of .pt (PyTorch) so that the dataset
    can be loaded without torch installed. The training scripts will
    convert to torch tensors when needed.

    Parameters
    ----------
    dataset : ParametricDataset
        The dataset to save.
    path : str or Path
        Output file path. Should end in .npz.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        params=dataset.params,
        r_fuel_mesh=dataset.r_fuel_mesh,
        r_full_mesh=dataset.r_full_mesh,
        phi1=dataset.phi1,
        phi2=dataset.phi2,
        temperature=dataset.temperature,
        q_volumetric=dataset.q_volumetric,
        k_eff=dataset.k_eff,
        converged=dataset.converged,
        param_names=dataset.param_names,
    )

    # Report file size
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Dataset saved to {path} ({size_mb:.1f} MB)")


def load_dataset(path: str | Path) -> ParametricDataset:
    """Load a dataset from a compressed NumPy archive (.npz).

    Parameters
    ----------
    path : str or Path
        Path to the .npz file.

    Returns
    -------
    ParametricDataset
        The loaded dataset.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    return ParametricDataset(
        params=data["params"],
        r_fuel_mesh=data["r_fuel_mesh"],
        r_full_mesh=data["r_full_mesh"],
        phi1=data["phi1"],
        phi2=data["phi2"],
        temperature=data["temperature"],
        q_volumetric=data["q_volumetric"],
        k_eff=data["k_eff"],
        converged=data["converged"],
        param_names=list(data["param_names"]),
    )


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate training dataset from parametric solver sweeps."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Number of samples (overrides config if provided).",
    )
    parser.add_argument(
        "--power", type=float, default=200.0,
        help="Target linear heat rate [W/cm].",
    )
    parser.add_argument(
        "--output", type=str, default="data/dataset.npz",
        help="Output file path.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for LHS.",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = ProblemConfig.from_yaml(args.config)
    cfg.validate()

    # Generate dataset
    dataset = generate_dataset(
        cfg,
        n_samples=args.n_samples,
        power_level=args.power,
        seed=args.seed,
    )

    # Save to disk
    save_dataset(dataset, args.output)
