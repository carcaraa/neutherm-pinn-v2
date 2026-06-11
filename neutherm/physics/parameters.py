"""
Physical parameters and configuration for the coupled neutronics-thermal problem.

This module defines dataclasses that hold all physical constants, geometry,
cross-section data, and solver settings. Parameters are loaded from a YAML
configuration file (see configs/default.yaml).

References
----------
.. [1] Duderstadt & Hamilton, "Nuclear Reactor Analysis" (1976), Ch. 5-7.
.. [3] Todreas & Kazimi, "Nuclear Systems I" (2012), Ch. 8.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class GeometryParams:
    """Fuel pin cell geometry (1D radial, cylindrical).

    The pin cell model includes fuel, cladding, and the equivalent
    annular moderator region out to the Wigner-Seitz cell radius.
    For a square lattice with pitch p, the equivalent radius is:
        r_cell = p / sqrt(π)
    """

    r_fuel: float = 0.4096e-2  # Fuel pellet radius [m]
    r_clad: float = 0.4750e-2  # Cladding outer radius [m]
    r_cell: float = 0.7174e-2  # Pin cell equivalent radius [m] (pitch ~1.27 cm)
    n_radial: int = 100  # Number of radial mesh points in fuel
    n_radial_mod: int = 40  # Number of radial mesh points in moderator


@dataclass
class NeutronicsParams:
    """Two-group neutron diffusion cross sections at reference temperature.

    The cross sections follow the Doppler broadening model:
        Σ(T) = Σ(T_ref) * (1 + α * sqrt(T - T_ref))

    where α is the temperature coefficient for each reaction type.

    All cross sections in [cm^-1], diffusion coefficients in [cm].
    """

    T_ref: float = 300.0  # Reference temperature [K]

    # Group 1 (fast)
    D1_ref: float = 1.255  # Diffusion coefficient [cm]
    sigma_r1_ref: float = 0.0265  # Removal cross section [cm^-1]
    nu_sigma_f1_ref: float = 0.0081  # nu*Sigma_f [cm^-1]

    # Group 2 (thermal)
    D2_ref: float = 0.211  # Diffusion coefficient [cm]
    sigma_a2_ref: float = 0.0750  # Absorption cross section [cm^-1]
    nu_sigma_f2_ref: float = 0.135  # nu*Sigma_f [cm^-1]
    sigma_s12_ref: float = 0.0177  # Down-scattering 1→2 [cm^-1]

    # Doppler temperature coefficients
    alpha_a2: float = -2.0e-4  # Absorption group 2
    alpha_f2: float = -1.8e-4  # Fission group 2
    alpha_f1: float = -0.5e-4  # Fission group 1

    # Moderator (water) cross sections — temperature-independent
    # These are homogenized values for light water at ~580 K
    D1_mod: float = 1.13  # Fast diffusion coefficient [cm]
    D2_mod: float = 0.16  # Thermal diffusion coefficient [cm]
    sigma_r1_mod: float = 0.0494  # Fast removal [cm^-1]
    sigma_a2_mod: float = 0.0197  # Thermal absorption [cm^-1]
    sigma_s12_mod: float = 0.0487  # Down-scattering 1→2 [cm^-1]


@dataclass
class ThermalParams:
    """Thermal-hydraulic parameters for the fuel pin.

    Fuel conductivity model (UO2):
        k_f(T) = 1 / (A + B*T) + C * T^3

    with empirical constants A, B, C from MATPRO correlations.

    References: Todreas & Kazimi (2012), Ch. 8.
    """

    kappa_fission: float = 3.204e-11  # Energy per fission [J]

    # UO2 thermal conductivity: k(T) = 1/(A + B*T) + C*T^3
    fuel_k_A: float = 0.0375  # [m·K/W]
    fuel_k_B: float = 2.165e-4  # [m/W]
    fuel_k_C: float = 4.715e-12  # [W/(m·K^4)]

    # Boundary conditions
    T_coolant: float = 580.0  # Bulk coolant temperature [K]
    h_gap: float = 5000.0  # Gap conductance [W/(m²·K)]
    h_conv: float = 30000.0  # Coolant HTC [W/(m²·K)]


@dataclass
class SolverParams:
    """Settings for the iterative numerical solver."""

    max_picard_iter: int = 200
    tol_temperature: float = 1.0e-5
    tol_keff: float = 1.0e-6
    power_iteration_tol: float = 1.0e-7
    power_iteration_max: int = 500


@dataclass
class DatasetParams:
    """Settings for training data generation."""

    n_samples: int = 5000
    sampling: str = "latin_hypercube"
    vary_T_coolant: tuple[float, float] = (550.0, 620.0)
    vary_r_fuel: tuple[float, float] = (0.38e-2, 0.44e-2)
    vary_enrichment_factor: tuple[float, float] = (0.8, 1.2)


@dataclass
class SurrogateParams:
    """Settings for the data-driven surrogate model."""

    architecture: str = "fnn"
    hidden_layers: list[int] = field(default_factory=lambda: [128, 128, 128, 128])
    activation: str = "tanh"
    learning_rate: float = 1.0e-3
    batch_size: int = 256
    epochs: int = 5000
    scheduler: str = "cosine"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class PINNParams:
    """Settings for the physics-informed neural network."""

    hidden_layers: list[int] = field(default_factory=lambda: [64, 64, 64, 64])
    activation: str = "tanh"
    learning_rate: float = 1.0e-3
    n_collocation: int = 2000
    n_boundary: int = 50
    epochs: int = 20000
    lambda_pde: float = 1.0
    lambda_bc: float = 10.0
    lambda_data: float = 0.1
    lambda_power: float = 10.0  # Weight of the power-normalization constraint
    adaptive_weights: bool = False  # NOTE: not implemented yet (roadmap)


@dataclass
class ProblemConfig:
    """Top-level configuration aggregating all parameter groups.

    Use `ProblemConfig.from_yaml(path)` to load from a YAML config file.
    """

    geometry: GeometryParams = field(default_factory=GeometryParams)
    neutronics: NeutronicsParams = field(default_factory=NeutronicsParams)
    thermal: ThermalParams = field(default_factory=ThermalParams)
    solver: SolverParams = field(default_factory=SolverParams)
    dataset: DatasetParams = field(default_factory=DatasetParams)
    surrogate: SurrogateParams = field(default_factory=SurrogateParams)
    pinn: PINNParams = field(default_factory=PINNParams)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProblemConfig":
        """Load configuration from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to the YAML configuration file.

        Returns
        -------
        ProblemConfig
            Fully populated configuration object.
        """
        path = Path(path)
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        config = cls()

        if "geometry" in raw:
            config.geometry = GeometryParams(**raw["geometry"])

        if "physics" in raw:
            p = raw["physics"]
            # Split physics section into neutronics and thermal
            neutronics_keys = {f.name for f in NeutronicsParams.__dataclass_fields__.values()}
            thermal_keys = {f.name for f in ThermalParams.__dataclass_fields__.values()}

            neutronics_dict = {k: v for k, v in p.items() if k in neutronics_keys}
            thermal_dict = {k: v for k, v in p.items() if k in thermal_keys}

            if neutronics_dict:
                config.neutronics = NeutronicsParams(**neutronics_dict)
            if thermal_dict:
                config.thermal = ThermalParams(**thermal_dict)

        if "solver" in raw:
            config.solver = SolverParams(**raw["solver"])

        if "dataset" in raw:
            d = raw["dataset"]
            # Convert list to tuple for range fields
            for key in ("vary_T_coolant", "vary_r_fuel", "vary_enrichment_factor"):
                if key in d and isinstance(d[key], list):
                    d[key] = tuple(d[key])
            config.dataset = DatasetParams(**d)

        if "surrogate" in raw:
            config.surrogate = SurrogateParams(**raw["surrogate"])

        if "pinn" in raw:
            config.pinn = PINNParams(**raw["pinn"])

        return config

    def validate(self) -> None:
        """Run basic sanity checks on the configuration.

        Raises
        ------
        ValueError
            If any parameter is physically unreasonable.
        """
        g = self.geometry
        if g.r_fuel <= 0 or g.r_clad <= g.r_fuel:
            raise ValueError(
                f"Invalid geometry: r_fuel={g.r_fuel}, r_clad={g.r_clad}. "
                "Need 0 < r_fuel < r_clad."
            )
        if g.r_cell <= g.r_clad:
            raise ValueError(
                f"Invalid geometry: r_cell={g.r_cell} must be > r_clad={g.r_clad}."
            )
        if g.n_radial < 10:
            raise ValueError(f"n_radial={g.n_radial} too coarse. Use at least 10.")

        n = self.neutronics
        if n.T_ref <= 0:
            raise ValueError(f"T_ref must be positive, got {n.T_ref}")
        for name in ("D1_ref", "D2_ref", "sigma_r1_ref", "sigma_a2_ref"):
            val = getattr(n, name)
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")

        t = self.thermal
        if t.T_coolant <= 0:
            raise ValueError(f"T_coolant must be positive, got {t.T_coolant}")
        if t.h_gap <= 0 or t.h_conv <= 0:
            raise ValueError("Heat transfer coefficients must be positive.")