"""
Physical models for the coupled neutronics-thermal problem.

Submodules
----------
parameters : Configuration dataclasses loaded from YAML.
cross_sections : Temperature-dependent macroscopic cross sections.
fuel_properties : UO2 thermal conductivity and heat generation.
"""

from neutherm.physics.cross_sections import (
    CrossSectionSet,
    build_pin_cell_xs_np,
    evaluate_cross_sections_np,
    evaluate_cross_sections_torch,
)
from neutherm.physics.fuel_properties import (
    fuel_conductivity_np,
    fuel_conductivity_torch,
    heat_generation_np,
    heat_generation_torch,
)
from neutherm.physics.parameters import (
    GeometryParams,
    NeutronicsParams,
    ProblemConfig,
    ThermalParams,
)

__all__ = [
    "ProblemConfig",
    "GeometryParams",
    "NeutronicsParams",
    "ThermalParams",
    "CrossSectionSet",
    "build_pin_cell_xs_np",
    "evaluate_cross_sections_np",
    "evaluate_cross_sections_torch",
    "fuel_conductivity_np",
    "fuel_conductivity_torch",
    "heat_generation_np",
    "heat_generation_torch",
]
