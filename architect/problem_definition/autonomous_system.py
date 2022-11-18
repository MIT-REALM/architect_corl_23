from abc import ABC, abstractmethod

import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import Array, Float, PyTree

from architect.problem_definition.parameters import Parameters


class AutonomousSystem(eqx.Module, ABC):
    """
    Defines a general autonomous system with design and exogenous parameters.

    The main function of an AutonomousSystem is to simulate its own behavior.
    Subclasses need to implement the `simulate` method that returns a trace (a
    PyTree of JAX arrays representing the behavior of the system over time).

    We represent AutonomousSystems as Equinox modules (callable PyTrees), where the
    `__call__` function simulates the behavior of the AutonomousSystem and returns the
    trace.

    Both the design and exogenous parameters are implemented as Equinox modules,
    which allow easy JIT and grad transformations. To enable freezing either the
    design or exogenous parameters, AutonomousSystem implements properties that
    return a filter specification allowing `eqx.filter_grad` (and other `eqx.filter_*`
    transformations) to select either only the design parameters (`dp_filter_spec`) or
    only the exogenous parameters (`ep_filter_spec`).
    """

    design_parameters: Parameters
    exogenous_parameters: Parameters

    def __init__(self, design_params: Parameters, exogenous_params: Parameters):
        """
        Create an AutonomousSystem object.

        args:
            design_params: the design parameters for this system
            exogenous_params: the design parameters for this system
        returns:
            a new AutonomousSystem
        """
        self.design_parameters = design_params
        self.exogenous_parameters = exogenous_params

    def __call__(self) -> PyTree[Float[Array, "..."]]:
        """Simulate the behavior of this AutonomousSystem"""
        return self.simulate()

    @abstractmethod
    def simulate(self) -> PyTree[Float[Array, "..."]]:
        """
        Simulate the behavior of this AutonomousSystem.

        returns:
            a PyTree of the simulation trace
        """
        raise NotImplementedError()

    @property
    def dp_filter_spec(self) -> PyTree[bool]:
        """Returns a PyTree filter selecting only the design parameters."""
        filter_spec = jtu.tree_map(lambda _: False, self)
        filter_spec = eqx.tree_at(
            lambda spec: spec.design_parameters,
            filter_spec,
            replace=True,
        )

        return filter_spec

    @property
    def ep_filter_spec(self) -> PyTree[bool]:
        """Returns a PyTree filter selecting only the exogenous parameters."""
        filter_spec = jtu.tree_map(lambda _: False, self)
        filter_spec = eqx.tree_at(
            lambda spec: spec.exogenous_parameters,
            filter_spec,
            replace=True,
        )

        return filter_spec
