from abc import ABC, abstractmethod, abstractproperty

import equinox as eqx
from jaxtyping import Array, Float, PyTree


class AutonomousSystem(eqx.Module, ABC):
    """
    Defines a general autonomous system with design and exogenous parameters.

    The main function of an AutonomousSystem is to simulate its own behavior.
    Subclasses need to implement the `simulate` method that returns a trace (a
    PyTree of JAX arrays representing the behavior of the system over time).

    We represent AutonomousSystems as Equinox modules (callable PyTrees), where the
    `__call__` function simulates the behavior of the AutonomousSystem and returns the
    trace.

    Both the design and exogenous parameters are implemented as parameters of an Equinox
    module, which allows easy JIT and grad transformations. To enable freezing either
    the design or exogenous parameters, AutonomousSystem implements properties that
    return a filter specification allowing `eqx.filter_grad` (and other `eqx.filter_*`
    transformations) to select either only the design parameters (`dp_filter_spec`) or
    only the exogenous parameters (`ep_filter_spec`).
    """

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

    @abstractproperty
    def dp_filter_spec(self) -> PyTree[bool]:
        """Returns a PyTree filter selecting only the design parameters."""
        # # Example of how to set up a parameter filter
        # filter_spec = jtu.tree_map(lambda _: False, self)
        # filter_spec = eqx.tree_at(
        #     lambda spec: spec.design_parameters,
        #     filter_spec,
        #     replace=True,
        # )

        # return filter_spec
        pass

    @abstractproperty
    def ep_filter_spec(self) -> PyTree[bool]:
        """Returns a PyTree filter selecting only the exogenous parameters."""
        # # Example of how to set up a parameter filter
        # filter_spec = jtu.tree_map(lambda _: False, self)
        # filter_spec = eqx.tree_at(
        #     lambda spec: spec.exogenous_parameters,
        #     filter_spec,
        #     replace=True,
        # )

        # return filter_spec
        pass
