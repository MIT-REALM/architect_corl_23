from abc import ABC, abstractclassmethod, abstractmethod

import equinox as eqx
from jax._src.prng import PRNGKeyArray
from jaxtyping import Array, Float


class Parameters(eqx.Module, ABC):
    """
    Parameters represent inputs that govern the behavior of a system.

    Parameters are represented using Equinox modules (callable PyTrees), where the
    `__call__` method returns the (potentially unnormalized) log likelihood of the
    parameters according to some prior distribution.

    Subclasses should define any needed parameter fields and implement the `__init__`
    and `log_likelihood` member functions and the `sample` class method that samples
    new values from the prior distribution.
    """

    def __call__(self) -> Float[Array, ""]:  # noqa F722: empty quotes for scalar array
        """Compute the log likelihood of these parameters"""
        return self.log_likelihood()

    @abstractmethod
    def log_likelihood(self) -> Float[Array, ""]:  # noqa F722
        """Compute the log likelihood of these parameters."""
        raise NotImplementedError()

    @abstractclassmethod
    def sample(self, prng_key: PRNGKeyArray) -> "Parameters":
        """Sample values for these parameters from this distribution.

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
                      This method will not split the key, it will be consumed.
        returns:
            a new Parameters instance sampled from the prior distribution.
        """
        raise NotImplementedError()
