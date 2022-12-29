"""Implement sampling algorithms"""
import operator

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from beartype import beartype
from beartype.typing import Union, NamedTuple
from jaxtyping import Array, Float, Bool, jaxtyped

from architect.types import LogLikelihood, PRNGKeyArray, Sampler, Params


# @jaxtyped
# @beartype  # Commented out until beartype 0.12.0 release (TODO@dawsonc)
class SamplerState(NamedTuple):
    """State for samplers."""

    position: Params
    logdensity: Float[Array, ""]
    logdensity_grad: Params


@jaxtyped
@beartype
def init_sampler(position: Params, logdensity_fn: LogLikelihood) -> SamplerState:
    grad_fn = jax.value_and_grad(logdensity_fn)
    logdensity, logdensity_grad = grad_fn(position)
    return SamplerState(position, logdensity, logdensity_grad)


@jaxtyped
@beartype
def make_kernel(
    logdensity_fn: LogLikelihood,
    step_size: float,
    use_gradients: Union[bool, Bool[Array, ""]] = True,
    use_stochasticity: Union[bool, Bool[Array, ""]] = True,
) -> Sampler:
    """
    Build a kernel for a sampling algorithm (either MALA, RMH, or MLE).

    Here's the behavior of this sampler as a function of the arguments:

                                    use_gradients = True | use_gradients = False
                              //====================================================
    use_stochasticity = True  ||            MALA         |          RMH
    --------------------------||----------------------------------------------------
    use_stochasticity = False ||      Gradient descent   |     doesn't move
                              \\====================================================

    args:
        logdensity_fn: the non-normalized log likelihood function
        step_size: the size of the step to take
        use_gradients: if True, use gradients in the proposal and acceptance steps
        use_stochasticity: if True, add a Gaussian to the proposal and acceptance steps
    """
    # A generic Metropolis-Hastings-style MCMC algorithm has 2 steps: a proprosal and
    # an accept/reject step.

    # Start by defining the proposal log likelihood
    @jaxtyped
    @beartype
    def transition_log_probability(
        state: SamplerState, new_state: SamplerState, step_size: float
    ):
        """Compute the log likelihood of proposing new_state starting in state."""
        # Based on the MALA implementation in Blackjax, but extended to allow
        # turning it into RMH/gradient descent

        # If we're not using gradients, zero out the gradient in a JIT compatible way
        grad = jax.lax.cond(
            use_gradients,
            lambda x: x,
            lambda x: jtu.tree_map(jnp.zeros_like, x),
            state.logdensity_grad,
        )

        theta = jtu.tree_map(
            lambda new_x, x, g: new_x - x - step_size * g,
            new_state.position,
            state.position,
            grad,
        )
        theta_dot = jtu.tree_reduce(
            operator.add, jtu.tree_map(lambda x: jnp.sum(x * x), theta)
        )

        transition_log_probability = -0.25 * (1.0 / step_size) * theta_dot
        return transition_log_probability

    @jaxtyped
    @beartype
    def one_step(prng_key: PRNGKeyArray, state: SamplerState) -> SamplerState:
        """Generate a new sample"""
        # Split the key
        proposal_key, acceptance_key = jrandom.split(prng_key)

        # Propose a new state
        delta_x = jtu.tree_map(
            jnp.zeros_like,
            state.position,
        )

        # Add gradients to the proposal if using
        delta_x = jax.lax.cond(
            use_gradients,
            lambda dx: jtu.tree_map(
                lambda leaf, grad: leaf + step_size * grad,
                dx,
                state.logdensity_grad,
            ),
            lambda dx: dx,
            delta_x,
        )

        # Generate noise with the same shape as the position pytree
        p, unravel_fn = jax.flatten_util.ravel_pytree(state.position)
        sample = jrandom.normal(proposal_key, shape=p.shape, dtype=p.dtype)
        noise = unravel_fn(sample)

        # Add it to the change in state if using stochasticity
        delta_x = jax.lax.cond(
            use_stochasticity,
            lambda dx: jtu.tree_map(
                lambda leaf, d: leaf + jnp.sqrt(2 * step_size) * d,
                dx,
                noise,
            ),
            lambda dx: dx,
            delta_x,
        )

        # Make the state proposal
        new_position = jtu.tree_map(
            lambda x, dx: x + dx,
            state.position,
            delta_x,
        )
        new_logdensity, new_logdensity_grad = jax.value_and_grad(logdensity_fn)(
            new_position
        )
        new_state = SamplerState(new_position, new_logdensity, new_logdensity_grad)

        # Accept/reject the state using the Metropolis-Hasting rule
        log_p_accept = (
            new_state.logdensity
            - state.logdensity
            + transition_log_probability(new_state, state, step_size)
            - transition_log_probability(state, new_state, step_size)
        )
        log_p_accept = jnp.where(jnp.isnan(log_p_accept), -jnp.inf, log_p_accept)
        p_accept = jnp.clip(jnp.exp(log_p_accept), a_max=1)
        # If we're not using stochasticity, then always accept
        do_accept = jax.lax.cond(
            use_stochasticity,
            lambda: jrandom.bernoulli(acceptance_key, p_accept),
            lambda: jnp.array(True),
        )

        return jax.lax.cond(
            do_accept,
            lambda: new_state,
            lambda: state,
        )

    return one_step
