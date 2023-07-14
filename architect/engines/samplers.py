"""Implement sampling algorithms"""
import operator

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from beartype import beartype
from beartype.typing import NamedTuple, Union
from jaxtyping import Array, Bool, Float, Integer, jaxtyped

from architect.types import LogLikelihood, Params, PRNGKeyArray, Sampler


# @jaxtyped
# @beartype  # Commented out until beartype 0.12.0 release (TODO@dawsonc)
class SamplerState(NamedTuple):
    """State for samplers."""

    position: Params
    logdensity: Float[Array, ""]
    logdensity_grad: Params
    num_accepts: Integer[Array, ""]


@jaxtyped
@beartype
def init_sampler(
    position: Params,
    logdensity_fn: LogLikelihood,
    normalize_gradients: Union[bool, Bool[Array, ""]] = True,
) -> SamplerState:
    """
    Initialize a sampler.

    args:
        position: the initial position of the sampler
        logdensity_fn: the non-normalized log likelihood function
        normalize_gradients: if True, normalize gradients to have unit L2 norm
    """
    grad_fn = jax.value_and_grad(logdensity_fn)
    logdensity, logdensity_grad = grad_fn(position)

    # Drop nans
    logdensity_grad = jtu.tree_map(
        lambda grad: jnp.where(jnp.isnan(grad), 0.0, grad), logdensity_grad
    )

    # Normalize the gradients if requested
    block_grad_norms = jtu.tree_map(
        lambda x: jnp.sqrt(jnp.sum(x**2) + 1e-3), logdensity_grad
    )
    block_grad_norms = jtu.tree_map(
        lambda x: jnp.where(x >= 1.0, x, 1.0), block_grad_norms
    )
    normalized_logdensity_grad = jtu.tree_map(
        lambda grad, block_norm: grad / block_norm,
        logdensity_grad,
        block_grad_norms,
    )
    logdensity_grad = jax.lax.cond(
        normalize_gradients,
        lambda: normalized_logdensity_grad,
        lambda: logdensity_grad,
    )

    return SamplerState(position, logdensity, logdensity_grad, jnp.array(0))


@jaxtyped
@beartype
def make_kernel(
    logdensity_fn: LogLikelihood,
    step_size: Union[float, Float[Array, ""]],
    use_gradients: Union[bool, Bool[Array, ""]] = True,
    use_stochasticity: Union[bool, Bool[Array, ""]] = True,
    grad_clip: Union[float, Float[Array, ""]] = float("inf"),
    normalize_gradients: Union[bool, Bool[Array, ""]] = True,
    use_mh: Union[bool, Bool[Array, ""]] = True,
) -> Sampler:
    """
    Build a kernel for a sampling algorithm (either MALA, RMH, or MLE).

    Here's the behavior of this sampler as a function of the arguments:

    (inside each block indicates behavior with use_mh = True/False if behavior differs)

                                    use_gradients = True | use_gradients = False
                              //====================================================
    use_stochasticity = True  ||         MALA/ULA        |    RMH/random walk
    --------------------------||----------------------------------------------------
    use_stochasticity = False ||      Gradient descent   |     doesn't move
                              \\====================================================

    args:
        logdensity_fn: the non-normalized log likelihood function
        step_size: the size of the step to take
        use_gradients: if True, use gradients in the proposal and acceptance steps
        use_stochasticity: if True, add a Gaussian to the proposal and acceptance steps
        grad_clip: maximum value to clip gradients
        normalize_gradients: if True, normalize gradients to have unit L2 norm
        use_mh: if True, use Metropolis-Hastings acceptance, otherwise always accept
    """
    # A generic Metropolis-Hastings-style MCMC algorithm has 2 steps: a proprosal and
    # an accept/reject step.

    # Start by defining the proposal log likelihood
    @jaxtyped
    @beartype
    def transition_log_probability(
        state: SamplerState,
        new_state: SamplerState,
        step_size: Union[float, Float[Array, ""]],
    ):
        """Compute the log likelihood of proposing new_state starting in state."""
        # Based on the MALA implementation in Blackjax, but extended to allow
        # turning it into RMH/gradient descent

        # If we're not using gradients, zero out the gradient in a JIT compatible way
        grad = jax.lax.cond(
            use_gradients,
            lambda x: jtu.tree_map(
                lambda v: jnp.clip(v, a_min=-grad_clip, a_max=grad_clip), x
            ),
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
        new_position = jtu.tree_map(lambda x, dx: x + dx, state.position, delta_x)
        new_logdensity, new_grad = jax.value_and_grad(logdensity_fn)(new_position)

        # Drop nans
        new_grad = jtu.tree_map(
            lambda grad: jnp.where(jnp.isnan(grad), 0.0, grad), new_grad
        )

        # Normalize the gradients if requested.
        # We do this by re-scaling gradients to match the rough order of the
        # Gaussian noise
        block_grad_norms = jtu.tree_map(
            lambda x: jnp.sqrt(jnp.sum(x**2) + 1e-3), new_grad
        )
        block_grad_norms = jtu.tree_map(
            lambda x: jnp.where(x >= 1.0, x, 1.0), block_grad_norms
        )
        block_noise_norms = jtu.tree_map(
            lambda x: jnp.sqrt(jnp.sum(x**2) + 1e-3), noise
        )
        normalized_new_grad = jtu.tree_map(
            lambda grad, block_grad_norm, block_noise_norm: grad
            * block_noise_norm
            / block_grad_norm,
            new_grad,
            block_grad_norms,
            block_noise_norms,
        )
        new_grad = jax.lax.cond(
            normalize_gradients,
            lambda: normalized_new_grad,
            lambda: new_grad,
        )

        new_state = SamplerState(
            new_position,
            new_logdensity,
            new_grad,
            state.num_accepts + 1,
        )

        # Accept/reject the state using the Metropolis-Hasting rule
        log_p_accept = (
            new_state.logdensity
            - state.logdensity
            + transition_log_probability(new_state, state, step_size)
            - transition_log_probability(state, new_state, step_size)
        )
        log_p_accept = jnp.where(jnp.isnan(log_p_accept), -jnp.inf, log_p_accept)
        p_accept = jnp.clip(jnp.exp(log_p_accept), a_max=1)
        # If we're not using stochasticity or MH has been disabled, then always accept
        do_accept = jax.lax.cond(
            jnp.logical_and(use_stochasticity, use_mh),
            lambda: jrandom.bernoulli(acceptance_key, p_accept),
            lambda: jnp.array(True),
        )

        accepted_state = jax.lax.cond(do_accept, lambda: new_state, lambda: state)
        return accepted_state

    return one_step
