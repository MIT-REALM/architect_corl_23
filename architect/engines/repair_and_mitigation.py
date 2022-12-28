"""
Failure mode prediction and mitigation code.

Provides system-agnostic code for failure mode prediction and mitigation.
"""
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.nn import logsumexp
import matplotlib.pyplot as plt
import seaborn as sns
from beartype import beartype
from beartype.typing import Tuple, Callable
from jaxtyping import Array, Float, Integer, jaxtyped

from architect.types import Params, Sampler, LogLikelihood
from architect.engines.samplers import SamplerState, init_sampler, make_kernel


def softmax(x: Float[Array, "..."], smoothing: float = 5):
    """Return the soft maximum of the given vector"""
    return 1 / smoothing * logsumexp(smoothing * x)


def softmin(x: Float[Array, "..."], smoothing: float = 5):
    """Return the soft minimum of the given vector"""
    return -softmax(-x, smoothing)


@jaxtyped
@beartype
def run_chain(
    prng_key: Integer[Array, "..."],
    kernel: Sampler,
    initial_state: SamplerState,
    num_samples: int,
) -> Tuple[Params, Float[Array, ""]]:
    """
    Run the given MCMC kernel for the specified number of steps, returning last sample
    and associated log likelihood.

    args:
        prng_key: JAX rng key
        kernel: the MCMC kernel to run
        initial_state: the starting state for the sampler
        num_samples: how many samples to take
    returns:
        the last state of the chain and associated log probability
    """

    @jax.jit
    def one_step(_, carry):
        # unpack carry
        state, prng_key = carry

        # Take a step
        prng_key, subkey = jrandom.split(prng_key)
        state = kernel(subkey, state)

        # Re-pack the carry and return
        carry = (state, prng_key)
        return carry

    initial_carry = (initial_state, prng_key)
    final_state, _ = jax.lax.fori_loop(0, num_samples, one_step, initial_carry)

    return final_state.position, final_state.logprob


@jaxtyped
@beartype
def predict_and_mitigate_failure_modes(
    prng_key: Integer[Array, "..."],
    design_params: Params,
    exogenous_params: Params,
    dp_logprior_fn: LogLikelihood,
    ep_logprior_fn: LogLikelihood,
    potential_fn: Callable[[Params, Params], Float[Array, ""]],
    num_rounds: int,
    num_mcmc_steps_per_round: int,
    mcmc_step_size: float,
    use_gradients: bool = True,
    use_stochasticity: bool = True,
) -> Tuple[Params, Params, Float[Array, " n"], Float[Array, " n"]]:
    """
    Run the sequential monte carlo algorithm for SC-ACOPF

    args:
        prng_key: JAX rng key
        design_params: initial controllable parameter population.
        exogenous_params: initial uncontrollable parameter population.
        dp_logprior_fn: function taking a single set of design parameters, returning
            the non-normalized log probability of those parameters according to the
            prior distribution.
        ep_logprior_fn: function taking a single set of exogenous parameters,returning
            the non-normalized log probability of those parameters according to the
            prior distribution.
        potential_fn: function taking a single set of design and exogenous parameters,
            returning the cost metric for the system's performance.
        num_rounds: how many steps to take for the top-level predict-repair algorithm.
        num_mcmc_steps_per_round: the number of steps to run for each set of MCMC chains
            within each round.
        mcmc_step_size: the size of MCMC steps.
        use_gradients: if True, use gradients in the proposal and acceptance steps.
        use_stochasticity: if True, add a Gaussian to the proposal and acceptance steps.
    returns:
        - The updated population of design parameters
        - The updated population of exogenous parameters
        - The overall log probabilities for the design parameters (before the last
            update to the exogenous parameters)
        - The overall log probabilities for the exogenous parameters
    """
    # Make the function to run one round of repair/predict
    @jax.jit
    def one_smc_step(_, carry):
        # Unpack the carry
        prng_key, current_dps, current_eps, _, _ = carry

        ###############################################
        # Repair the currently predicted failure modes
        ###############################################

        # Create a loglikelihood function that minimizes the average potential across
        # all currently predicted failure modes. Need to make this negative so that
        # small potentials/costs -> higher likelihoods
        dp_potential_fn = lambda dp: -jax.vmap(potential_fn, in_axes=(None, 0))(
            dp, current_eps
        ).mean()
        dp_logprob_fn = lambda dp: dp_logprior_fn(dp) + dp_potential_fn(dp)

        # Initialize the chains for this kernel
        initial_dp_sampler_states = jax.vmap(init_sampler, in_axes=(0, None))(
            current_dps, dp_logprob_fn
        )

        # Make the sampling kernel
        dp_kernel = make_kernel(
            dp_logprob_fn, mcmc_step_size, use_gradients, use_stochasticity
        )

        # Run the chains and update the design parameters
        n_chains = initial_dp_sampler_states.logdensity.shape[0]
        prng_key, sample_key = jrandom.split(prng_key)
        sample_keys = jrandom.split(sample_key, n_chains)
        current_dps, dp_logprobs = jax.vmap(run_chain, in_axes=(0, None, 0, None))(
            sample_keys,
            dp_kernel,
            initial_dp_sampler_states,
            num_mcmc_steps_per_round,
        )

        ################################################
        # Predict failures for current design parameters
        ################################################

        # Create a loglikelihood function that maximizes the minimum potential across
        # all current design parameters. Need to make this positive so that
        # large potentials/costs -> higher likelihoods
        ep_potential_fn = lambda ep: softmin(
            jax.vmap(potential_fn, in_axes=(0, None))(current_dps, ep)
        )
        ep_logprob_fn = lambda ep: ep_logprior_fn(ep) + ep_potential_fn(ep)

        # Initialize the chains for this kernel
        initial_ep_sampler_states = jax.vmap(init_sampler, in_axes=(0, None))(
            current_eps, ep_logprob_fn
        )

        # Make the sampling kernel
        ep_kernel = make_kernel(
            ep_logprob_fn, mcmc_step_size, use_gradients, use_stochasticity
        )

        # Run the chains and update the design parameters
        n_chains = initial_ep_sampler_states.logdensity.shape[0]
        prng_key, sample_key = jrandom.split(prng_key)
        sample_keys = jrandom.split(sample_key, n_chains)
        current_eps, ep_logprobs = jax.vmap(run_chain, in_axes=(0, None, 0, None))(
            sample_keys,
            ep_kernel,
            initial_ep_sampler_states,
            num_mcmc_steps_per_round,
        )

        # Return the updated carry
        return (prng_key, current_dps, current_dps, dp_logprobs, ep_logprobs)

    # Run the appropriate number of steps
    initial_carry = (
        prng_key,
        design_params,
        exogenous_params,
        jnp.array(0.0),  # dummy logprobs for now
        jnp.array(0.0),
    )
    _, dispatches, dispatch_logprobs, networks, network_logprobs = jax.lax.fori_loop(
        0, num_rounds, one_smc_step, initial_carry
    )

    return dispatches, dispatch_logprobs, networks, network_logprobs
