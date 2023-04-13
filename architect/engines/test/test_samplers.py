import jax
import jax.numpy as jnp
import jax.random as jrandom
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from architect.engines.samplers import init_sampler, make_kernel


# Define a test likelihood
@jaxtyped
@beartype
def quadratic_potential(x: Float[Array, " n"]):
    return -0.5 * (x**2).sum()


def test_init_sampler():
    """Test the init_sampler function."""
    n = 4
    position = jnp.ones(n)
    state = init_sampler(position, quadratic_potential, normalize_gradients=False)

    # State should be initialized correctly
    assert jnp.allclose(state.position, position)
    assert jnp.allclose(state.logdensity, -n / 2)
    assert jnp.allclose(state.logdensity_grad, -position)


def test_make_kernel_mala():
    """Test the make_kernel function for the MALA sampler."""
    step_size = 1e-2

    # Scale up the potential to make the test more consistent
    potential = lambda x: 100 * quadratic_potential(x)

    kernel = make_kernel(
        potential, step_size, use_gradients=True, use_stochasticity=True
    )

    # Kernel should be initialized correctly
    assert kernel is not None

    # We should be able to call the kernel
    n = 2
    position = jnp.ones(n)
    state = init_sampler(position, potential, normalize_gradients=False)
    prng_key = jrandom.PRNGKey(0)
    next_state = kernel(prng_key, state)
    assert next_state is not None

    # If we run the kernel for a while, we should get samples that average around the
    # minimum at x = 0
    n_steps = 1000
    tolerance = 1e-2

    @jax.jit
    def one_step(state, rng_key):
        state = kernel(rng_key, state)
        return state, state

    keys = jrandom.split(prng_key, n_steps * 2)
    _, states = jax.lax.scan(one_step, next_state, keys)

    assert jnp.allclose(
        states.position[n_steps:].mean(axis=0), jnp.zeros(n), atol=tolerance
    )
