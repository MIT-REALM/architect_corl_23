import pytest

import jax

from architect.design.optimization import (
    VarianceRegularizedOptimizerAD,
)
from architect.design.examples.multi_agent_manipulation.mam_design_problem import (
    make_mam_design_problem,
)


@pytest.mark.slow
def test_optimizer():
    """Run design optimization and plot the results"""
    prng_key = jax.random.PRNGKey(0)

    # Make the design problem
    layer_widths = (2 * 3 + 3, 32, 2 * 2)
    dt = 0.01
    prng_key, subkey = jax.random.split(prng_key)
    mam_design_problem = make_mam_design_problem(layer_widths, dt, subkey)

    # Create the optimizer
    variance_weight = 0.1
    sample_size = 2
    vr_opt = VarianceRegularizedOptimizerAD(
        mam_design_problem, variance_weight, sample_size
    )

    # Optimize! Make sure nothing fails
    prng_key, subkey = jax.random.split(prng_key)
    vr_opt.optimize(subkey, disp=True, maxiter=1, jit=True)
