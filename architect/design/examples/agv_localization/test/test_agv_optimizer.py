import pytest

import jax

from architect.design.optimization import (
    VarianceRegularizedOptimizerAD,
)
from architect.design.examples.agv_localization.agv_design_problem import (
    make_agv_localization_design_problem,
)


@pytest.mark.slow
def test_agv_optimizer():
    # Make the design problem (very short time horizon for testing)
    T = 2
    dt = 0.5
    agv_design_problem = make_agv_localization_design_problem(T, dt)

    # Create the optimizer
    variance_weight = 0.1
    sample_size = 2
    vr_opt = VarianceRegularizedOptimizerAD(
        agv_design_problem, variance_weight, sample_size
    )

    # Optimize!
    prng_key = jax.random.PRNGKey(0)
    # Just make sure we don't hit an error
    vr_opt.optimize(prng_key, maxiter=1, jit=True)
