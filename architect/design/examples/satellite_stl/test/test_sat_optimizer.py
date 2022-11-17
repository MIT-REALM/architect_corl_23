import pytest

from architect.design.examples.satellite_stl.run_adversarial_local_optimizer import (
    run_optimizer
)


@pytest.mark.slow
def test_sat_optimizer():
    # Run the optimization just to make sure the plumbing is intact (it won't converge
    # in just 1 step)
    run_optimizer(rounds=2, maxiter=1)
