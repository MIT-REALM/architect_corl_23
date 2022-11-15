import pytest

import jax
import jax.numpy as jnp

from architect.problem_definition import (
    DesignProblem,
    DesignParameters,
    ExogenousParameters,
)
from architect.design.analysis import SensitivityAnalyzer


def create_test_design_problem():
    """Create a simple DesignProblem instance to use during testing.

    This problem has a cost minimum at dp[0] = 0 regardless of regularization, with
    cost = dp[0] ** 2 + abs(ep[0])

    This should have Lipschitz constant 1 with respect to ep

    args:
        simple: bool. If True, do not include exogenous parameters in cost. Otherwise,
                add the exogenous parameters to the cost
    """
    # Create simple design and exogenous parameters
    n = 1
    dp = DesignParameters(n)
    ep = ExogenousParameters(n)

    # Create a simple simulator that passes through the design params and a cost
    # function that computes the squared 2-norm of the design params
    def simulator(dp, ep):
        return jnp.concatenate([dp, ep])

    def cost_fn(dp, ep):
        trace = simulator(dp, ep)
        return (trace[0] ** 2).sum() + jnp.abs(trace[-1])

    problem = DesignProblem(dp, ep, cost_fn, simulator)

    return problem


def test_SensitivityAnalyzer_init():
    """Test initializing a SensitivityAnalyzer"""
    # Get the test problem
    problem = create_test_design_problem()

    # Initialize and make sure it worked
    sample_size = 100
    block_size = 50
    sensitivity_analyzer = SensitivityAnalyzer(problem, sample_size, block_size)
    assert sensitivity_analyzer is not None


@pytest.mark.slow
def test_SensitivityAnalyzer_analyze_detailed():
    """Test using a SensitivityAnalyzer to analyze variance"""
    # Get the test problem
    problem = create_test_design_problem()

    # Initialize the analyzer
    sample_size = 100
    block_size = 50
    sensitivity_analyzer = SensitivityAnalyzer(problem, sample_size, block_size)

    # Get a PRNG key
    key = jax.random.PRNGKey(0)

    # Run the analysis
    summary, _ = sensitivity_analyzer.analyze(key, inject_noise=True)

    # Make sure that the estimated upper bound is about right
    mu = summary["mean"]["mu"]
    sigma = summary["mean"]["sigma"]
    xi = summary["mean"]["xi"]
    assert xi < 0.0, "should be in Weibull regime with upper-bounded support"
    upper_bound = mu - sigma / xi
    assert jnp.isclose(upper_bound, 1.0, rtol=1e-2)


def test_SensitivityAnalyzer_analyze():
    """Test using a SensitivityAnalyzer to analyze variance (just make sure it runs)"""
    # Get the test problem
    problem = create_test_design_problem()

    # Initialize the analyzer
    sample_size = 5
    block_size = 2
    sensitivity_analyzer = SensitivityAnalyzer(problem, sample_size, block_size)

    # Get a PRNG key
    key = jax.random.PRNGKey(0)

    # Run the analysis and make sure it doesn't fail
    sensitivity_analyzer.analyze(key, inject_noise=True, mcmc_steps=10)
