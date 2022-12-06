import pytest

import jax.numpy as jnp
import jax.random as jrandom

from architect.systems.power_systems.acopf import ACOPF, Dispatch, Network


@pytest.fixture
def test_system():
    """Define a 2-bus test system"""
    nominal_network = Network(
        lines=jnp.array([[0, 1]]),
        line_conductances=jnp.array([1.0]),
        line_susceptances=jnp.array([1.0]),
        shunt_conductances=jnp.array([0.1, 0.1]),
        shunt_susceptances=jnp.array([0.0, 0.0]),
    )
    return ACOPF(
        nominal_network,
        bus_active_limits=jnp.array([[0, 10.0], [-10.0, 0]]),
        bus_reactive_limits=jnp.array([[-10.0, 10.0], [-10.0, 10.0]]),
        bus_voltage_limits=jnp.array([[0.9, 1.1], [0.9, 1.1]]),
        bus_active_linear_costs=jnp.array([1.0, -1.0]),
        bus_active_quadratic_costs=jnp.array([1.0, 0.0]),
        bus_reactive_linear_costs=jnp.array([0.0, 0.0]),
        constraint_penalty=100.0,
        prob_line_failure=jnp.array([0.1]),
        line_conductance_stddev=jnp.array([0.1]),
        line_susceptance_stddev=jnp.array([0.1]),
    )


def test_ACOPF___post_init__(test_system: ACOPF):
    """Test the derived values"""
    assert test_system.n_bus == 2
    assert test_system.n_line == 1


def test_ACOPF___call__(test_system: ACOPF):
    """Test the forward evaluation"""
    voltage_amplitudes = jnp.array([1.0, 1.0])
    voltage_angles = jnp.array([0.0, 0.1])
    dispatch = Dispatch(voltage_amplitudes, voltage_angles)
    result = test_system(dispatch, test_system.nominal_network)

    assert result is not None


def test_ACOPF_dispatch_prior_logprob(test_system: ACOPF):
    """Test the dispatch prior logprobability"""
    # Since the prior distribution is uniform, two different valid dispatches should
    # have the same log probability
    dispatch_1 = Dispatch(jnp.array([1.05, 1.05]), jnp.array([0.0, 0.1]))
    dispatch_2 = Dispatch(jnp.array([0.95, 0.95]), jnp.array([-1.0, 1.0]))

    logprob_1 = test_system.dispatch_prior_logprob(dispatch_1)
    logprob_2 = test_system.dispatch_prior_logprob(dispatch_2)

    assert jnp.allclose(logprob_1, logprob_2)


def test_ACOPF_network_prior_logprob(test_system: ACOPF):
    """Test the network prior logprobability"""
    # TODO make a more sophisticated test.
    # Test to make sure the function runs and returns the correct thing
    # (it's typeguarded so if we avoid an error here we're all good)
    logprob = test_system.network_prior_logprob(test_system.nominal_network)
    assert logprob is not None


def test_ACOPF_sample_random_dispatch(test_system: ACOPF):
    """Test the dispatch sampling function"""
    # TODO make a more sophisticated test.
    # Test to make sure the function runs and returns the correct thing
    # (it's typeguarded so if we avoid an error here we're all good)
    dispatch = test_system.sample_random_dispatch(jrandom.PRNGKey(0))
    assert dispatch is not None


def test_ACOPF_sample_random_network(test_system: ACOPF):
    """Test the network sampling function"""
    # TODO make a more sophisticated test.
    # Test to make sure the function runs and returns the correct thing
    # (it's typeguarded so if we avoid an error here we're all good)
    network = test_system.sample_random_network(jrandom.PRNGKey(0))
    assert network is not None
