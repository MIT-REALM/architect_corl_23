#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:24:04 2023

@author: rchanna
"""

import jax
import jax.numpy as jnp
from architect.systems.turtle_bot.turtle_bot_types import (
    EnvironmentState,
    Policy,
    Arena,
)


def test_Policy():
    """Test Policy class.

    Tests the __init__ and __call__ methods of the Policy class
    for two different memory_length examples.
    """
    # Test the init and call method of Policy for memory length 1.
    # create a policy
    memory_length = 1
    prng_key = jax.random.PRNGKey(1)
    policy = Policy(prng_key, memory_length)
    x_hist = jnp.array([[0.0, 1.0, 2.0]])
    conc_hist = jnp.array([0.0])
    assert (
        len(policy(x_hist, conc_hist)) == 2
    )  # Checking that output layer has 2 variables/"neurons" (v and w)

    # Test the init and call method of Policy for memory length 5.
    new_memory_length = 5
    new_x_hist = jnp.zeros((5, 3)) + x_hist
    new_conc_hist = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4])
    new_policy = Policy(prng_key, new_memory_length)
    assert (
        len(new_policy(new_x_hist, new_conc_hist)) == 2
    )  # Checking that output layer has 2 variables/"neurons" (v and w)


def test_get_concentration():
    """Test get_concentration method in EnvironmentState class.

    This test assumes the leak forms a Gaussian distribution.

    The concentration of a leak follows 1/sigma * e^-(distance^2/sigma), where
    distance = distance to a source
    sigma = value that influences concentration strength

    If there are multiple sources, the concentrations of each leak
    at the given point are summed for an overall concentration.

    The values being compared to the concentration at (0.0,0.0) were
    calculated in desmos.
    """
    # test case where there is only 1 target. There is only 1 initial condition.
    target_pos1 = jnp.array([[3.0, 4.0]])
    logsigma1 = jnp.log(jnp.array([0.3]))
    prng_key1 = jax.random.PRNGKey(2)
    x_inits1 = 1 * jax.random.uniform(prng_key1, shape=(1, 3))
    new_env1 = EnvironmentState(target_pos1, logsigma1, x_inits1)
    answer = 2.1462085467591996e-36
    concentration = new_env1.get_concentration(
        jnp.array(0.0), jnp.array(0.0), 1
    )  # calculate concentration at origin
    assert jnp.isclose(concentration, answer)

    # test case where there is 2 targets. There is 1 initial condition.
    target_pos2 = jnp.array([[2.3, 4.1], [3.1, 6.8]])
    logsigma2 = jnp.log(jnp.array([0.6, 0.71]))
    prng_key2 = jax.random.PRNGKey(2)
    x_inits2 = 1 * jax.random.uniform(prng_key2, shape=(1, 3))
    new_env2 = EnvironmentState(target_pos2, logsigma2, x_inits2)
    answer = 2.146208e-16
    concentration = new_env2.get_concentration(
        jnp.array(0.0), jnp.array(0.0), 2
    )  # calculate concentration at origin
    assert jnp.allclose(concentration, answer)


def test_policy_prior_logprob_uniform():
    """Test policy_prior_logprob_uniform method in Arena class.

    Assume uniform prior distribution
    (hence equal logprobs for each instance of policy).
    """
    arena = Arena(6.0, 6.0, 1)

    prng_key1 = jax.random.PRNGKey(1)
    memory_length = 1
    policy1 = Policy(prng_key1, memory_length)
    logprob_policy1_uniform = arena.policy_prior_logprob_uniform(policy1)
    prng_key2 = jax.random.PRNGKey(2)
    policy2 = Policy(prng_key2, memory_length)
    logprob_policy2_uniform = arena.policy_prior_logprob_uniform(policy2)
    # assert that the logprobs are equal (uniform) when given different keys
    assert jnp.allclose(logprob_policy1_uniform, logprob_policy2_uniform)


def test_target_pos_prior_logprob():
    """Test target_pos_prior_logprob method in Arena class.

    Assume uniform prior distribution
    (hence equal logprobs for each instance of target_pos).

    Two arena objects are used to test the case when there is 1
    and 2 targets in the environment.
    """

    arena1 = Arena(6.0, 6.0, 1, smoothing=500.0)
    arena2 = Arena(6.0, 6.0, 2, smoothing=500.0)

    target_pos1 = jnp.array([[1.0, 1.5]])
    target_pos2 = jnp.array([[2.1, 1.9]])
    logprob_target1 = arena1.target_pos_prior_logprob(target_pos1)
    logprob_target2 = arena1.target_pos_prior_logprob(target_pos2)
    # assert that the logprobs are equal (uniform)
    assert jnp.allclose(logprob_target1, logprob_target2)

    target_pos3 = jnp.array([[1.0, 1.3], [2.0, 2.3]])
    target_pos4 = jnp.array([[0.1, -0.8], [-1.9, 2.4]])
    logprob_target3 = arena2.target_pos_prior_logprob(target_pos3)
    logprob_target4 = arena2.target_pos_prior_logprob(target_pos4)
    # assert that the logprobs are equal (uniform)
    assert jnp.allclose(logprob_target3, logprob_target4)

    # Check that the logprobs are different for target positions
    # inside and outside the bounds
    target_pos5 = jnp.array([[-3.4, 0.7]])  # out of bounds for 1 target
    logprob_target5 = arena1.target_pos_prior_logprob(target_pos5)
    assert not jnp.allclose(logprob_target1, logprob_target5)
    target_pos6 = jnp.array([[-4.0, 2.0], [0.7, 3.6]])  # out of bounds for 2 targets
    logprob_target6 = arena2.target_pos_prior_logprob(target_pos6)
    assert not jnp.allclose(logprob_target3, logprob_target6)


def test_logsigma_prior_logprob():
    """Test sigma_prior_logprob function in Arena class.

    Assume uniform prior distribution (hence equal logprobs for each instance of sigma).

    Two arena objects are used to test the case when there is 1
    and 2 targets in the environment.
    """

    # smoothing is increased to test the uniform distribution
    arena1 = Arena(6.0, 6.0, 1, smoothing=500.0)
    arena2 = Arena(6.0, 6.0, 2, smoothing=500.0)

    logsigma1 = jnp.log(jnp.array([0.3]))
    logsigma2 = jnp.log(jnp.array([0.6]))
    logprob_logsigma1 = arena1.logsigma_prior_logprob(logsigma1)
    logprob_logsigma2 = arena1.logsigma_prior_logprob(logsigma2)
    # assert that the logprobs are equal (uniform)
    assert jnp.allclose(logprob_logsigma1, logprob_logsigma2, atol=1e-3)

    logsigma3 = jnp.log(jnp.array([0.36, 0.61]))
    logsigma4 = jnp.log(jnp.array([0.49, 0.12]))
    logprob_logsigma3 = arena2.logsigma_prior_logprob(logsigma3)
    logprob_logsigma4 = arena2.logsigma_prior_logprob(logsigma4)
    # assert that the logprobs are equal (uniform)
    assert jnp.allclose(logprob_logsigma3, logprob_logsigma4, atol=1e-3)

    # Check that the logprobs are different for sigmas inside and outside the bounds
    logsigma5 = jnp.log(jnp.array([-0.05]))  # out of bounds sigma for 1 target
    logprob_logsigma5 = arena1.logsigma_prior_logprob(logsigma5)
    assert not jnp.allclose(logprob_logsigma1, logprob_logsigma5)
    logsigma6 = jnp.log(jnp.array([1.2, 0.3]))  # out of bounds sigma for 2 targets
    logprob_logsigma6 = arena2.logsigma_prior_logprob(logsigma6)
    assert not jnp.allclose(logprob_logsigma3, logprob_logsigma6)


def test_x_inits_prior_logpob():
    """Test x_inits_prior_logprob method in Arena class.

    Assume uniform prior distribution
    (hence equal logprobs for each instance of logsigma).

    Tests cases where there is 1 initial condition and 4 initial conditions.
    """
    arena = Arena(6.0, 6.0, 1, smoothing=500.0)

    x_init_spread = 2

    # Note:
    #   values in the arrays must be between -x_init_spread and +x_init_spread
    x_inits1 = jnp.array([[1.6, 0.2, -0.6]])  # N = 1
    x_inits2 = jnp.array([[0.9, -0.3, 0.3]])
    x_inits3 = jnp.array(
        [[-0.9, 0.4, 1.3], [-0.01, 1.4, 1.9], [1.2, 1.1, -0.4]]
    )  # N = 3
    x_inits4 = jnp.array([[1.0, 1.3, 1.2], [0.1, -0.8, 0.6], [-1.9, 1.4, 0.9]])
    logprob_xinit1 = arena.x_inits_prior_logprob(x_inits1, x_init_spread)
    logprob_xinit2 = arena.x_inits_prior_logprob(x_inits2, x_init_spread)
    logprob_xinit3 = arena.x_inits_prior_logprob(x_inits3, x_init_spread)
    logprob_xinit4 = arena.x_inits_prior_logprob(x_inits4, x_init_spread)

    # assert that the logprobs are equal (uniform)
    assert jnp.allclose(logprob_xinit1, logprob_xinit2)
    assert jnp.allclose(logprob_xinit3, logprob_xinit4)


def test_sample_random_policy():
    """Test sample_random_policy method in Arena class.

    Checks that random samples are indeed unique when provided different PRNGkeys.
    """
    arena = Arena(6.0, 6.0, 1)

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 5)
    memory_length = 3

    make = lambda key: arena.sample_random_policy(
        key, memory_length
    )  # anonymous function to get samples for 1 target
    samples_policy = [make(key) for key in keys]
    empty_list = []
    for i in range(len(keys)):
        assert type(samples_policy[i]) == Policy
        if samples_policy[i] not in empty_list:
            empty_list.append(samples_policy[i])
    assert len(empty_list) == len(
        samples_policy
    )  # assert that the policies are truly unique/random


def test_sample_random_target_pos():
    """Test sample_random_target_pos method in Arena class.

    Checks that random samples are indeed unique when provided different PRNGkeys
    and that the returned array has the correct shape
    (depending on number of target locations).
    """
    arena1 = Arena(6.0, 6.0, 1)
    arena2 = Arena(6.0, 6.0, 2)

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 5)

    make1 = lambda key: arena1.sample_random_target_pos(
        key
    )  # anonymous function to get samples for 1 target
    make2 = lambda key: arena2.sample_random_target_pos(
        key
    )  # anonymous function to get samples for 2 targets

    # create arrays of samples
    # shape (5,1,2)
    samples_target_pos1 = jnp.array([make1(key) for key in keys])
    # shape (5,2,2)
    samples_target_pos2 = jnp.array([make2(key) for key in keys])

    for i in range(len(keys)):
        assert (samples_target_pos1[i, :, :].shape) == (
            1,
            2,
        )  # check the shape of target position if 1 target
        assert (samples_target_pos2[i, :, :].shape) == (
            2,
            2,
        )  # check the shape of target positions if 2 targets
    # for each of the sample arrays, find the unique target arrays and their indeces
    # check that the number of unique target arrays equals the number of keys given
    u1, indices1 = jnp.unique(samples_target_pos1, return_index=True, axis=0)
    assert len(indices1) == len(keys)
    u2, indeces2 = jnp.unique(samples_target_pos2, return_index=True, axis=0)
    assert len(indeces2) == len(keys)


def test_sample_random_logsigma():
    """Test sample_random_sigma method in Arena class.

    Checks that random samples are indeed unique when provided different PRNGkeys
    and that the returned array has the correct shape
    (depending on number of target locations).
    """
    arena1 = Arena(6.0, 6.0, 1)
    arena2 = Arena(6.0, 6.0, 2)
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 5)

    make1 = lambda key: arena1.sample_random_logsigma(
        key
    )  # anonymous function to get samples for 1 target
    make2 = lambda key: arena2.sample_random_logsigma(
        key
    )  # anonymous function to get samples for 2 targets

    # create arrays of samples
    samples_logsigma1 = jnp.array([make1(key) for key in keys])
    samples_logsigma2 = jnp.array([make2(key) for key in keys])

    for i in range(len(keys)):
        assert (samples_logsigma1[i, :].shape) == (
            1,
        )  # check the shape of target position if 1 target
        assert (samples_logsigma2[i, :].shape) == (
            2,
        )  # check the shape of target positions if 2 targets
    # for each of the sample arrays, find the unique target arrays and their indeces
    # check that the number of unique target arrays equals the number of keys given
    u1, indices1 = jnp.unique(samples_logsigma1, return_index=True, axis=0)
    assert len(indices1) == len(keys)
    u2, indeces2 = jnp.unique(samples_logsigma2, return_index=True, axis=0)
    assert len(indeces2) == len(keys)


def test_sample_random_x_inits():
    """Test sample_random_x_inits method in Arena class.

    Checks that random samples are indeed unique when provided different PRNGkeys
    and that the returned array has the correct shape.
    """
    arena = Arena(6.0, 6.0, 1)
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 5)
    x_init_spread = 1
    N1 = 1
    N2 = 4

    make1 = lambda key: arena.sample_random_x_inits(
        key, N1, x_init_spread
    )  # anonymous function to get samples for 1 target
    make2 = lambda key: arena.sample_random_x_inits(key, N2, x_init_spread)
    samples_x_inits1 = jnp.array([make1(key) for key in keys])  # shape (5,1,3)
    samples_x_inits2 = jnp.array([make2(key) for key in keys])  # shape (5,4,3)

    for i in range(len(keys)):
        assert samples_x_inits1[i, :, :].shape == (
            N1,
            3,
        )  # assert that returned initial position array is of correct shape
        assert samples_x_inits2[i, :, :].shape == (
            N2,
            3,
        )  # assert that returned initial positions array is of correct shape
    u1, indices1 = jnp.unique(samples_x_inits1, return_index=True, axis=0)
    assert len(indices1) == len(keys)
    u2, indeces2 = jnp.unique(samples_x_inits2, return_index=True, axis=0)
    assert len(indeces2) == len(keys)
