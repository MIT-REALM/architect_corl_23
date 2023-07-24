#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:24:04 2023

@author: rchanna
"""

import jax
import jax.numpy as jnp
from architect.systems.turtle_bot import turtle_bot_types
from architect.systems.turtle_bot.turtle_bot_types import Environment_state, Policy, Arena
from jax.nn import log_sigmoid
from architect import utils

def test_Policy():
    """
    Tests the __init__ and __call__ methods of the Policy class for two different memory_length examples.
    """
    # Test the init and call method of Policy for memory length 1.
    # create a policy
    memory_length = 1
    prng_key = jax.random.PRNGKey(1)
    policy = Policy(prng_key, memory_length)
    x_hist = jnp.array([[0.0, 1.0, 2.0]])
    conc_hist = jnp.array([0.0])
    assert(len(policy.layers)==4) # Checking for 4 layers
    (policy(x_hist, conc_hist))
    assert (len(policy(x_hist, conc_hist)) == 2) # Checking that output layer has 2 variables/"neurons" (v and w)
    
    # Test the init and call method of Policy for memory length 5.
    new_memory_length = 5
    new_x_hist = jnp.zeros((5,3)) + x_hist
    new_conc_hist = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4])
    new_policy = Policy(prng_key, new_memory_length)
    assert(len(new_policy.layers)==4) # Checking for 4 layers
    print ("Length:",len(new_policy(new_x_hist, new_conc_hist)))
    assert (len(new_policy(new_x_hist, new_conc_hist)) == 2) # Checking that output layer has 2 variables/"neurons" (v and w)
    


def test_get_concentration(): 
# check the shape of inputs and outputs
    
    """
    This test assumes the leak forms a Gaussian distribution (as does the get_concentration 
    method in Environment_states).
    
    The concentration of a leak follows 1/sigma * e^-(distance^2/sigma) with distance being the distance to a source 
    and sigma influencing the concentration strength. If there are multiple sources, the concentrations of each leak
    at the given point are summed for an overall concentration.
    
    The values being compared to the concentration at (0.0,0.0) were 
    calculated in desmos.
    """
    # test case where there is only 1 target. There is only 1 initial condition.
    
    target_pos1 = jnp.array([[3.0,4.0]])
    sigma1 = jnp.array([0.3])
    prng_key1 = jax.random.PRNGKey(2)
    x_inits1 = 1 * jax.random.uniform(prng_key1, shape=(1, 3)) 
    new_env1 = Environment_state(target_pos1, sigma1, x_inits1)
    answer = (1/0.3*jnp.exp(-25/0.3))
    concentration = new_env1.get_concentration(0.0,0.0, 1)
    print (concentration.shape)
    assert(jnp.isclose(concentration,answer))
    
    
    # test case where there is 2 targets. There is 1 initial condition.
    target_pos = jnp.array([[2.3,4.1], [3.1,6.8]])
    sigma = jnp.array([0.6,0.71])
    prng_key = jax.random.PRNGKey(2)
    x_inits = 1 * jax.random.uniform(prng_key, shape=(1, 3)) 
    new_env = Environment_state(target_pos, sigma, x_inits)
    answer = 2.146208e-16
    concentration = new_env.get_concentration(0.0,0.0,2)
    #print(concentration)
    close = jnp.isclose(concentration, answer)
    assert(jnp.all(close))
    

def test_policy_prior_logprob_uniform():
    """
    Tests the policy_prior_logprob_uniform function that returns the logprob of a given policy.
    
    Assume uniform prior distribution (hence equal logprobs for each instance of policy).
    """
    arena1 = Arena(6.0,6.0,1)
    arena2 = Arena(6.0,6.0,2)
    
    prng_key1 = jax.random.PRNGKey(1)
    memory_length = 1
    policy1 = Policy(prng_key1, memory_length)
    logprob_policy1_uniform = arena1.policy_prior_logprob_uniform(policy1)
    prng_key2 = jax.random.PRNGKey(2)
    policy2 = Policy(prng_key2, memory_length)
    logprob_policy2_uniform = arena1.policy_prior_logprob_uniform(policy2)
    # assert that the logprobs are equal (uniform) when given different keys
    assert(logprob_policy1_uniform == logprob_policy2_uniform) 


def test_target_pos_prior_logprob():
    """
    Tests the target_pos_prior_logprob function that returns the logprob of a given target_pos array.
    
    Assume uniform prior distribution (hence equal logprobs for each instance of target_pos).
    
    Two arena objects are used to test the case when there is 1 and 2 targets in the environment.
    """
    
    
    arena1 = Arena(6.0,6.0,1, smoothing=500.0)
    arena2 = Arena(6.0,6.0,2, smoothing=500.0)
    
    target_pos1 = jnp.array([[1.0,1.5]])
    target_pos2 = jnp.array([[2.1,1.9]])
    logprob_target1 = arena1.target_pos_prior_logprob(target_pos1)
    logprob_target2 = arena1.target_pos_prior_logprob(target_pos2)
    # assert that the logprobs are equal (uniform) 
    print (logprob_target1)
    print (logprob_target2)
    assert(jnp.allclose(logprob_target1, logprob_target2))
    
    target_pos3 = jnp.array([[1.0,1.3], [2.0,2.3]])
    target_pos4 = jnp.array([[0.1, -0.8], [-1.9, 2.4]])
    logprob_target3 = arena2.target_pos_prior_logprob(target_pos3)
    logprob_target4 = arena2.target_pos_prior_logprob(target_pos4)
    # assert that the logprobs are equal (uniform)
    assert jnp.allclose(logprob_target3, logprob_target4)

def test_sigma_prior_logprob():
    """
    Tests the sigma_prior_logprob function that returns the logprob of a given sigma array.
    
    Assume uniform prior distribution (hence equal logprobs for each instance of sigma).
    
    Two arena objects are used to test the case when there is 1 and 2 targets in the environment.
    """
    
    # smoothing is increased to test the uniform distribution
    arena1 = Arena(6.0, 6.0,1, smoothing=500.0)
    arena2 = Arena(6.0,6.0,2, smoothing=500.0)

    sigma1 = jnp.array([0.3])
    sigma2 = jnp.array([0.6])
    logprob_sigma1 = arena1.sigma_prior_logprob(sigma1)
    logprob_sigma2 = arena1.sigma_prior_logprob(sigma2)
    # assert that the logprobs are equal (uniform)
    assert (abs(logprob_sigma1-logprob_sigma2))< 1e-3

    
    sigma3 = jnp.array([0.36, 0.61])
    sigma4 = jnp.array([0.49, 0.12])
    logprob_sigma3 = arena2.sigma_prior_logprob(sigma3)
    logprob_sigma4 = arena2.sigma_prior_logprob(sigma4)
    # print(logprob_sigma3)
    # print(logprob_sigma4)
    # # assert that the logprobs are equal (uniform)
    assert (abs(logprob_sigma3-logprob_sigma4))< 1e-3

    
def test_x_inits_prior_logpob():
    """
    Tests the x_inits_prior_logprob function that returns the logprob of a given sigma array.
    
    Assume uniform prior distribution (hence equal logprobs for each instance of sigma).
    
    Tests cases where there is 1 initial condition and 4 initial conditions. x_init_spread is assumed to be 1.
    """
    arena1 = Arena(6.0,6.0,1, smoothing=500.0)
    
    x_init_spread = 2
    
    # Note that the values in the arrays must be between -x_init_spread and +x_init_spread
    x_inits1 = jnp.array([[1.6,0.2,-0.6]]) # N = 1
    x_inits2 = jnp.array([[0.9,-0.3,0.3]])
    x_inits3 = jnp.array([[-0.9, 0.4, 1.3], [-0.01, 1.4, 1.9], [1.2, 1.1, -0.4]]) # N = 3
    x_inits4 = jnp.array([[1.0, 1.3, 1.2], [0.1, -0.8, 0.6], [-1.9, 1.4, 0.9]])
    logprob_xinit1 = arena1.x_inits_prior_logprob(x_inits1, x_init_spread)
    logprob_xinit2 = arena1.x_inits_prior_logprob(x_inits2, x_init_spread)
    logprob_xinit3 = arena1.x_inits_prior_logprob(x_inits3, x_init_spread)
    logprob_xinit4 = arena1.x_inits_prior_logprob(x_inits4, x_init_spread)

    # assert that the logprobs are equal (uniform)
    assert jnp.allclose(logprob_xinit1, logprob_xinit2)
    assert jnp.allclose(logprob_xinit3, logprob_xinit4)
    
def test_sample_random_policy():
    """
    Checks that random samples are indeed unique when provided different PRNGkeys.
    """
    arena1 = Arena(6.0,6.0,1)
    
    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4, key5 = jax.random.split(key, 5)
    keys = [key1, key2, key3, key4, key5]
    memory_length = 3
       
    make1 = lambda key: arena1.sample_random_policy(key, memory_length) # anonymous function to get samples for 1 target 
    samples_policy1 = [make1(key1), make1(key2), make1(key3), make1(key4), make1(key5)]
    empty_list = []
    for i in range (len(keys)):
        assert type(samples_policy1[i])== Policy
        if samples_policy1[i] not in empty_list:
            empty_list.append(samples_policy1[i])
    assert len(empty_list) == len(samples_policy1) # assert that the policies are truly unique/random

def test_sample_random_target_pos():
    """
    Checks that random samples are indeed unique when provided different PRNGkeys and that the returned array
    
    has the correct shape (depending on number of target locations).
    """
    arena1 = Arena(6.0,6.0,1)
    arena2 = Arena(6.0,6.0,2)
    
    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4, key5 = jax.random.split(key, 5)
    keys = key1, key2, key3, key4, key5
    
    make1 = lambda key: arena1.sample_random_target_pos(key) # anonymous function to get samples for 1 target 
    make2 = lambda key: arena2.sample_random_target_pos(key) # anonymous function to get samples for 2 targets
    
    # create arrays of samples
    # shape (5,1,2)
    samples_target_pos1 = jnp.array([make1(key1), make1(key2), make1(key3), make1(key4), make1(key5)])
    # shape (5,2,2)
    samples_target_pos2 = jnp.array([make2(key1), make2(key2), make2(key3), make2(key4), make2(key5)])

    for i in range(len(keys)):
        assert (samples_target_pos1[i,:,:].shape) == (1,2) # check the shape of target position if 1 target
        assert (samples_target_pos2[i,:,:].shape) == (2,2) # check the shape of target positions if 2 targets
    # for each of the sample arrays, find the unique target arrays and their indeces
    # check that the number of unnique target arrays is the same as the number of keys given
    u1, indices1 = jnp.unique(samples_target_pos1, return_index=True, axis = 0)
    assert (len(indices1)==len(keys))
    u2, indeces2 = jnp.unique(samples_target_pos2, return_index=True, axis = 0)
    assert (len(indeces2)==len(keys))


    
def test_sample_random_sigma():
    """
    Checks that random samples are indeed unique when provided different PRNGkeys and that the returned array
    
    has the correct shape (depending on number of target locations).
    """
    arena1 = Arena(6.0,6.0,1)
    arena2 = Arena(6.0,6.0,2)
    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4, key5 = jax.random.split(key, 5)
    keys = [key1, key2, key3, key4, key5]
    
    make1 = lambda key: arena1.sample_random_sigma(key) # anonymous function to get samples for 1 target 
    make2 = lambda key: arena2.sample_random_sigma(key) # anonymous function to get samples for 2 targets
    
    # create arrays of samples
    samples_sigma1 = jnp.array([make1(key1), make1(key2), make1(key3), make1(key4), make1(key5)])
    samples_sigma2 = jnp.array([make2(key1), make2(key2), make2(key3), make2(key4), make2(key5)])
    
    
    for i in range(len(keys)):
        assert (samples_sigma1[i,:].shape) == (1,) # check the shape of target position if 1 target
        assert (samples_sigma2[i,:].shape) == (2,) # check the shape of target positions if 2 targets
    # for each of the sample arrays, find the unique target arrays and their indeces
    # check that the number of unnique target arrays is the same as the number of keys given
    u1, indices1 = jnp.unique(samples_sigma1, return_index=True, axis = 0)
    assert (len(indices1)==len(keys))
    u2, indeces2 = jnp.unique(samples_sigma2, return_index=True, axis = 0)
    assert (len(indeces2)==len(keys))



def test_sample_random_x_inits():
    """
    Checks that random samples are indeed unique when provided different PRNGkeys and that the returned array
    
    has the correct shape (depending on number of target locations).
    """
    arena1 = Arena(6.0,6.0,1)
    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4, key5 = jax.random.split(key, 5)
    keys = [key1, key2, key3, key4, key5]
    x_init_spread = 1
    N1 = 1
    N2 = 4
    
    make1 = lambda key: arena1.sample_random_x_inits(key, N1, x_init_spread) # anonymous function to get samples for 1 target 
    make2 = lambda key: arena1.sample_random_x_inits(key, N2, x_init_spread)
    samples_x_inits1 = jnp.array([make1(key1), make1(key2), make1(key3), make1(key4), make1(key5)]) # shape (5,1,3)
    samples_x_inits2 = jnp.array([make2(key1), make2(key2), make2(key3), make2(key4), make2(key5)]) # shape (5,4,3)
    
    for i in range(len(keys)):
        assert (samples_x_inits1[i,:,:].shape == (N1, 3)) # assert that returned initial position array is of correct shape
        assert (samples_x_inits2[i,:,:].shape == (N2, 3)) # assert that returned initial positions array is of correct shape
    u1, indices1 = jnp.unique(samples_x_inits1, return_index=True, axis = 0)
    assert (len(indices1)==len(keys))
    u2, indeces2 = jnp.unique(samples_x_inits2, return_index=True, axis = 0)
    assert (len(indeces2)==len(keys))
    

if __name__ == "__main__":
    test_Policy()
    test_get_concentration()
    test_policy_prior_logprob_uniform()
    test_target_pos_prior_logprob()
    test_sigma_prior_logprob()
    test_x_inits_prior_logpob()
    test_sample_random_policy()
    test_sample_random_target_pos()
    test_sample_random_sigma()
    test_sample_random_x_inits()
