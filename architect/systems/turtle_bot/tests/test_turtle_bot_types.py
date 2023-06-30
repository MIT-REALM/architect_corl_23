#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:24:04 2023

@author: rchanna
"""


import jax
import jax.numpy as jnp

def test_Policy():
def test_Policy():
    # create a policy
    memory_length = 5
    prng_key = jax.random.PRNGKey(1)
    memory_length = 3
    policy = Policy(prng_key, memory_length)
    assert(len(policy.layers)==4) #TODO: what am I trying to test here?
    
    
    conc_hist = 
    policy = Polu
    pass

def test_Environment_states():
    
    
    pass

def test_Environment_states():
    # test case where there is only 1 target. There is only 1 initial condition.
    target_pos = jnp.array([3.0,4.0])
    sigma = jnp.array([0.3])
    prng_key = jax.random.PRNGKey(2)
    x_inits = 1 * jax.random.normal(prng_key, shape=(1, 3)) 
    new_env = Environment_state(target_pos, sigma, x_inits)
    assert(new_env.get_concentration(0.0,0.0) == (1/0.3*jnp.exp(-25/0.3)) ) # based on the distance to source, I've calculated the concentration

    # test case where there is 2 targets. There is 1 initial condition.
    
    
    # test case where there is 1 target. There is N initial conditions.
    
    # test case where there is 2 targets. There is N initial conditions.
def test_Arena():
    pass

def test_TurtleBotResult():
    pass