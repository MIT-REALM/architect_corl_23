#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:05:18 2023

@author: rchanna
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import time
from beartype import beartype
from jax.nn import logsumexp
from jaxtyping import Array, Float, jaxtyped
from architect.systems.components.dynamics.dubins import dubins_next_state
from architect.systems.turtle_bot.turtle_bot_types import (
    Arena,
    Policy,
    EnvironmentState, 
    TurtleBotResult
)
from architect.systems.turtle_bot.turtle_bot_game import Game
import matplotlib.pyplot as plt

def test_Game(plot=False):
    
    # Run this function if plots are requested
    def make_plot(result):    
        def get_concentration(x: float, y: float):
            concentration: float = 0
            targets = target_pos
            sigmas = sigma
            make_gaussian = lambda target, sig: 1/sig*jax.lax.exp(-1/sig* ((target[0]-x)**2 + (target[1]-y)**2))
            concs = [make_gaussian(targets[n], sigmas[n]) for n in range(num_targets)]
            concentration = sum(concs)
            concentration = jnp.take(concentration, 0)
            return concentration
                
        delta = 0.025
        
        x = jnp.arange(-3.0, 3.0, delta)
        y = jnp.arange(-2.0, 2.0, delta)
        X, Y = jnp.meshgrid(x, y)
                
        data = jax.vmap(jax.vmap(get_concentration))(X,Y)
        plt.figure()
        plt.contour(X,Y, data)
        for i in range(N):
            plt.plot(result.xs[i,:,0], result.xs[i,:,1])
            plt.plot(result.xs[i,0,0], result.xs[i,0,1], "bo")
            plt.plot(result.xs[i,-1,0], result.xs[i,-1,1], "ro")
        label = f"Testing the game with: n_targets = {num_targets}, memory_length = {memory_length}"
        plt.title(label)



    
    # For a memory length of 1 and 1 target position
    # Create policy
    memory_length = 1
    prng_key = jax.random.PRNGKey(1)
    prng_key, subkey = jax.random.split(prng_key)
    policy = Policy(subkey,memory_length)
    
    # Create x_inits
    N=20
    prng_key = jax.random.PRNGKey(0)
    x_inits = 1 * jax.random.uniform(prng_key, shape=(N, 3), minval=-1.0, maxval=1.0) 
    
    # Create x_hists
    def make_x_hist(x_init):
        x_hist = jnp.array([x_init for i in range(memory_length)])
        return x_hist
    make_x_hists = jax.vmap(make_x_hist)
    x_hists = make_x_hists(x_inits)
    
    # Create conc_hists
    def make_conc_hist(x_init):
        conc = 0.05*x_init[0]
        conc_hist = jnp.array([conc for i in range(memory_length)])
        return conc_hist
    make_conc_hists = jax.vmap(make_conc_hist)
    conc_hists = make_conc_hists(x_inits)
    
    # Create the game
    game = Game(
        memory_length = 1,
        n_targets = 1,
        duration = 200.0,
        dt = 0.1
    )

    # Define test target position
    target_pos = jnp.array([[0.0,0.5]])
    num_targets = target_pos.shape[0]
    # Define test sigma
    sigma = jnp.array([0.6])
    # Evaluate the game
    result = game(target_pos, sigma, x_inits, policy)
    if plot:
        make_plot(result)
    
    
    
    
    # For a memory_length of 4 and 2 target positions
    # Create policy
    memory_length = 4
    prng_key = jax.random.PRNGKey(1)
    prng_key, subkey = jax.random.split(prng_key)
    policy = Policy(subkey,memory_length)
    
    # Create x_inits
    N=20
    prng_key = jax.random.PRNGKey(0)
    x_inits = 2 * jax.random.uniform(prng_key, shape=(N, 3), minval=-1.0, maxval=1.0) 
    
    # Create x_hists
    def make_x_hist(x_init):
        x_hist = jnp.array([x_init for i in range(memory_length)])
        return x_hist
    make_x_hists = jax.vmap(make_x_hist)
    x_hists = make_x_hists(x_inits)
    
    # Create conc_hists
    def make_conc_hist(x_init):
        conc = 0.05*x_init[0]
        conc_hist = jnp.array([conc for i in range(memory_length)])
        return conc_hist
    make_conc_hists = jax.vmap(make_conc_hist)
    conc_hists = make_conc_hists(x_inits)
    
    # Create the game
    game = Game(
        memory_length = 4,
        n_targets = 1,
        duration = 200.0,
        dt = 0.1
    )

    # Define test target position
    target_pos = jnp.array([[1.0,0.5], [-2.0,-.1]])
    num_targets = target_pos.shape[0]
    # Define test sigma
    sigma = jnp.array([0.6, 0.8])
    # Evaluate the game
    result = game(target_pos, sigma, x_inits, policy)
    if plot:
        make_plot(result)
    
    
    plt.show()


if __name__ == "__main__":
    test_Game(plot=True)
