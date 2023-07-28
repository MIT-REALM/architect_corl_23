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
    """
    Tests the Game class in turtle_bot_game. 
    Computes the result of two games called with different initial conditions and/or memory lengths.
    Checks that the resulting controls, potential (loss), and trajectories have the correct shapes.
    If plotting is enabled, the scenarios will be plotted on a contour plot.
    """
    # Run this function if plots are requested
    def make_plot(result, environment, num_targets):    
        def get_concentration(x: float, y: float):
            concentration: float = 0
            targets = environment.get_target_pos()
            logsigma = environment.get_logsigma()
            sigmas = jnp.exp(logsigma)
        #     make_gaussian = lambda target, sigma: jax.scipy.stats.norm.pdf(
        #     ((target[0] - x) ** 2 + (target[1] - y) ** 2), target, sigma
        # )
            make_gaussian = lambda target, sig: 1/sig*jax.lax.exp(-1/sig* ((target[0]-x)**2 + (target[1]-y)**2))
            concs = [make_gaussian(targets[n], sigmas[n]) for n in range(num_targets)]
            concentration = sum(concs)
            concentration = jnp.take(concentration, 0)
            return concentration
        
        # def dummy_fn(x, y):
        #     return environment.get_concentration(x, y, n_targets)
        
        delta = 0.025
        
        x = jnp.arange(-3.0, 3.0, delta)
        y = jnp.arange(-2.0, 2.0, delta)
        X, Y = jnp.meshgrid(x, y)
       
        
        data = (jax.vmap(jax.vmap(get_concentration))(X,Y))
        # data = jax.vmap(jax.vmap(dummy_fn))(X,Y)
        #data = (jax.vmap(environment.get_concentration, in_axes = (0,0,None)))(X,Y,1)
        
        plt.figure()
        plt.contour(X,Y,data)
        for i in range(N):
            plt.plot(result.xs[i,:,0], result.xs[i,:,1])
            plt.plot(result.xs[i,0,0], result.xs[i,0,1], "bo")
            plt.plot(result.xs[i,-1,0], result.xs[i,-1,1], "ro")
        label = f"Testing the game with: n_targets = {num_targets}, memory_length = {memory_length}"
        plt.title(label)



    
    # For a memory length of 1 and 1 target position
    # Create policy
    memory_length = 3
    prng_key = jax.random.PRNGKey(1)
    prng_key, subkey = jax.random.split(prng_key)
    policy = Policy(subkey,memory_length)
    
    # Create x_inits
    N=20
    prng_key = jax.random.PRNGKey(0)
    x_inits = 1 * jax.random.normal(prng_key, shape=(N, 3))#, minval=-1.0, maxval=1.0) 
    
    
    # Create the game
    game = Game(
        memory_length = 3,
        n_targets = 2,
        duration = 500.0,
        dt = 0.1
    )
    
    # Define test target position
    target_pos = jnp.array([[-1.5,0.0],[0.5,0.0]])
    num_targets = target_pos.shape[0]
    # Define test sigma
    logsigma = jnp.log(jnp.array([0.435, 0.435]))
    
    env = EnvironmentState(target_pos, logsigma, x_inits)
    # Evaluate the game
    result = game(target_pos, logsigma, x_inits, policy)
    assert result.squared_controls.shape == (N, game.duration, 2)
    #assert result.potential.shape == ()
    assert result.xs.shape == (N, game.duration, 3)
    assert jnp.all(result.sigma>0)
    import pdb
    pdb.set_trace()
    if plot:
        make_plot(result, env, num_targets)

    
    # For a memory_length of 4 and 2 target positions
    # Create policy
    memory_length = 4
    prng_key = jax.random.PRNGKey(2)
    prng_key, subkey = jax.random.split(prng_key)
    policy = Policy(subkey,memory_length)
    
    # Create x_inits
    N=20
    prng_key = jax.random.PRNGKey(0)
    x_inits = 2 * jax.random.uniform(prng_key, shape=(N, 3), minval=-1.0, maxval=1.0) 

    
    # Create the game
    game = Game(
        memory_length = 4,
        n_targets = 1,
        duration = 500.0,
        dt = 0.1
    )

    # Define test target position
    target_pos = jnp.array([[1.0,0.5], [-2.0,-.1]])
    num_targets = target_pos.shape[0]
    # Define test sigma
    logsigma = jnp.log(jnp.array([0.6, 0.8]))
    
    env = EnvironmentState(target_pos, logsigma, x_inits)

    # Evaluate the game
    result = game(target_pos, logsigma, x_inits, policy)
    jax.debug.print("result.squared_controls {result.squared_controls}", result = result)
    assert result.squared_controls.shape == (N, game.duration, 2)
    #assert result.potential.shape == ()
    assert result.xs.shape == (N, game.duration, 3)
    assert jnp.all(result.sigma>0)
    if plot:
        make_plot(result, env, num_targets)

    
    plt.show()


if __name__ == "__main__":
    test_Game(plot=True)
