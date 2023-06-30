#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:55:15 2023

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
    Environment_state,# get_sigma, get_concentration, get_target_pos,
    TurtleBotResult
)



@jaxtyped
@beartype
class Game(eqx.Module):
    """
    The turtle bot game engine

    args:
        policy: provides control inputs
        x_init: 
        x_hist:
        conc_hist:
        duration:
        dt:
    """

    policy: Policy
    x_init: Float[Array, " n "]
    x_hist: Float[Array, " n "]
    conc_hist: Float[Array, " n "]
    memory_length: int
    duration: float = 100.0
    dt: float = 0.1
    x_init_spread = 1 # static variable for x_init spread
    @jaxtyped
    @beartype
    def __call__(
        self,
        policy: Policy,
        x_init: Float[Array, " n "],
        x_hist: Float[Array, " n "],
        conc_hist: Float[Array, " n "], #jaxtyping syntax
    ) -> TurtleBotResult:
        """
        Play out a game of turtle bots.

        args:

        """
        T = self.duration
        # Define a function to execute one step in the game

        def step_policy(carry, dummy_input): 
            x_hist, conc_hist, x_current = carry
            control_inputs = jax.nn.tanh(policy(x_hist, conc_hist)) 
            # finding the next state (x, y, theta)
            x_next = dubins_next_state(x_current, control_inputs, self.dt) 
            jnp.delete(x_hist, -1, 0)
            jnp.delete(conc_hist, -1, 0)
            # insert the next state into the beginning of the list of position history
            jnp.insert(x_hist, 0, x_next, axis=0)
            # insert the newest concentration found at x_next
            jnp.insert(conc_hist, 0, Environment_state.get_concentration(x_next[0], x_next[1]), axis = 0)
            return (x_hist, conc_hist, x_next), (x_next, control_inputs) 
        
        
# =============================================================================
# 
#         prng_key = jax.random.PRNGKey(0)
#         x_inits = self.x_init_spread * jax.random.normal(prng_key, shape=(self.N, 3)) 
#         
#         def make_x_hist(x_init):
#             x_hist = jnp.array([x_init for i in range(self.memory_length)])
#             return x_hist
#         make_x_hists = jax.vmap(make_x_hist)
#         x_hists = make_x_hists(x_inits)
# 
#         def make_conc_hist(x_init):
#             conc = get_concentration(x_init[0], x_init[1])
#             conc_hist = jnp.array([conc for i in range(self.memory_length)])
#             return conc_hist
#         make_conc_hists = jax.vmap(make_conc_hist)
#         conc_hists = make_conc_hists(x_inits)
# 
#         # Execute the game (attempted batched version)
#         def get_squared_distance(trajectory, target: Float[Array, "n 2"]):
#             distance = ((trajectory[:,:,0]-target[:,:,0])**2 + (trajectory[:,:,1]-target[:,:,1])**2+(trajectory[:,:,2]-target[:,:,2])**2)
#             return distance
#         #batched_squared_distance = jax.vmap(get_squared_distance)
#         simulate_scan = lambda x_init, x_hist, conc_hist: jax.lax.scan(step_policy, (x_hist, conc_hist, x_init), xs=None, length=T)[1]
#         simulate_scan = jax.vmap(simulate_scan)
#         loss = lambda sigma, distance: -1/sigma*jax.lax.exp(-distance/sigma).mean()
#         trajectories, controls = simulate_scan(x_inits, x_hists, conc_hists)
#         cost = [loss(get_sigma()[n], get_squared_distance(trajectories, jnp.zeros_like(trajectories)+get_target_pos()[n])) for n in range(get_n_targets())]
#         cost = sum(cost) + controls.mean() # TODO: THIS MIGHT NOT WORK WITH CONTROLS.MEAN()
#         
# =============================================================================
        simulate_scan = lambda x_init, x_hist, conc_hist: jax.lax.scan(step_policy, (x_hist, conc_hist, x_init), xs=None, length=T)[1]
        simulate_scan = jax.vmap(simulate_scan)
        trajectory, control = simulate_scan(x_init, x_hist, conc_hist)
        control = control**2
        
        def get_squared_distance(trajectory, target: Float[Array, "n 2"]):
            distance = ((trajectory[:,:,0]-target[:,:,0])**2 + (trajectory[:,:,1]-target[:,:,1])**2+(trajectory[:,:,2]-target[:,:,2])**2)
            return distance
        
        # find a way to find distances without knowing how many targets there are  
        loss = lambda sigma, distance: -1/sigma*jax.lax.exp(-distance/sigma).mean()
        cost = [loss(get_sigma()[n], get_squared_distance(trajectory, (jnp.zeros_like(trajectory)+ get_target_pos()[n]))) for n in range(Arena.get_n_targets())]
        cost = sum(cost) + control.mean()

        

#TODO: define x_inits as an exogenous parameter

        return TurtleBotResult(
            n_targets=Arena.get_n_targets(),
            target_pos=get_target_pos(),
            sigma=get_sigma(),
            game_duration=self.duration,
            squared_controls=control,
            potential=cost,
            xs=trajectory
        )
