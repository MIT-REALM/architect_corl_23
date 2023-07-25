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
    EnvironmentState, 
    TurtleBotResult
)



@jaxtyped
@beartype
class Game(eqx.Module):
    """
    The turtle bot game engine

    args:
        x_inits: turtle-bot's starting positions
        x_hists: array of turtle-bot's past positions
        conc_hists: array of turtle-bot's past concentration values
        memory_length: number of past states remembered
        duration: length of game (seconds)
        dt: timestep of the game's simulation (seconds)
    """

    # x_inits: jnp.ndarray
    # x_hists: jnp.ndarray
    # conc_hists: jnp.ndarray
    memory_length: int
    n_targets: int
    duration: float = 100.0
    dt: float = 0.1

    @jaxtyped
    @beartype
    def __call__(
        self,
        
        target_pos: jnp.ndarray,
        sigma: jnp.ndarray,
        x_inits: jnp.ndarray,
        policy: Policy, 
        
    ) -> TurtleBotResult:
        """
        Play out a game of turtle bots.

        args:
            target_pos: the source(s) of the leak(s)
            sigma: array of values representing the concentration "strength" of a leak
            x_inits: turtle-bot's starting positions
            policy: neural network policy used to for control input

        """
        esigma = jnp.exp(sigma)
        T = self.duration
        exogenous_env = EnvironmentState(target_pos, esigma, x_inits)
        
        # initialize x_hists
        def make_x_hist(x_init):
            x_hist = jnp.array([x_init for i in range(self.memory_length)])
            return x_hist
        make_x_hists = jax.vmap(make_x_hist)
        x_hists = make_x_hists(x_inits)

         # initialize conc_hists
        def make_conc_hist(x_init):#, arena):
            conc = exogenous_env.get_concentration(x_init[0], x_init[1], self.n_targets)
            conc_hist = jnp.array([conc for i in range(self.memory_length)])
            return conc_hist
        make_conc_hists = jax.vmap(make_conc_hist)
        conc_hists = make_conc_hists(x_inits)
     
       
        # Define a function to execute one step in the game

        def step_policy(carry, dummy_input): 
            x_hist, conc_hist, x_current = carry
            control_inputs = jax.nn.tanh(policy(x_hist, conc_hist)) 
            # finding the next state (x, y, theta)
            x_next = dubins_next_state(x_current, control_inputs, actuation_noise = jnp.array([0.0,0.0,0.0]), dt=0.1) 
            #jax.debug.print("x_hist before removing {x_hist}", x_hist = x_hist)
            x_hist = jnp.delete(x_hist, -1, 0)
            #jax.debug.print("x_hist after removing {x_hist}", x_hist = x_hist)
            conc_hist = jnp.delete(conc_hist, -1, 0)
            # insert the next state into the beginning of the list of position history
            x_hist = jnp.insert(x_hist, 0, x_next, axis=0)
            #jax.debug.print("x_hist after adding {x_hist}", x_hist = x_hist)
            # insert the newest concentration found at x_next
            #jax.debug.print("control_inputs {control_inputs}", control_inputs = control_inputs)
            conc_hist = jnp.insert(conc_hist, 0, exogenous_env.get_concentration(x_next[0], x_next[1], self.n_targets), axis = 0)
            return (x_hist, conc_hist, x_next), (x_next, control_inputs) 
        
        
        simulate_scan = lambda x_init, x_hist, conc_hist: jax.lax.scan(step_policy, (x_hist, conc_hist, x_init), xs=None, length=T)[1]
        simulate_batched = jax.vmap(simulate_scan)
        trajectory, control = simulate_batched(x_inits, x_hists, conc_hists)
        control = control**2
        
        
        def get_squared_distance(trajectory, target: Float[Array, "n 2"]):
            distance = ((trajectory[:,:,0]-target[:,:,0])**2 + (trajectory[:,:,1]-target[:,:,1])**2)
            return distance
        # find distances without knowing how many targets there are  
        loss = lambda sigma, distance: -1/sigma*jax.lax.exp(-distance/sigma).mean()
        cost = [loss(esigma[n], get_squared_distance(trajectory, (jnp.zeros_like(trajectory[:,:,:2])+ target_pos[n]))) for n in range(self.n_targets)]
        cost = sum(cost) + control.mean()
        

        return TurtleBotResult(
            n_targets=self.n_targets,
            target_pos=exogenous_env.get_target_pos(),
            sigma=esigma,
            game_duration=self.duration,
            squared_controls=control,
            potential=cost,
            xs=trajectory
        )
