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
    Environment_state, #get_sigma, get_concentration, get_target_pos,
    TurtleBotResult
)



@jaxtyped
@beartype
class Game(eqx.Module):
    """
    The turtle bot game engine

    args:
        
        x_init: 
        x_hist:
        conc_hist:
        memory_length:
        duration:
        dt:
        x_init_spread: 
    """
    #TODO: added self before every "policy" & "exogenous_env"
    #exogenous_env: Environment_state #TODO: this was added only to test the issue with predict_and_mitigate
    #policy: Policy #TODO: this was added only to test the issue with predict_and_mitigate
    x_inits: jnp.ndarray#Float[Array, " n "]
    x_hists: jnp.ndarray#Float[Array, " n "]
    conc_hists: jnp.ndarray#Float[Array, "memory_length"]
    memory_length: int
    n_targets: int
    duration: float = 100.0
    dt: float = 0.1
    #x_init_spread = 1 # static variable for x_init spread

    @jaxtyped
    @beartype
    def __call__(
        self,
        
        target_pos: jnp.ndarray,
        sigma: jnp.ndarray,
        x_inits: jnp.ndarray,
        policy: Policy, #TODO: this was removed only to test the issue with predict_and_mitigate
        #exogenous_env: Environment_state, #TODO: this was removed only to test the issue with predict_and_mitigate
        
    ) -> TurtleBotResult:
        """
        Play out a game of turtle bots.

        args:

        """
        T = self.duration
        exogenous_env = Environment_state(target_pos, sigma, x_inits)
        # Define a function to execute one step in the game

        def step_policy(carry, dummy_input): 
            #print (carry, "carry")
            #print (len(carry), "shape of carry")
            #print (jnp.shape(carry[0]), "shape in carry")
            x_hist, conc_hist, x_current = carry
            #print (x_hist.shape, "SHAPE in step_policy")
            control_inputs = jax.nn.tanh(policy(x_hist, conc_hist)) 
            # finding the next state (x, y, theta)
            x_next = dubins_next_state(x_current, control_inputs, actuation_noise = jnp.array([0.0,0.0,0.0]), dt=0.1) 
            jnp.delete(x_hist, -1, 0)
            jnp.delete(conc_hist, -1, 0)
            # insert the next state into the beginning of the list of position history
            jnp.insert(x_hist, 0, x_next, axis=0)
            # insert the newest concentration found at x_next
            jnp.insert(conc_hist, 0, exogenous_env.get_concentration(x_next[0], x_next[1], self.n_targets), axis = 0)
            return (x_hist, conc_hist, x_next), (x_next, control_inputs) 
        
        
# =============================================================================
# 
#         prng_key = jax.random.PRNGKey(0)
#         x_inits = self.x_init_spread * jax.random.uniform(prng_key, shape=(self.N, 3)) 
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
        #print (self.x_hists.shape, "SHAPE")
        test_carry = (self.x_hists, self.conc_hists, self.x_inits)
        #print (jnp.shape(test_carry[0]), "TEST CARRY")
        simulate_scan = lambda x_init, x_hist, conc_hist: jax.lax.scan(step_policy, (x_hist, conc_hist, x_init), xs=None, length=T)[1]
        simulate_batched = jax.vmap(simulate_scan)
        trajectory, control = simulate_batched(self.x_inits, self.x_hists, self.conc_hists)
        control = control**2
        
        
        
                
        def get_squared_distance(trajectory, target: Float[Array, "n 2"]):
            distance = ((trajectory[:,:,0]-target[:,:,0])**2 + (trajectory[:,:,1]-target[:,:,1])**2)
            return distance
        # find a way to find distances without knowing how many targets there are  
        loss = lambda sigma, distance: -1/sigma*jax.lax.exp(-distance/sigma).mean()

        cost = [loss(exogenous_env.get_sigma()[n], get_squared_distance(trajectory, (jnp.zeros_like(trajectory[:,:,:2])+ target_pos[n]))) for n in range(self.n_targets)]
        cost = sum(cost) + control.mean()
        
        


        return TurtleBotResult(
            n_targets=self.n_targets,
            target_pos=exogenous_env.get_target_pos(),
            sigma=exogenous_env.get_sigma(),
            game_duration=self.duration,
            squared_controls=control,
            potential=cost,
            xs=trajectory
        )
