#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:28:49 2023

@author: rchanna
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from beartype import beartype
from beartype.typing import List, NamedTuple
from jax.nn import log_sigmoid
from jaxtyping import Array, Float, jaxtyped
from scipy.special import binom
import jax.tree_util as jtu

from architect.types import PRNGKeyArray

def change_sincos(state_3):
    x = state_3[0]
    y = state_3[1]
    theta = state_3[2]
    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)
    state_4 = jnp.array([x, y, sintheta, costheta])
    return state_4

# design parameters
class Policy(eqx.Module):
    """
    Design parameters (Neural Network)
    """

    layers: list
    memory_length: int
    
    def __init__(self, key, memory_length): 
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.layers = [eqx.nn.Linear(5*memory_length, 64, key=key1), 
                       eqx.nn.Linear(64, 64, key=key3),
                       eqx.nn.Linear(64, 2, key=key4)] 

    def __call__(self, 
                 x_hist: jnp.ndarray,
                 conc_hist: jnp.ndarray): 
        x_hist: jnp.ndarray # jnp arrays of jnp arrays
        conc_hist: jnp.ndarray# conc_hist is a 1D jnp array
        new_x_hist = [change_sincos(x) for x in x_hist] #uses dubins dynamics
        new_x_hist = jnp.array(new_x_hist)
        new_x_hist = jnp.ravel(new_x_hist)
        combined_x_conc = jnp.concatenate((new_x_hist, conc_hist))
        for layer in self.layers[:-1]:
            combined_x_conc = jax.nn.relu(layer(combined_x_conc))   
        return self.layers[-1](combined_x_conc)
    
# exogeneous parameters
class Environment_state(NamedTuple):
    # disturbances? what would they be?
    """
    There can be up to 2 sources of a Gaussian form. The concentration of a leak
    follows 1/sigma * e^-(distance^2/sigma) with distance being the distance to each source 
    and sigma influencing the concentration strength. 
    """

    target_pos: Float[Array, "n 2"]
    sigma: Float[Array, " n"]
    x_inits: jnp.ndarray

  #  def get_x_inits(self):
       # return x_inits
    
    def get_concentration(self, x: float, y: float):
        concentration: float = 0
        make_gaussian = lambda target: 1/self.sigma*jax.lax.exp(-1/self.sigma* ((target[0]-x)**2 + (target[1]-y)**2))
        concs = [make_gaussian(self.target_pos[n]) for n in range(Arena.get_n_targets())]
        concentration = sum(concs)                         
        return concentration
        
    def get_target_pos(self):
        return self.target_pos

    def get_sigma(self):
        return self.sigma
  
    
@jaxtyped
@beartype
class Arena(eqx.Module):
    """
    The arena in which hide and seek takes place

    args:
        width: width of the arena (m)
        height: height of the arena (m)
        n_targets: number of targets/sources to be found in the environment
        smoothing: higher values -> sharper approximation of a uniform distribution
    """

    width: float
    height: float
    n_targets: int
    smoothing: float = 20.0
    def get_n_targets(self):
        return self.n_targets
    def policy_prior_logprob(dp, initial_dp: Policy):
        block_logprobs = jtu.tree_map(
            lambda x_updated, x: jax.scipy.stats.norm.logpdf(
                x_updated
            ).mean(),
            dp,
            initial_dp,
        )
        # Take a block mean rather than the sum of all blocks to avoid a crazy large
        # logprob
        overall_logprob = jax.flatten_util.ravel_pytree(block_logprobs)[0].mean()
        return overall_logprob
      
    def sample_random_policy(self, key: PRNGKeyArray, memory_length: int) -> Policy:
        policy = Policy(key, memory_length)
        return policy
        #pass


#Provided n_targets as a static variable 
# =============================================================================
# 
# 
#     def n_targets_prior_logprob(self, n_targets: int):
#         
#         def log_smooth_uniform(x, x_min, x_max):
#             return log_sigmoid(self.smoothing * (x - x_min)) + log_sigmoid(
#                 self.smoothing * (x_max - x)
#             )
#         
#         n_min = 1
#         n_max = 2
#         logprob_n = log_smooth_uniform(n_targets, n_min, n_max)
#         return logprob_n
# 
#   # specify as static parameter  
#     def sample_random_n_targets(self, key): #TODO: using a key will always get the same number...isn't that bad?
#         new_key, sub_key = jrandom.split(key)
#         n_targets = jrandom.randint(sub_key, shape = (), minval = 1, maxval = 3)
#         return n_targets
#         #pass
# 
# 
# =============================================================================

    def target_pos_prior_logprob(self, target_pos):
        # TODO: can make part of utils file
        def log_smooth_uniform(x, x_min, x_max):
            return log_sigmoid(self.smoothing * (x - x_min)) + log_sigmoid(
                self.smoothing * (x_max - x)
            )
        
        x_min, x_max = -self.width / 2.0, self.width / 2.0
        y_min, y_max = -self.height / 2.0, self.height / 2.0
        
        logprob_x = log_smooth_uniform(target_pos[:, 0], x_min, x_max).sum()
        logprob_y = log_smooth_uniform(target_pos[:, 1], y_min, y_max).sum()
        return logprob_x + logprob_y    
    
    # TODO: should I be using separate x_keys and y_keys for each of the n targets?
    def sample_random_target_pos(self, key: PRNGKeyArray):
        x_min, x_max = -self.width / 2.0, self.width / 2.0
        y_min, y_max = -self.height / 2.0, self.height / 2.0
        x_key, y_key = jrandom.split(key)
        x = jrandom.uniform(x_key, shape=(self.n_targets, 1), minval=x_min, maxval=x_max)
        y = jrandom.uniform(y_key, shape=(self.n_targets, 1), minval=y_min, maxval=y_max)
        pos = jnp.hstack((x, y))
        return pos
    
    def sigma_prior_logpob(self, sigma):
        def log_smooth_uniform(x, x_min, x_max):
            return log_sigmoid(self.smoothing * (x - x_min)) + log_sigmoid(
                self.smoothing * (x_max - x)
            )
        
        sigma_min = 0.1
        sigma_max = 1.0
        logprob_sigma = log_smooth_uniform(sigma, sigma_min, sigma_max)
        return logprob_sigma
    

    def sample_random_sigma(self, key):
        #key1, sub_key = jrandom.split(key)
        random_sigma = jrandom.uniform(key, shape = (self.n_targets,), minval=0.1, maxval=1.0)
        return random_sigma
    


class TurtleBotResult(NamedTuple):
    """
    Result of playing a game of turtle bots

    Includes:
        
    """

    
    #exogenous:
    n_targets: int
    target_pos: Float[Array, "n 2"]
    sigma: Float[Array, " n "]
    
    #design:
    game_duration: float
    squared_controls: Float[Array, "n 2"] #jnp.array of v,w
    potential: Float[Array, ""]
    #turtle_positions:
    xs: jnp.array
