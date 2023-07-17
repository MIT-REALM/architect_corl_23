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
from beartype import beartype
from beartype.typing import List, NamedTuple
from jaxtyping import Array, Float, jaxtyped
from scipy.special import binom
import jax.tree_util as jtu
from architect import utils
from architect.types import PRNGKeyArray



def change_sincos(state_3):
    """ Return the turtle bot's state converted from [x, y, theta] to [x, y, sintheta, costheta] """
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
    
    
    def __init__(self, key, memory_length): 
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.layers = [eqx.nn.Linear(5*memory_length, 64, key=key1), 
                       eqx.nn.Linear(64, 64, key=key2),
                       eqx.nn.Linear(64, 64, key=key3),
                       eqx.nn.Linear(64, 2, key=key4)] 

    def __call__(self, x_hist: jnp.ndarray, conc_hist: jnp.ndarray): 
        new_x_hist = [change_sincos(x) for x in x_hist] #uses dubins dynamics
        new_x_hist = jnp.array(new_x_hist)
        new_x_hist = jnp.ravel(new_x_hist)
        combined_x_conc = jnp.concatenate((new_x_hist, conc_hist))
        for layer in self.layers[:-1]:
            combined_x_conc = jax.nn.relu(layer(combined_x_conc))   
        return self.layers[-1](combined_x_conc)
    
    
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
        """Return n_targets"""
        return self.n_targets
    
    @jaxtyped
    @beartype
    def policy_prior_logprob_gaussian(self, dp: Policy):
        """
        Compute the prior log probability of the given policy. Assumes a gaussian prior distribution.

        Probability is not necessarily normalized.
        
        args:
            dp: design parameter (policy)
        """
        block_logprobs = jtu.tree_map(
            lambda x_updated, x: jax.scipy.stats.norm.logpdf(
                x_updated
            ).mean(),
            dp,
        )
        # Take a block mean rather than the sum of all blocks to avoid a crazy large logprob
        overall_logprob = jax.flatten_util.ravel_pytree(block_logprobs)[0].mean()
        return overall_logprob
    
    @jaxtyped
    @beartype
    def policy_prior_logprob_uniform(self, dp: Policy):
        """
        Return log probability of 0 given a policy. This indicates unifrom distribution and no knowledge
        of the prior distribution.
        
        args:
            dp: design parameter (policy)
        """
        log_prob = 0
        return log_prob
      
    @jaxtyped
    @beartype
    def sample_random_policy(self, key: PRNGKeyArray, memory_length: int) -> Policy:
        """Return random policy"""
        policy = Policy(key, memory_length)
        return policy

    @jaxtyped
    @beartype
    def target_pos_prior_logprob(self, target_pos):
        # TODO: complete the bear typing in utils file for log_smooth_uniform
        """
        Compute the prior log probability of a given array of target position(s). Assumes a uniform prior distribution.
        
        Probability is not necessarily normalized.
        
        args:
            target_pos: the given jnp array of target position(s)
        """
        smoothing = self.smoothing
        x_min, x_max = -self.width / 2.0, self.width / 2.0
        y_min, y_max = -self.height / 2.0, self.height / 2.0
        
        logprob_x = utils.log_smooth_uniform(target_pos[:, 0], x_min, x_max, smoothing).sum()
        logprob_y = utils.log_smooth_uniform(target_pos[:, 1], y_min, y_max, smoothing).sum()
        return logprob_x + logprob_y    
    
    @jaxtyped
    @beartype
    # TODO: should I be using separate x_keys and y_keys for each of the n targets?
    def sample_random_target_pos(self, key: PRNGKeyArray):
        """
        Sample a random target position. Returns a 2D array of shape (n_targets, 2).
        
        args:
            key: PRNG key to use for sampling
        """
        x_min, x_max = -self.width / 2.0, self.width / 2.0
        y_min, y_max = -self.height / 2.0, self.height / 2.0
        x_key, y_key = jrandom.split(key)
        x = jrandom.uniform(x_key, shape=(self.n_targets, 1), minval=x_min, maxval=x_max)
        y = jrandom.uniform(y_key, shape=(self.n_targets, 1), minval=y_min, maxval=y_max)
        pos = jnp.hstack((x, y))
        return pos
    
    @jaxtyped
    @beartype
    def sigma_prior_logprob(self, sigma):
        """
        Compute the prior log probability of a given array of sigma value(s). Assumes a uniform prior distribution.
        
        The sigma value is used in computing the concentation of a leak that behaves like a Gaussian distribution.
        
        Probability is not necessarily normalized.
        
        args:
            sigma: the given jnp array of sigma value(s)
        """
        sigma_min = 0.1
        sigma_max = 1.0
        smoothing = self.smoothing
        ##adding
        log_probs = []
        for i in range (self.n_targets):
            logprob_sigma = utils.log_smooth_uniform(sigma[i], sigma_min, sigma_max, smoothing)
            log_probs.append(logprob_sigma)
        logprob_sigma = sum(log_probs)
        #done adding
        #logprob_sigma = utils.log_smooth_uniform(sigma, sigma_min, sigma_max, smoothing)
        return logprob_sigma

    @jaxtyped
    @beartype
    def sample_random_sigma(self, key: PRNGKeyArray):
        """
        Sample a random jnp array of sigma value(s). Returns a 1D array of shape (n_targets).
        
        args:
            key: PRNG key to use for sampling
        """
        random_sigma = jrandom.uniform(key, shape = (self.n_targets,), minval=0.1, maxval=1.0)
        return random_sigma
    
    @jaxtyped
    @beartype
    def x_inits_prior_logprob(self, x_inits, x_init_spread):
        """
        Compute the prior log probability of a given array of x_init value(s). Assumes a uniform prior distribution.
        
        Probability is not necessarily normalized.
        
        args:
            x_inits: the given jnp array of inital position value(s)
        """
        
        
        smoothing = self.smoothing
        x_min = y_min = theta_min = -1*x_init_spread
        x_max = y_max = theta_max = 1*x_init_spread
        
        logprob_x = utils.log_smooth_uniform(x_inits[:, 0], x_min, x_max, smoothing).sum()
        logprob_y = utils.log_smooth_uniform(x_inits[:, 1], y_min, y_max, smoothing).sum()
        logprob_theta = utils.log_smooth_uniform(x_inits[:, 2], theta_min, theta_max, smoothing).sum()

        return logprob_x + logprob_y + logprob_theta
    
    @jaxtyped
    @beartype
    def sample_random_x_inits(self, key: PRNGKeyArray, N, x_init_spread):
        """
        Sample a random jnp array of x_init value(s). 
        
        args:
            key: PRNG key to use for sampling
            N: number of initial conditions
            x_init_spread: range of initial positions
        """
        x_inits = x_init_spread * jax.random.uniform(key, shape=(N, 3), minval=-1, maxval=1)
        return x_inits


class Environment_state(NamedTuple):
    """
    The collection of exogenous parameters.
    
    There can be up to 2 sources of a Gaussian form. The concentration of a leak
    follows 1/sigma * e^-(distance^2/sigma) with distance being the distance to each source 
    and sigma influencing the concentration strength. 
    
    args:
        target_pos: a 2D jnp array of target position(s) with shape (arena.n_targets, 2)
        sigma: a 1D jnp array of sigma value(s) used to calculate concentration; has shape (arena.n_targets)
        x_inits: a jnp array of a turtle bot's initial ([x, y, theta]) positions
                 Note: x_inits has shape (N, 3) where N is an argument provided in solve_turtle_bot.py
        arena: a Arena object in which the game is set up
    """
    
    target_pos: Float[Array, "n 2"]
    sigma: Float[Array, " n"] 
    x_inits: jnp.ndarray

    def get_x_inits(self):
        return self.x_inits
        
    def get_target_pos(self):
        return self.target_pos

    def get_sigma(self):
        return self.sigma
    
    def get_concentration(self, x: float, y: float, num_targets: int):
        concentration: float = 0
        targets = self.target_pos
        sigmas = self.sigma
        make_gaussian = lambda target, sig: 1/sig*jax.lax.exp(-1/sig* ((target[0]-x)**2 + (target[1]-y)**2))
        concs = [make_gaussian(targets[n], sigmas[n]) for n in range(num_targets)]
        concentration = sum(concs)  
        concentration = jnp.take(concentration, 0)
        return concentration
  
    

class TurtleBotResult(NamedTuple):
    """
    Result of playing a game of turtle bots

    Includes:
        Exogenous Parameters
        n_targets: number of sources in the environment
        target_pos: positions of source(s)
        sigma: measure of concentration/strength of leaks at each source
        
         
        game_duration: duration of the game
        squared_controls: control inputs provided by policy
        potential: cost for the turtle bot
        
        xs: trajectory of the turtle bot
        
    """

    
    #exogenous:
    n_targets: int
    target_pos: Float[Array, "n 2"]
    sigma: Float[Array, " n"]
    
    #design:
    game_duration: float
    squared_controls: Float[Array, "n 2"] #jnp.array of v,w
    potential: Float[Array, ""]
    #turtle_positions:
    xs: jnp.array
