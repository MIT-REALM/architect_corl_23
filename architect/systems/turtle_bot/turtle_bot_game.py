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
    TurtleBotResult,
)


@jaxtyped
@beartype
class Game(eqx.Module):
    """
    The turtle bot game engine

    args:
        memory_length: number of past states remembered
        n_targets: number of sources/targets in the environment
        duration: length of game (seconds)
        dt: timestep of the game's simulation (seconds)
    """

    memory_length: int
    n_targets: int
    duration: float = 100.0
    dt: float = 0.1

    @jaxtyped
    @beartype
    def __call__(
        self,
        target_pos: Float[Array, "n_targets 2"],
        logsigma: Float[Array, "n_targets "],
        x_inits: Float[Array, "N 3"],
        policy: Policy,
    ) -> TurtleBotResult:
        """
        Play out a game of turtle bots.
        
        x_hist is an array with memory_length-past positions (x, y, theta)
        conc_hist is an array with memory_length-past concentrations at corresponding past positions
        
        Computes first x_hist and conc_hist are computed for time t=0.
        At each step:
            Evaluates the policy using x_hist and conc_hist as inputs to determine the controls
            Updates x_hist and conc_hist (oldest entry removed and newest entry inserted)
        Calculates loss (to be minimized). Loss is defined as the negative concentration 
        (which is minimized as distance to source decreases) plus the controls 
        (to disincentivize excessive movement).
        
        args:
            target_pos: the source(s) of the leak(s)
            sigma: array of values representing the concentration "strength" of a leak
            x_inits: turtle-bot's starting positions
            policy: neural network policy used to for control input

        """
        e_sigma = jnp.exp(logsigma)
        T = self.duration
        exogenous_env = EnvironmentState(target_pos, e_sigma, x_inits)

        # initialize x_hists
        def make_x_hist(x_init: Float[Array, "N 3"]) -> Float[Array, "N memory_length 3"]:
            x_hist = jnp.array([x_init for i in range(self.memory_length)])
            return x_hist
        make_x_hists = jax.vmap(make_x_hist)
        x_hists = make_x_hists(x_inits)
        # initialize conc_hists
        def make_conc_hist(x_init: Float[Array, "N 3"]) -> Float[Array, "N memory_length"]:
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
            x_next = dubins_next_state(
                x_current,
                control_inputs,
                actuation_noise=jnp.array([0.0, 0.0, 0.0]),
                dt=0.1,
            )
            x_hist = jnp.delete(x_hist, -1, 0)
            conc_hist = jnp.delete(conc_hist, -1, 0)
            # insert the next state into the beginning of the list of position history
            x_hist = jnp.insert(x_hist, 0, x_next, axis=0)
            # insert the newest concentration found at x_next
            conc_hist = jnp.insert(
                conc_hist,
                0,
                exogenous_env.get_concentration(x_next[0], x_next[1], self.n_targets),
                axis=0,
            )
            return (x_hist, conc_hist, x_next), (x_next, control_inputs)

        simulate_scan = lambda x_init, x_hist, conc_hist: jax.lax.scan(
            step_policy, (x_hist, conc_hist, x_init), xs=None, length=T
        )[1]
        simulate_batched = jax.vmap(simulate_scan)
        trajectory, control = simulate_batched(x_inits, x_hists, conc_hists)
        control = control**2

        def get_squared_distance(trajectory, target: Float[Array, "n 2"]):
            distance = (trajectory[:, :, 0] - target[:, :, 0]) ** 2 + (
                trajectory[:, :, 1] - target[:, :, 1]
            ) ** 2
            return (distance)


        # find distances without knowing how many targets there are
        loss = (
            lambda sigma, distance: -1 / sigma * jax.lax.exp(-distance / sigma).mean()
        )

        cost = [
            loss(
                e_sigma[n],
                get_squared_distance(
                    trajectory, (jnp.zeros_like(trajectory[:, :, :2]) + target_pos[n])
                ),
            )
            for n in range(self.n_targets)
        ]
        a = [get_squared_distance(trajectory, (jnp.zeros_like(trajectory[:,:,:2])+target_pos[n])) for n in range(self.n_targets)]
        #jax.debug.print("a, {a}", a=a)
        #cost = jnp.array(cost)
        jax.debug.print("cost, {cost}", cost = cost)
        #jax.debug.print("a {a}", a=a)
        cost = sum(cost) + control.mean()
        #jax.debug.print("cost, {cost}", cost = cost)

        
        return TurtleBotResult(
            n_targets=self.n_targets,
            target_pos=exogenous_env.get_target_pos(),
            sigma=e_sigma,
            game_duration=self.duration,
            squared_controls=control,
            potential=cost,
            xs=trajectory,
        )
