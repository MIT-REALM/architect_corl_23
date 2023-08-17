import argparse
import json
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, NamedTuple
from jax.nn import logsumexp
from jaxtyping import Array, Float, jaxtyped

from architect.systems.hide_and_seek.hide_and_seek_types import (
    Trajectory2D,
    Arena
)
from architect.systems.formation.formation import (
    make_adj_matrix,
    init_bot_positions,
    current_dist_angle,
    velocity_vectors
)


@jax.jit
def softnorm(x):
    """Compute the 2-norm, but if x is too small replace it with the squared 2-norm
    to make sure it's differentiable. This function is continuous and has a derivative
    that is defined everywhere, but its derivative is discontinuous.
    From hide_and_seek.py
    """
    eps = 1e-5
    scaled_square = lambda x: (eps * (x / eps) ** 2).sum()
    return jax.lax.cond(jnp.linalg.norm(x) >= eps, jnp.linalg.norm, scaled_square, x)


def softmax(x: Float[Array, "..."], sharpness: float = 0.05):
    """Return the soft maximum of the given vector"""
    return 1 / sharpness * logsumexp(sharpness * x)


def algebraic_connectivity( # from formation2D/simulator.py
    positions: Float[Array, "n 2"], communication_range: float, sharpness: float = 20.0
) -> Float[Array, ""]:
    """Compute the connectivity of a formation of drones"""
    # Get the pairwise distance between drones
    squared_distance_matrix = jnp.sum(
        (positions[:, None, :] - positions[None, :, :]) ** 2, axis=-1
    )

    # Compute adjacency and degree matrices (following Cavorsi RSS 2023)
    adjacency_matrix = jax.nn.sigmoid(
        sharpness * (communication_range**2 - squared_distance_matrix)
    )
    degree_matrix = jnp.diag(jnp.sum(adjacency_matrix, axis=1))

    # Compute the laplacian
    laplacian = degree_matrix - adjacency_matrix

    # Compute the connectivity (the second smallest eigenvalue of the laplacian)
    connectivity = jnp.linalg.eigvalsh(laplacian)[1]

    return connectivity


# ----- SIMULATOR -----


class FormationResult(NamedTuple):
    """
    The result of a formation simulation

    args:
        positions: the positions of the drones over time
        potential: the potential/cost assigned to this rollout
        connectivity: the connectivity of the formation over time
    """

    positions: Float[Array, "T n 2"]
    potential: Float[Array, " "]
    connectivity: Float[Array, "T"]


def simulate(
        n,
        formation_shape,
        desired_r_theta,
        duration,
        goal_position,
        dt = 0.1,
        bot_max_speed = 0.5,
        b = 100.0,
        communication_range = 0.5,
) -> FormationResult:
    '''
    Simulator engine

    args:
    * duration: the length of the game (seconds)
    * dt: the timestep of the game's simulation (seconds)
    * bot_max_speed: max speed of bots (m/s)
    * b: parameter used to smooth min/max
    * n: number of bots
    * initial_bot_positions: (n,2) array of initial bot location coordinates
    * formation_shape: 'all-follow-leader', 'single-chain', 'v-formation'
    * adj_matrix: info about which bot follows which
    * desired_r_theta: desired distance-angle btwn bots in the formation
    '''
    adj_matrix = make_adj_matrix(n, formation_shape)
    # Note: make a function that lets you customize the desired distance-angles

    def formation_adjustment(bot_positions, desired_r_theta):
        '''
        Returns the velocity vectors that arranges the follower bots into the formation we want.
        '''
        current_r_theta = current_dist_angle(bot_positions, adj_matrix)
        return velocity_vectors(current_r_theta, desired_r_theta, adj_matrix)

    def step(carry, t):
        '''
        Executes one timestep of the simulation.
        * carry: current_bot_positions
        '''
        # Unpack carry - The 0th leader will always be the global leader
        current_bot_positions = carry
        current_leader_position = current_bot_positions[0] # (1,2) array

        # << Leader >>
        # The point along the trajectory that the leader should move towards
        leader_target = leader_trajectory(t / duration)
        # Steer the leader towards leader_target + clip with max speeds
        v_leader = leader_target - current_leader_position
        leader_speed = softnorm(v_leader)
        v_leader = jnp.where(
            leader_speed >= bot_max_speed,
            v_leader * bot_max_speed / (1e-3 + leader_speed),
            v_leader
        )
        # Calculate new position
        leader_position = current_leader_position + dt * v_leader

        # << Followers >>
        # Steer followers to match the formation + clip with max speeds
        v_follower = formation_adjustment(current_bot_positions, desired_r_theta)
        follower_speed = softnorm(v_follower)
        v_follower = jnp.where(
            follower_speed >= bot_max_speed,
            v_follower * bot_max_speed / (1e-3 + follower_speed),
            v_follower
        )
        # Calculate new position
        current_follower_positions = current_bot_positions[1:,]
        follower_positions = current_follower_positions + dt * v_follower

        bot_positions = jnp.concatenate((jnp.array([leader_position]), follower_positions), axis=0)

        # Cost function for global leader

        # Cost function
        # formation_error = abs(desired_r_theta - current_dist_angle(current_bot_positions, adj_matrix))

        output = bot_positions
        return bot_positions, output
    
    # Define the initial carry and inputs
    carry = init_bot_positions(n)

    # Run the loop
    _, qs = jax.lax.scan(step, carry, jnp.arange(0.0, duration, dt))

    # import pdb; pdb.set_trace()

    # Compute the potential based on the minimum connectivity of the formation over
    # time. The potential penalizes connectivities that approach zero
    connectivity = jax.vmap(algebraic_connectivity, in_axes=(0, None))(
        qs[:, :, :2], communication_range
    )
    inverse_connectivity = 1 / (connectivity + 1e-2)
    potential = softmax(inverse_connectivity, b)
    # Add a term that encourages getting the formation COM to the desired position
    formation_final_com = jnp.mean(qs[-1, :, :2], axis=0)
    potential += 10 * jnp.sqrt(
        jnp.sum((formation_final_com - goal_position) ** 2) + 1e-2
    )

    # Return the result
    return FormationResult(positions=qs, potential=potential, connectivity=connectivity)


if __name__ == '__main__':

    # --- SETTING UP THE SIMULATION ---
    n = 5
    formation_shape = 'v-formation'
    duration_time = 1000.0

    # if n == 1:
    #     desired_r_theta = jnp.zeros((0,2))
    # else:
    #     r_theta = jnp.array([[1,jnp.pi/4]])
    #     desired_r_theta = jnp.tile(r_theta, (n-1,1))

    initial_states = init_bot_positions(n)

    # DEBUG TEST 1
    desired_r_theta = jnp.array([[jnp.sqrt(50), jnp.pi/4],
                                 [jnp.sqrt(50), 3*jnp.pi/4],
                                 [jnp.sqrt(50), jnp.pi/4],
                                 [jnp.sqrt(50), 3*jnp.pi/4]])

    leader_trajectory = Trajectory2D(jnp.array([[0,25]]))
    goal_position = jnp.array([-5,-5])

    result = simulate(
        n,
        formation_shape,
        desired_r_theta,
        duration_time,
        goal_position
    )

    # ----- MATPLOTLIB -----

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, axes = plt.subplots(1, 1, figsize=(7, 7))

    # Plot the trajectories
    traj = axes.plot(result.positions[:, 0, 0], result.positions[:, 0, 1], color='black')
    for i in range(1,n):
        axes.plot(result.positions[:, i, 0], result.positions[:, i, 1], color='dodgerblue')
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title("Positions")

    # Plot the initial position of global leader
    lead_x = initial_states[0,0]
    lead_y = initial_states[0,1]
    axes.plot(lead_x, lead_y, marker = 'o', color = 'k')

    # Plot the initial position of everyone else
    for i in range(1,n):
        follow_x = initial_states[i,0]
        follow_y = initial_states[i,1]
        axes.plot(follow_x, follow_y, marker = 'o', color = 'dodgerblue')

    # Plot the final positions of each bot
    axes.plot(result.positions[int(duration_time), 0, 0], result.positions[int(duration_time), 0, 1], marker = 'x', color = 'k')
    for i in range(1,n):
        final_x = result.positions[int(duration_time), i, 0]
        final_y = result.positions[int(duration_time), i, 1]
        axes.plot(final_x, final_y, marker = 'x', color = 'b')

    # Goal position
    # axes.plot(goal_position[0], goal_position[1], marker = 'X', color = 'darkviolet')
    # axes.annotate('Goal', xy=(goal_position[0], goal_position[1]),xytext=(1.5, 1.5), textcoords='offset points')

    plt.gca().set_aspect('equal')

    # plt.xlim([-5, 10])

    plt.show()
