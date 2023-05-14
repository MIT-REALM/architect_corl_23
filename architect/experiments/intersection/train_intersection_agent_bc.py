"""Train an agent for the drone environment using behavior cloning."""
import os
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import matplotlib.image
import optax
from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax_tqdm import scan_tqdm
from jaxtyping import Array, Bool, Float, jaxtyped
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from architect.experiments.highway.predict_and_mitigate import sample_non_ego_actions
from architect.systems.components.sensing.vision.render import CameraIntrinsics
from architect.systems.intersection.env import HighwayObs, IntersectionEnv
from architect.systems.intersection.oracle import oracle_policy
from architect.systems.intersection.policy import DrivingPolicy
from architect.systems.intersection.scene import IntersectionScene
from architect.types import PRNGKeyArray

#############################################################################
# Define some utilities for generating training trajectories and generalized
# advantage estimation.
#############################################################################


class Trajectory(NamedTuple):
    observations: HighwayObs
    actions: Float[Array, " n_actions"]
    expert_actions: Float[Array, " n_actions"] = None


@jaxtyped
@beartype
def generate_trajectory_learner(
    env: IntersectionEnv,
    policy: DrivingPolicy,
    key: PRNGKeyArray,
    rollout_length: int,
) -> Trajectory:
    """Rollout the policy and generate a trajectory to train on.

    Args:
        env: The environment to rollout in.
        policy: The policy to rollout.
        key: The PRNG key to use for sampling.
        rollout_length: The length of the trajectory.

    Returns:
        The trajectory generated by the rollout.
    """

    # Create a function to take one step with the policy
    @scan_tqdm(rollout_length, message="Learner rollout")
    def step(carry, scan_input):
        # Unpack the input
        _, key, non_ego_action = scan_input  # first element is index for tqdm
        # Unpack the carry
        obs, state = carry

        # Sample an action from the policy
        action = policy(obs, key)

        # Also get the expert's label for this state
        key, subkey = jrandom.split(key)
        expert_action = oracle_policy(state, 50, env, subkey)

        # Take a step in the environment using that action
        next_state, next_observation, _, _ = env.step(
            state, action, non_ego_action, key
        )

        next_carry = (next_observation, next_state)
        output = (obs, action, expert_action)
        return next_carry, output

    # Get the initial state
    reset_key, action_key, rollout_key = jrandom.split(key, 3)
    initial_state = env.reset(reset_key)
    initial_obs = env.get_obs(initial_state)

    # Sample non-ego actions
    non_ego_actions = sample_non_ego_actions(
        action_key,
        env,
        rollout_length,
        initial_state.non_ego_states.shape[0],
        noise_scale=0.05,
    )

    # Transform and rollout!
    keys = jrandom.split(rollout_key, rollout_length)
    _, (obs, learner_actions, expert_actions) = jax.lax.scan(
        step,
        (initial_obs, initial_state),
        (jnp.arange(rollout_length), keys, non_ego_actions),
    )

    # Save all this information in a trajectory object
    trajectory = Trajectory(
        observations=obs,
        actions=learner_actions,
        expert_actions=expert_actions,
    )

    return trajectory


@jaxtyped
@beartype
def generate_trajectory_expert(
    env: IntersectionEnv,
    key: PRNGKeyArray,
    rollout_length: int,
    action_noise: float = 0.05,
) -> Trajectory:
    """Rollout the policy and generate an expert trajectory to train on.

    Args:
        env: The environment to rollout in.
        key: The PRNG key to use for sampling.
        rollout_length: The length of the trajectory.
        action_noise: The amount of noise to add to the actions.

    Returns:
        The trajectory generated by the rollout.
    """

    # Create a function to take one step with the policy
    @scan_tqdm(rollout_length, message="Expert rollout")
    def step(carry, scan_input):
        # Unpack the input
        _, key, non_ego_action = scan_input  # first element is index for tqdm
        action_key, noise_key, step_key = jrandom.split(key, 3)

        # Unpack the carry
        obs, state = carry

        # Also get the expert's label for this state
        expert_action = oracle_policy(state, 50, env, action_key)
        expert_action += jrandom.normal(noise_key, expert_action.shape) * action_noise

        # Take a step in the environment using that action
        next_state, next_observation, _, _ = env.step(
            state, expert_action, non_ego_action, step_key
        )

        next_carry = (next_observation, next_state)
        output = (obs, expert_action)
        return next_carry, output

    # Get the initial state
    reset_key, non_ego_action_key, rollout_key = jrandom.split(key, 3)
    initial_state = env.reset(reset_key)
    initial_obs = env.get_obs(initial_state)

    # Sample non-ego actions
    non_ego_actions = sample_non_ego_actions(
        non_ego_action_key,
        env,
        rollout_length,
        initial_state.non_ego_states.shape[0],
        noise_scale=0.05,
    )

    # Transform and rollout!
    keys = jrandom.split(rollout_key, rollout_length)
    _, (obs, actions) = jax.lax.scan(
        step,
        (initial_obs, initial_state),
        (jnp.arange(rollout_length), keys, non_ego_actions),
    )

    # Save all this information in a trajectory object
    trajectory = Trajectory(
        observations=obs,
        actions=actions,
    )

    return trajectory


@jaxtyped
@beartype
def shuffle_trajectory(traj: Trajectory, key: PRNGKeyArray) -> Trajectory:
    """Shuffle the trajectory.

    Args:
        traj: The trajectory to shuffle.
        key: The PRNG key to use for shuffling.

    Returns:
        The shuffled trajectory.
    """
    # Shuffle the trajectory
    traj_len = traj.actions.shape[0]
    permutation = jrandom.permutation(key, traj_len)
    traj = jtu.tree_map(lambda x: x[permutation], traj)

    return traj


def save_traj_imgs(trajectory: Trajectory, logdir: str, epoch_num: int) -> None:
    """Save the given trajectory to a gif."""
    color_images = trajectory.observations.color_image
    img_dir = os.path.join(logdir, f"epoch_{epoch_num}_imgs")
    os.makedirs(img_dir, exist_ok=True)
    for i, img in enumerate(color_images):
        matplotlib.image.imsave(
            os.path.join(img_dir, f"img_{i}.png"), img.transpose(1, 0, 2)
        )


def make_intersection_env(image_shape: Tuple[int, int]):
    """Make the intersection environment."""
    scene = IntersectionScene()
    intrinsics = CameraIntrinsics(
        focal_length=0.1,
        sensor_size=(0.1, 0.1),
        resolution=image_shape,
    )
    initial_ego_state = jnp.array([-20, -3.75, 0.0, 3.0])
    initial_non_ego_states = jnp.array(
        [
            [-3.75, 10, -jnp.pi / 2, 4.5],
            [-3.75, 20, -jnp.pi / 2, 4.5],
        ]
    )
    initial_state_covariance = jnp.diag(jnp.array([0.5, 0.5, 0.001, 0.2]) ** 2)

    # Set the direction of light shading
    shading_light_direction = jnp.array([-0.2, -1.0, 1.5])
    shading_light_direction /= jnp.linalg.norm(shading_light_direction)
    shading_direction_covariance = (0.25) ** 2 * jnp.eye(3)

    env = IntersectionEnv(
        scene,
        intrinsics,
        dt=0.1,
        initial_ego_state=initial_ego_state,
        initial_non_ego_states=initial_non_ego_states,
        initial_state_covariance=initial_state_covariance,
        collision_penalty=5.0,
        mean_shading_light_direction=shading_light_direction,
        shading_light_direction_covariance=shading_direction_covariance,
    )
    return env


def train_intersection(
    image_shape: Tuple[int, int],
    learning_rate: float = 1e-5,
    seed: int = 0,
    steps_per_epoch: int = 32 * 70,
    epochs: int = 100,
    gd_steps_per_update: int = 500,
    minibatch_size: int = 32,
    logdir: str = "./tmp/intersection",
):
    """
    Train the agent using behavior cloning.

    Args: various hyperparameters.
    """
    # Set up logging
    writer = SummaryWriter(logdir)

    # Set up the environment
    env = make_intersection_env(image_shape)

    # Set up the policy
    key = jrandom.PRNGKey(seed)
    key, subkey = jrandom.split(key)
    policy = DrivingPolicy(subkey, image_shape)

    # Set up the optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    # Compile a loss function and optimization update
    @eqx.filter_value_and_grad
    def loss_fn(policy: DrivingPolicy, trajectory: Trajectory) -> Float[Array, ""]:
        # Compute the predicted action for each observation
        predicted_actions = jax.vmap(policy, in_axes=(0, None))(
            trajectory.observations, jrandom.PRNGKey(0)
        )

        # Minimize L2 loss between predicted and actual actions
        action_loss = jnp.mean(
            jnp.square(predicted_actions - trajectory.expert_actions)
        )

        return action_loss

    @eqx.filter_jit
    def step_fn(
        opt_state: optax.OptState, policy: DrivingPolicy, trajectory: Trajectory
    ):
        loss, grad = loss_fn(policy, trajectory)
        updates, opt_state = optimizer.update(grad, opt_state)
        policy = eqx.apply_updates(policy, updates)
        return (
            loss,
            policy,
            opt_state,
        )

    # Training loop
    for epoch in range(epochs):
        # Generate a trajectory with the learner policy, getting labels from the expert
        key, subkey = jrandom.split(key)
        trajectory = generate_trajectory_learner(env, policy, subkey, steps_per_epoch)
        # trajectory = generate_trajectory_expert(env, subkey, steps_per_epoch)

        # Save regularly
        if epoch % 5 == 0 or epoch == epochs - 1:
            # Save trajectory images; can be converted to video using this command:
            # ffmpeg -framerate 10 -i img_%d.png -c:v libx264 -r 30 -vf \
            #   scale=320x320:flags=neighbor out.mp4
            save_traj_imgs(trajectory, logdir, epoch)
            # Save policy
            eqx.tree_serialise_leaves(
                os.path.join(logdir, f"policy_{epoch}.eqx"), policy
            )

        # Shuffle the trajectory into minibatches
        key, subkey = jrandom.split(key)
        trajectory = shuffle_trajectory(trajectory, subkey)

        # Compute the loss and gradient
        for i in tqdm(range(gd_steps_per_update)):
            epoch_loss = 0.0
            batches = 0
            for batch_start in range(0, steps_per_epoch, minibatch_size):
                key, subkey = jrandom.split(key)
                (
                    loss,
                    policy,
                    opt_state,
                ) = step_fn(
                    opt_state,
                    policy,
                    jtu.tree_map(
                        lambda x: x[batch_start : batch_start + minibatch_size],
                        trajectory,
                    ),
                )
                batches += 1
                epoch_loss += loss.item()

            # Average
            epoch_loss /= batches

        # Log the loss
        print(f"Epoch {epoch:03d}; loss: {epoch_loss:.2f}")
        writer.add_scalar("loss", epoch_loss, epoch)


if __name__ == "__main__":
    train_intersection((32, 32))
