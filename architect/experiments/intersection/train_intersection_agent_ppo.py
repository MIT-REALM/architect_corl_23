"""Train an agent for the intersection environment using PPO."""
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

from architect.systems.components.sensing.vision.render import CameraIntrinsics
from architect.systems.highway.highway_env import HighwayObs
from architect.systems.intersection.env import IntersectionEnv
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
    action_log_probs: Float[Array, " n_actions"]
    rewards: Float[Array, ""]
    returns: Float[Array, ""]
    advantages: Float[Array, ""]
    done: Bool[Array, ""]


@jaxtyped
@beartype
def generalized_advantage_estimate(
    rewards: Float[Array, " n_steps"],
    values: Float[Array, " n_steps+1"],
    dones: Bool[Array, " n_steps"],
    gamma: float,
    lam: float,
) -> Tuple[Float[Array, " n_steps"], Float[Array, " n_steps"]]:
    """Compute the generalized advantage estimation for a trajectory.

    Args:
        rewards: The rewards for each step in the trajectory.
        values: The values for each step in the trajectory (plus the value for the final
            state).
        dones: True if a step in the trajectory is a terminal state, False otherwise.
        gamma: The discount factor for GAE.
        lam: The lambda factor for GAE.

    Returns:
        The advantages and returns for each step in the trajectory.
    """

    def gae_step(advantage, gae_input):
        # Unpack input
        (
            reward,  # reward in current state
            current_value,  # value estimate in current state
            next_value,  # value estimate in next state
            terminal,  # is the current state terminal?
        ) = gae_input

        # Difference between current value estimate and reward-to-go
        delta = reward + gamma * next_value * (1.0 - terminal) - current_value
        # Advantage estimate
        advantage = delta + gamma * lam * advantage * (1.0 - terminal)

        # Reshape to scalars
        delta = delta.reshape()
        advantage = advantage.reshape()

        return advantage, (advantage, delta)  # carry and output

    # Compute the advantage estimate for each step in the trajectory
    _, (advantages, delta) = jax.lax.scan(
        gae_step,
        jnp.array(0.0),
        (rewards, values[:-1], values[1:], dones),
        reverse=True,
    )

    returns = delta + values[:-1]

    return advantages, returns


@jaxtyped
@beartype
def generate_trajectory(
    env: IntersectionEnv,
    policy: DrivingPolicy,
    non_ego_actions: Float[Array, "n_non_ego n_actions"],
    key: PRNGKeyArray,
    rollout_length: int,
    gamma: float,
    gae_lambda: float,
) -> Trajectory:
    """Rollout the policy and generate a trajectory to train on.

    Args:
        env: The environment to rollout in.
        policy: The policy to rollout.
        non_ego_actions: The actions for the non-ego vehicles (held constant).
        key: The PRNG key to use for sampling.
        rollout_length: The length of the trajectory.
        gamma: The discount factor for GAE.
        gae_lambda: The lambda parameter for GAE.

    Returns:
        The trajectory generated by the rollout.
    """

    # Create a function to take one step with the policy
    @scan_tqdm(rollout_length, message="Rollout")
    def step(carry, scan_input):
        # Unpack the input
        _, key = scan_input  # first element is index for tqdm
        # Unpack the carry
        obs, state = carry

        # PRNG key management
        step_key, action_subkey = jrandom.split(key)

        # Sample an action from the policy
        action, action_logprob, value = policy(obs, action_subkey, deterministic=False)

        # Take a step in the environment using that action
        next_state, next_observation, reward, done = env.step(
            state, action, non_ego_actions, step_key
        )

        next_carry = (next_observation, next_state)
        output = (obs, action, action_logprob, reward, value, done)
        return next_carry, output

    # Get the initial state
    reset_key, rollout_key = jrandom.split(key)
    initial_state = env.reset(reset_key)
    initial_obs = env.get_obs(initial_state)

    # Transform and rollout!
    keys = jrandom.split(rollout_key, rollout_length)
    _, (obs, actions, action_logprobs, rewards, values, dones) = jax.lax.scan(
        step, (initial_obs, initial_state), (jnp.arange(rollout_length), keys)
    )

    # Normalize rewards used to compute advantage estimate
    normalized_rewards = (rewards - jnp.mean(rewards)) / (jnp.std(rewards) + 1e-8)

    # Compute the advantage estimate. This requires the value estimate at the
    # end of the rollout. The key we use doesn't matter here.
    _, _, final_value = policy(jtu.tree_map(lambda x: x[-1], obs), keys[-1])
    values = jnp.concatenate([values, jnp.expand_dims(final_value, 0)], axis=0)
    values = values.reshape(-1)
    advantage, returns = generalized_advantage_estimate(
        normalized_rewards, values, dones, gamma, gae_lambda
    )

    # Save all this information in a trajectory object
    trajectory = Trajectory(
        observations=obs,
        actions=actions,
        action_log_probs=action_logprobs,
        rewards=rewards,
        returns=returns,
        advantages=advantage,
        done=dones,
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
    traj_len = traj.done.shape[0]
    permutation = jrandom.permutation(key, traj_len)
    traj = jtu.tree_map(lambda x: x[permutation], traj)

    return traj


@jaxtyped
@beartype
def ppo_clip_loss_fn(
    policy: DrivingPolicy,
    traj: Trajectory,
    epsilon: float,
    critic_weight: float,
    entropy_weight: float,
) -> Tuple[
    Float[Array, ""], Tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]
]:
    """Compute the clipped PPO loss.

    Args:
        policy: The current policy.
        traj: The training trajectory.
        epsilon: The epsilon parameter for the PPO loss.
        critic_weight: The weight for the critic loss.
        entropy_weight: The weight for the entropy loss.

    Returns:
        The total PPO loss and a tuple of component losses
    """
    # Get the action log probabilities using the current policy, so we can compute
    # the ratio of the action probabilities under the new and old policies.
    # Also get the value, which we'll use to compute the critic loss
    action_logprobs, value_estimate = jax.vmap(
        policy.action_log_prob_and_value, in_axes=(0, 0)
    )(traj.observations, traj.actions)
    likelihood_ratio = jnp.exp(action_logprobs - traj.action_log_probs)
    clipped_likelihood_ratio = jnp.clip(likelihood_ratio, 1 - epsilon, 1 + epsilon)

    # Normalize the advantage
    advantages = (traj.advantages - traj.advantages.mean()) / (
        traj.advantages.std() + 1e-8
    )

    # The PPO loss for the actor is the average minimum of the product of these ratios
    # with the advantage estimate
    actor_loss = -jnp.minimum(
        likelihood_ratio * advantages, clipped_likelihood_ratio * advantages
    ).mean()

    entropy_loss = -entropy_weight * policy.entropy()

    # The critic loss is the mean squared error between the value estimate and the
    # reward-to-go
    critic_loss = critic_weight * jnp.square(traj.returns - value_estimate).mean()

    loss = actor_loss + critic_loss + entropy_loss

    return loss, (actor_loss, critic_loss, entropy_loss)


def save_traj_imgs(trajectory: Trajectory, logdir: str, epoch_num: int) -> None:
    """Save the given trajectory to a gif."""
    color_images = trajectory.observations.color_image
    img_dir = os.path.join(logdir, f"epoch_{epoch_num}_imgs")
    os.makedirs(img_dir, exist_ok=True)
    for i, img in enumerate(color_images):
        try:
            matplotlib.image.imsave(
                os.path.join(img_dir, f"img_{i}.png"), img.transpose(1, 0, 2)
            )
        except ValueError:
            pass


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
            [3.75, -30, jnp.pi / 2, 4.5],
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


def train_ppo_driver(
    image_shape: Tuple[int, int],
    learning_rate: float = 1e-5,
    gamma: float = 0.99,
    gae_lambda: float = 0.97,
    epsilon: float = 0.2,
    critic_weight: float = 1.0,
    entropy_weight: float = 0.1,
    seed: int = 0,
    steps_per_epoch: int = 32 * 50,
    epochs: int = 200,
    gd_steps_per_update: int = 50,
    minibatch_size: int = 32,
    max_grad_norm: float = 1.0,
    logdir: str = "./tmp/intersection_ppo_reward",
):
    """Train the driver using PPO.

    Args: various hyperparameters.
    """
    # Set up logging
    writer = SummaryWriter(logdir)

    # Set up the environment
    env = make_intersection_env(image_shape)

    # Fix non-ego actions to be constant (drive straight at fixed speed)
    n_non_ego = 3
    non_ego_actions = jnp.zeros((n_non_ego, 2))

    # Set up the policy
    key = jrandom.PRNGKey(seed)
    key, subkey = jrandom.split(key)
    policy = DrivingPolicy(subkey, image_shape)

    # Set up the optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))
    grad_clip = optax.clip_by_block_rms(max_grad_norm)
    grad_clip_state = grad_clip.init(eqx.filter(policy, eqx.is_array))

    # Compile a loss function and optimization update
    @partial(eqx.filter_value_and_grad, has_aux=True)
    def loss_fn(policy: DrivingPolicy, trajectory: Trajectory) -> Float[Array, ""]:
        return ppo_clip_loss_fn(
            policy, trajectory, epsilon, critic_weight, entropy_weight
        )

    @eqx.filter_jit
    def step_fn(
        opt_state: optax.OptState, policy: DrivingPolicy, trajectory: Trajectory
    ):
        (loss, (actor_loss, critic_loss, entropy_loss)), grad = loss_fn(
            policy, trajectory
        )
        grad, _ = grad_clip.update(grad, grad_clip_state)  # empty state
        updates, opt_state = optimizer.update(grad, opt_state)
        policy = eqx.apply_updates(policy, updates)
        grad_norm = jtu.tree_reduce(jnp.add, jax.tree_map(jnp.linalg.norm, grad))
        return (
            loss,
            policy,
            opt_state,
            (actor_loss, critic_loss, entropy_loss, grad_norm),
        )

    # Training loop
    for epoch in range(epochs):
        # Generate a trajectory
        key, subkey = jrandom.split(key)
        trajectory = generate_trajectory(
            env,
            policy,
            non_ego_actions,
            key,
            steps_per_epoch,
            gamma,
            gae_lambda,
        )
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
        for i in range(gd_steps_per_update):
            epoch_loss = 0.0
            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0
            epoch_entropy_loss = 0.0
            epoch_grad_norm = 0.0
            batches = 0
            for batch_start in range(0, steps_per_epoch, minibatch_size):
                key, subkey = jrandom.split(key)
                (
                    loss,
                    policy,
                    opt_state,
                    (actor_loss, critic_loss, entropy_loss, grad_norm),
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
                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += critic_loss.item()
                epoch_entropy_loss += entropy_loss.item()
                epoch_grad_norm += grad_norm.item()

            # Average
            epoch_loss /= batches
            epoch_actor_loss /= batches
            epoch_critic_loss /= batches
            epoch_entropy_loss /= batches
            epoch_grad_norm /= batches

        # Log the loss
        print(
            (
                f"Epoch {epoch:03d}; loss: {epoch_loss:.2f} "
                f"(actor {epoch_actor_loss:.2f}, critic {epoch_critic_loss:.2f}) "
                f"total_reward: {trajectory.rewards.mean():.2f} "
                f"total_return: {trajectory.returns.mean():.2f} "
                f"entropy loss: {epoch_entropy_loss:.2f} "
                f"grad_norm: {epoch_grad_norm:.2f}"
            )
        )
        writer.add_scalar("loss", epoch_loss, epoch)
        writer.add_scalar("actor loss", epoch_actor_loss, epoch)
        writer.add_scalar("critic loss", epoch_critic_loss, epoch)
        writer.add_scalar("episode reward", trajectory.rewards.mean().item(), epoch)
        writer.add_scalar("episode return", trajectory.returns.mean().item(), epoch)
        writer.add_scalar("entropy loss", epoch_entropy_loss, epoch)
        writer.add_scalar("grad norm", epoch_grad_norm, epoch)


if __name__ == "__main__":
    train_ppo_driver((32, 32))
