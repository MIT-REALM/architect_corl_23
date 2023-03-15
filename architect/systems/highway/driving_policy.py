"""Define a neural network policy for driving in the highway environment."""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype
from beartype.typing import Tuple

from architect.types import PRNGKeyArray
from architect.systems.highway.highway_env import HighwayObs


class DrivingPolicy(eqx.Module):
    """A neural network actor-critic policy for driving in the highway environment.

    The policy has a convolutional encoder to compute a latent embedding of the
    given image observation, and a fully-connected final to compute the action
    based on the latent embedding concatenated with the current forward velocity.
    """

    encoder_conv_1: eqx.nn.Conv2d
    encoder_conv_2: eqx.nn.Conv2d
    encoder_conv_3: eqx.nn.Conv2d

    actor_fcn: eqx.nn.MLP
    critic_fcn: eqx.nn.MLP

    @jaxtyped
    @beartype
    def __init__(
        self,
        key: PRNGKeyArray,
        image_shape: Tuple[int, int],
        image_channels: int = 1,
        cnn_channels: int = 32,
        fcn_layers: int = 2,
        fcn_width: int = 32,
    ):
        """Initialize the policy."""
        cnn_key, fcn_key = jrandom.split(key)

        # Create the convolutional encoder
        cnn_keys = jrandom.split(cnn_key, 3)
        kernel_size = 6
        stride = 2
        self.encoder_conv_1 = eqx.nn.Conv2d(
            key=cnn_keys[0],
            in_channels=image_channels,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        # Get new image shape after convolution
        image_shape = jnp.floor(
            (jnp.array(image_shape) - (kernel_size - 1) - 1) / stride + 1
        )
        self.encoder_conv_2 = eqx.nn.Conv2d(
            key=cnn_keys[1],
            in_channels=cnn_channels,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        image_shape = jnp.floor((image_shape - (kernel_size - 1) - 1) / stride + 1)
        self.encoder_conv_3 = eqx.nn.Conv2d(
            key=cnn_keys[2],
            in_channels=cnn_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
        )
        image_shape = jnp.floor((image_shape - (kernel_size - 1) - 1) / stride + 1)
        embedding_size = image_shape[0] * image_shape[1]
        embedding_size = int(embedding_size)

        # Create the fully connected layers
        self.actor_fcn = eqx.nn.MLP(
            in_size=embedding_size + 1,
            out_size=2,
            width_size=fcn_width,
            depth=fcn_layers,
            key=fcn_key,
        )
        self.critic_fcn = eqx.nn.MLP(
            in_size=embedding_size + 1,
            out_size=1,
            width_size=fcn_width,
            depth=fcn_layers,
            key=fcn_key,
        )

    def forward(self, obs: HighwayObs) -> Tuple[Float[Array, " 2"], Float[Array, ""]]:
        """Compute the mean action and value estimate for the given state.

        Args:
            obs: The observation of the current state.

        Returns:
            The mean action and value estimate.
        """
        # Compute the image embedding
        depth_image = obs.depth_image
        depth_image = jnp.expand_dims(depth_image, axis=0)
        y = jax.nn.relu(self.encoder_conv_1(depth_image))
        y = jax.nn.relu(self.encoder_conv_2(y))
        y = jax.nn.relu(self.encoder_conv_3(y))
        y = jnp.reshape(y, (-1,))

        # Concatenate the embedding with the forward velocity
        y = jnp.concatenate((y, obs.speed.reshape(-1)))

        # Compute the action and value estimate
        value = self.critic_fcn(y)
        action_mean = self.actor_fcn(y)

        return action_mean, value

    def __call__(
        self, obs: HighwayObs, key: PRNGKeyArray, action_noise: float = 0.1
    ) -> Tuple[Float[Array, " 2"], Float[Array, ""], Float[Array, ""]]:
        """Compute the action and value estimate for the given state.

        Args:
            obs: The observation of the current state.
            key: The random key for sampling the action.
            action_noise: The standard deviation of the Gaussian noise to add
                to the action.

        Returns:
            The action, action log probability, and value estimate.
        """
        action_mean, value = self.forward(obs)

        action = jrandom.multivariate_normal(
            key, action_mean, action_noise * jnp.eye(action_mean.shape[0])
        )
        action_logp = jax.scipy.stats.multivariate_normal.logpdf(
            action, action_mean, action_noise * jnp.eye(action_mean.shape[0])
        )

        return action, action_logp, value

    def action_log_prob_and_value(
        self, obs: HighwayObs, action: Float[Array, " 2"], action_noise: float = 0.1
    ) -> Float[Array, ""]:
        """Compute the log probability of an action with the given observation.

        Args:
            obs: The observation of the current state.
            action: The action to compute the log probability of.
            action_noise: The standard deviation of the Gaussian noise to add

        Returns:
            The log probability of the action and the value estimate.
        """
        action_mean, value = self.forward(obs)

        action_logp = jax.scipy.stats.multivariate_normal.logpdf(
            action, action_mean, action_noise * jnp.eye(action_mean.shape[0])
        )

        return action_logp, value
