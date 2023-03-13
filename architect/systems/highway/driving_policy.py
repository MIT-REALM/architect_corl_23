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
    """A neural network policy for driving in the highway environment.

    The policy has a convolutional encoder to compute a latent embedding of the
    given image observation, and a fully-connected final to compute the action
    based on the latent embedding concatenated with the current forward velocity.
    """

    encoder_conv_1: eqx.nn.Conv2d
    encoder_conv_2: eqx.nn.Conv2d
    encoder_conv_3: eqx.nn.Conv2d

    fcn: eqx.nn.MLP

    @jaxtyped
    @beartype
    def __init__(
        self,
        key: PRNGKeyArray,
        image_shape: Tuple[int, int],
        image_channels: int = 1,
        cnn_channels: int = 32,
        fcn_layers: int = 3,
        fcn_width: int = 32,
    ):
        """Initialize the policy."""
        cnn_key, fcn_key = jrandom.split(key)

        # Create the convolutional encoder
        cnn_keys = jrandom.split(cnn_key, 3)
        kernel_size = 16
        stride = 4
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
        self.fcn = eqx.nn.MLP(
            in_size=embedding_size + 1,
            out_size=2,
            width_size=fcn_width,
            depth=fcn_layers,
            key=fcn_key,
        )

    def __call__(self, obs: HighwayObs) -> Float[Array, " 2"]:
        """Compute the action for the given state."""
        # Compute the image embedding
        depth_image = obs.depth_image
        depth_image = jnp.expand_dims(depth_image, axis=0)
        y = jax.nn.relu(self.encoder_conv_1(depth_image))
        y = jax.nn.relu(self.encoder_conv_2(y))
        y = jax.nn.relu(self.encoder_conv_3(y))
        y = jnp.reshape(y, (-1,))

        # Concatenate the embedding with the forward velocity
        y = jnp.concatenate((y, obs.speed.reshape(-1)))

        # Compute the action
        action = self.fcn(y)
        return action
