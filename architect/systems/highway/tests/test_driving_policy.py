"""Test the driving policy network."""
import pytest
import jax
import jax.numpy as jnp


from architect.systems.highway.driving_policy import DrivingPolicy
from architect.systems.highway.highway_env import HighwayObs


@pytest.fixture
def driving_policy():
    return DrivingPolicy(
        key=jax.random.PRNGKey(0),
        image_shape=(512, 512),
        image_channels=1,
        cnn_channels=32,
        fcn_layers=3,
        fcn_width=32,
    )


@pytest.fixture
def obs():
    return HighwayObs(
        depth_image=jnp.zeros((512, 512)),
        speed=jnp.array(10.0),
    )


def test_driving_policy(driving_policy, obs):
    """Test the driving policy network."""
    action = driving_policy(obs)
    assert action.shape == (2,)
