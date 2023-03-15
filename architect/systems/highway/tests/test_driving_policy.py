"""Test the driving policy network."""
import pytest
import jax
import jax.numpy as jnp
import jax.random as jrandom

from architect.systems.highway.driving_policy import DrivingPolicy
from architect.systems.highway.highway_env import HighwayObs


@pytest.fixture
def driving_policy():
    return DrivingPolicy(
        key=jax.random.PRNGKey(0),
        image_shape=(128, 128),
        image_channels=1,
        cnn_channels=32,
        fcn_layers=3,
        fcn_width=32,
    )


@pytest.fixture
def obs():
    return HighwayObs(
        depth_image=jnp.zeros((128, 128)),
        speed=jnp.array(10.0),
    )


@pytest.fixture
def key():
    return jrandom.PRNGKey(0)


def test_driving_policy(driving_policy, obs, key):
    """Test the driving policy network."""
    action, action_logp, value = driving_policy(obs, key)
    assert action.shape == (2,)
    assert action_logp.shape == ()
    assert value.shape == (1,)
