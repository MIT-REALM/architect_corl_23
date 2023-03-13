"""Test the highway environment."""
import pytest
import jax.numpy as jnp

from architect.systems.components.sensing.vision.render import CameraIntrinsics
from architect.systems.highway.highway_scene import HighwayScene
from architect.systems.highway.highway_env import HighwayEnv, HighwayState


@pytest.fixture
def highway_scene():
    """A highway scene to use in testing."""
    return HighwayScene(num_lanes=3, lane_width=4.0)


@pytest.fixture
def intrinsics():
    """Camera intrinsics to use during testing."""
    intrinsics = CameraIntrinsics(
        focal_length=0.1,
        sensor_size=(0.1, 0.1),
        resolution=(512, 512),
    )
    return intrinsics


@pytest.fixture
def highway_env(highway_scene, intrinsics):
    """Highway environment system under test."""
    return HighwayEnv(highway_scene, intrinsics, dt=0.1)


def test_highway_env_init(highway_env):
    """Test the highway environment initialization."""
    assert highway_env is not None


def test_highway_env_step(highway_env):
    """Test the highway environment step."""
    # Create actions for the ego and non-ego agents (driving straight at fixed speed)
    ego_action = jnp.array([0.0, 0.0])
    non_ego_actions = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    # Initialize a state
    state = HighwayState(
        ego_state=jnp.array([-15.0, 0.0, 0.0, 10.0]),
        non_ego_states=jnp.array(
            [
                [7.0, 0.0, 0.0, 10.0],
                [0.0, 4.0, 0.0, 9.0],
                [-5.0, -4.0, 0.0, 11.0],
            ]
        ),
    )

    # Take a step
    next_state, obs, reward, done = highway_env.step(state, ego_action, non_ego_actions)

    assert next_state is not None
    assert obs is not None
    assert reward is not None
    assert done is not None
