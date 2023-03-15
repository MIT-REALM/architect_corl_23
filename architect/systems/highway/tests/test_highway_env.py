"""Test the highway environment."""
import pytest
import jax.numpy as jnp
import jax.random as jrandom

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
    initial_ego_state = jnp.array([-15.0, 0.0, 0.0, 10.0])
    initial_non_ego_states = jnp.array(
        [
            [7.0, 0.0, 0.0, 10.0],
            [0.0, 4.0, 0.0, 9.0],
            [-5.0, -4.0, 0.0, 11.0],
        ]
    )
    return HighwayEnv(
        highway_scene,
        intrinsics,
        0.1,
        initial_ego_state,
        initial_non_ego_states,
        0.1 * jnp.eye(4),
    )


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
    next_state, obs, reward, done = highway_env.step(
        state, ego_action, non_ego_actions, jrandom.PRNGKey(0)
    )

    assert next_state is not None
    assert obs is not None
    assert reward is not None
    assert done is not None


def test_highway_env_reset(highway_env):
    """Test the highway environment reset."""
    state = highway_env.reset(jrandom.PRNGKey(0))
    assert state is not None


def test_highway_env_sample_non_ego_action(highway_env):
    """Test sampling a non-ego action."""
    key = jrandom.PRNGKey(0)
    n_non_ego = 3
    n_actions = 2  # per agent
    noise_cov = jnp.diag(0.1 * jnp.ones(n_actions))
    non_ego_actions = highway_env.sample_non_ego_actions(key, noise_cov, n_non_ego)
    assert non_ego_actions.shape == (n_non_ego, n_actions)


def test_highway_env_non_ego_action_prior_logprob(highway_env):
    """Test computing the log probability of a non-ego action."""
    n_non_ego = 3
    n_actions = 2  # per agent
    non_ego_actions = jnp.zeros((n_non_ego, n_actions))
    noise_cov = jnp.diag(0.1 * jnp.ones(n_actions))
    logprob = highway_env.non_ego_actions_prior_logprob(non_ego_actions, noise_cov)
    assert logprob.shape == ()
