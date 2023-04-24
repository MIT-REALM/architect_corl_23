"""Test the highway scene."""
import jax.numpy as jnp

from architect.systems.components.sensing.vision.render import (
    CameraExtrinsics, CameraIntrinsics)
from architect.systems.components.sensing.vision.util import look_at
from architect.systems.highway.highway_env import HighwayScene


def test_highway_scene():
    # Create a test highway scene and render it
    highway = HighwayScene(num_lanes=3, lane_width=4.0)
    car_states = jnp.array(
        [
            [7.0, 0.0, 0.0],
            [0.0, highway.lane_width, 0.0],
            [-5.0, -highway.lane_width, 0.0],
        ]
    )

    # Set the camera parameters
    intrinsics = CameraIntrinsics(
        focal_length=0.1,
        sensor_size=(0.1, 0.1),
        resolution=(512, 512),
    )
    extrinsics = CameraExtrinsics(
        camera_origin=jnp.array([-10.0, 0.0, 10.0]),
        camera_R_to_world=look_at(jnp.array([-10.0, 0.0, 10.0]), jnp.zeros(3)),
    )

    depth_image = highway.render_depth(intrinsics, extrinsics, car_states)

    assert depth_image is not None
