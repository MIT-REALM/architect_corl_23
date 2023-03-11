"""Functions for rendering a scene."""

import jax
import jax.numpy as jnp

from jaxtyping import Float, Integer, Array, jaxtyped
from beartype import beartype
from beartype.typing import Callable


@jaxtyped
@beartype
def pinhole_camera_rays(
    camera_R_to_world: Float[Array, "3 3"],
    sensor_size: Float[Array, " 2"],
    resolution: Integer[Array, " 2"],
    focal_length: Float[Array, ""],
) -> Float[Array, "n_rays 3"]:
    """Return the rays originating from a pinhole camera, in world coordinates.

    Args:
        camera_R_to_world: The rotation matrix from camera to world coordinates.
        sensor_size: The (x, y) size of the camera sensor, in meters.
        resolution: The (x, y) resolution of the camera, in pixels/m.
        focal_length: The focal length of the camera, in meters.

    Returns:
        The rays originating from the camera, in world coordinates.
    """
    # Get the points where the rays pass through the image plane of the camera,
    # in camera coordinates.
    sensor_x, sensor_y = sensor_size
    res_x, res_y = resolution
    x_img, y_img = jnp.mgrid[
        -sensor_x : sensor_x : res_x * 1j,  # complex slice -> linspace-like behavior
        -sensor_y : sensor_y : res_y * 1j,
    ].reshape(2, -1)

    # The rays are the unit vectors from the camera origin to these points
    rays = jnp.c_[x_img, y_img, focal_length * jnp.ones_like(x_img)]
    rays = rays / jnp.linalg.norm(rays, axis=1, keepdims=True)

    # Convert to world coordinates
    return rays @ camera_R_to_world.T


@jaxtyped
@beartype
def raycast(
    sdf: Callable[[Float[Array, " 3"]], Float[Array, ""]],
    origin: Float[Array, " 3"],
    ray: Float[Array, " 3"],
    max_steps: int = 50,
) -> Float[Array, " 3"]:
    """Find the points at which the given ray intersects the given SDF.

    Args:
        sdf: The signed distance function of the scene.
        origin: The origin of the ray, in world coordinates.
        ray: The ray to cast, in world coordinates.
        max_steps: The maximum number of steps to take along the ray.

    Returns:
        The point at which the ray intersects the SDF, in world coordinates.
    """

    # At each iteration, we can step the ray a distance equal to the current value
    # of the SDF without intersecting the surface. The first argument passed by
    # fori_loop is the iteration number, which we don't need.
    def step_ray(_, current_pt: Float[Array, " 3"]) -> Float[Array, " 3"]:
        return current_pt + sdf(current_pt) * ray

    return jax.lax.fori_loop(0, max_steps, step_ray, origin)


def render_depth(
    hit_pts: Float[Array, "n_rays 3"],
    camera_origin: Float[Array, " 3"],
    camera_R_to_world: Float[Array, "3 3"],
    resolution: Integer[Array, " 2"],
    max_dist: float = 10.0,
) -> Float[Array, " n_rays"]:
    """Render the depth of the given hit points.

    Args:
        hit_pts: The hit points of the rays, in world coordinates.
        camera_origin: The origin of the camera, in world coordinates.
        camera_R_to_world: The rotation matrix from camera to world coordinates.
        resolution: The (x, y) resolution of the camera, in pixels.
        max_dist: The maximum distance to render, in meters. Set the depth for all rays
            without an intersection within this distance to 0.

    Returns:
        The depth of the hit points, in meters
    """
    # Get the depth image by getting the distance along the camera z axis to each hit
    # point
    depth_values = jax.vmap(jnp.dot, in_axes=(0, None))(
        hit_pts - camera_origin, camera_R_to_world @ jnp.array([0.0, 0.0, 1.0])
    )

    # Clip the depth image at a maximum distance
    depth_values = jnp.clip(depth_values, a_min=0.0, a_max=max_dist)
    depth_values = jnp.where(depth_values == max_dist, 0.0, depth_values)

    return depth_values
