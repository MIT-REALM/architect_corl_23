"""Test scene rendering."""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from architect.systems.components.sensing.vision.render import (
    pinhole_camera_rays,
    raycast,
    render_depth,
)
from architect.systems.components.sensing.vision.shapes import (
    Sphere,
    Box,
    Halfspace,
    Scene,
)
from architect.systems.components.sensing.vision.util import look_at


def test_pinhole_camera_rays():
    """Test the pinhole camera ray generation function."""
    # Define the camera parameters
    camera_R_to_world = jnp.eye(3)
    sensor_size = jnp.array([0.1, 0.1])
    resolution = jnp.array([20, 20])
    focal_length = jnp.array(0.1)

    # Generate the rays
    rays = pinhole_camera_rays(camera_R_to_world, sensor_size, resolution, focal_length)

    # Check that the rays are unit vectors in 3D
    assert jnp.allclose(jnp.linalg.norm(rays, axis=1), 1.0)
    assert rays.shape[1] == 3
    # Check that there are the right number of rays
    assert rays.shape[0] == resolution[0] * resolution[1]


def test_raycast_single():
    """Test the raycasting function on a single ray."""
    # Create some test shapes
    sphere1 = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(1.0))
    sphere2 = Sphere(center=jnp.array([0.0, 0.0, 2.0]), radius=jnp.array(1.0))
    box = Box(
        center=jnp.array([2.0, 0.0, 0.0]),
        extent=jnp.array([1.0, 1.0, 1.0]),
        rotation=jnp.eye(3),
    )

    # Combine them into a scene
    scene = Scene([sphere1, sphere2, box], sharpness=10.0)

    # Create a ray
    origin = jnp.array([0.0, 0.0, -5.0])
    ray = jnp.array([0.0, 0.0, 1.0])

    # Raycast the scene, which returns the world coordinates of the first intersection
    # of each ray with objects in the scene.
    hit_pt = raycast(scene, origin, ray)

    # Make sure the hit point has the appropriate shape and only one dimension
    assert hit_pt.ndim == 1
    assert hit_pt.shape[0] == 3


def test_raycast_multiple():
    """Test the raycasting function on multiple rays."""
    # Create some test shapes
    sphere1 = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(1.0))
    sphere2 = Sphere(center=jnp.array([0.0, 0.0, 2.0]), radius=jnp.array(1.0))
    box = Box(
        center=jnp.array([2.0, 0.0, 0.0]),
        extent=jnp.array([1.0, 1.0, 1.0]),
        rotation=jnp.eye(3),
    )

    # Combine them into a scene
    scene = Scene([sphere1, sphere2, box])

    # Define the camera parameters
    camera_R_to_world = jnp.eye(3)
    sensor_size = jnp.array([0.1, 0.1])
    resolution = jnp.array([20, 20])
    focal_length = jnp.array(0.1)

    # Generate the rays
    origin = jnp.array([0.0, 0.0, -5.0])
    rays = pinhole_camera_rays(camera_R_to_world, sensor_size, resolution, focal_length)

    # Raycast the scene, which returns the world coordinates of the first intersection
    # of each ray with objects in the scene.
    hit_pts = jax.vmap(raycast, in_axes=(None, None, 0))(scene, origin, rays)

    # Make sure the hit points have the appropriate shape
    assert hit_pts.shape[0] == resolution[0] * resolution[1]
    assert hit_pts.shape[1] == 3


def test_render_depth_image():
    """Test rendering a depth image of a scene."""
    # Make a scene
    ground = Halfspace(
        point=jnp.array([0.0, 0.0, 0.0]),
        normal=jnp.array([0.0, 0.0, 1.0]),
    )
    sphere = Sphere(center=jnp.array([0.0, -1.0, 0.0]), radius=jnp.array(1.0))
    box = Box(
        center=jnp.array([0.0, 2.0, 1.0]),
        extent=jnp.array([3.0, 0.5, 0.5]),
        rotation=jnp.eye(3),
    )
    scene = Scene([ground, sphere, box], sharpness=100.0)

    # Set the camera parameters
    camera_origin = jnp.array([3.0, 3.0, 3.0])
    camera_R_to_world = look_at(camera_origin, jnp.zeros(3))
    sensor_size = jnp.array([0.1, 0.1])
    resolution = jnp.array([512, 512])
    focal_length = jnp.array(0.1)

    # Generate the rays
    rays = pinhole_camera_rays(camera_R_to_world, sensor_size, resolution, focal_length)

    # Raycast the scene, which returns the world coordinates of the first intersection
    # of each ray with objects in the scene.
    hit_pts = jax.vmap(raycast, in_axes=(None, None, 0))(scene, camera_origin, rays)

    # Render to depth
    depth_image = render_depth(
        hit_pts, camera_origin, camera_R_to_world, resolution, max_dist=10.0
    ).reshape(resolution)

    # Make sure the depth image has the right shape
    assert depth_image.shape[0] == resolution[0]
    assert depth_image.shape[1] == resolution[1]

    plt.imshow(depth_image.T, cmap="Greys")
    plt.colorbar()
    plt.show()
