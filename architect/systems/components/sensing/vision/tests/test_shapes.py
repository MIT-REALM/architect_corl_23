"""Test implementation of SDF shapes."""
import jax.numpy as jnp

from architect.systems.components.sensing.vision.shapes import (
    Sphere,
    Halfspace,
    Box,
    Scene,
)


def test_sphere():
    """Test the sphere SDF."""
    # Create a sphere centered at the origin with radius one
    sphere = Sphere(
        jnp.array([0.0, 0.0, 0.0]),  # center
        jnp.array(1.0),  # radius
    )
    # Test the SDF at a few points
    assert jnp.isclose(sphere(jnp.array([1.0, 0.0, 0.0])), 0.0)  # on the surface
    assert jnp.isclose(sphere(jnp.array([2.0, 0.0, 0.0])), 1.0)  # outside
    assert jnp.isclose(sphere(jnp.array([0.0, 0.0, 0.0])), -1.0)  # inside


def test_halspace():
    """Test the halfspace SDF."""
    # Create a halfspace with normal pointing in the positive x direction and
    # containing the origin
    halfspace = Halfspace(
        jnp.array([1.0, 0.0, 0.0]),  # normal
        jnp.array([0.0, 0.0, 0.0]),  # point
    )
    # Test the SDF at a few points
    assert jnp.isclose(halfspace(jnp.array([0.0, 0.0, 0.0])), 0.0)  # on the surface
    assert jnp.isclose(halfspace(jnp.array([1.0, 0.0, 0.0])), 1.0)  # outside
    assert jnp.isclose(halfspace(jnp.array([-1.0, 0.0, 0.0])), -1.0)  # inside


def test_box():
    """Test the box SDF."""
    # Create a box centered at the origin with sides of length 1, without rotation
    box = Box(
        jnp.array([0.0, 0.0, 0.0]),  # center
        jnp.array([1.0, 1.0, 1.0]),  # extent
        jnp.eye(3),  # rotation
    )
    # Test the SDF at a few points
    assert box(jnp.array([0.0, 0.0, 0.0])) < 0  # in the center
    assert jnp.isclose(box(jnp.array([0.5, 0.0, 0.0])), 0.0)  # on the face
    assert jnp.isclose(box(jnp.array([0.5, 0.5, 0.5])), 0.0)  # on the corner
    assert box(jnp.array([2.0, 0.0, 0.0])) > 0.0  # outside
    assert box(jnp.array([-3.0, -3.0, 0.0])) > 0.0  # outside


def test_box_rotation():
    """Test the box SDF with rotation."""
    # Create a box centered at the origin with sides of length 1, rotated 45 degrees
    angle = jnp.pi / 4
    # about the z axis
    box = Box(
        jnp.array([0.0, 0.0, 0.0]),  # center
        jnp.array([1.0, 1.0, 1.0]),  # extent
        jnp.array(  # rotation
            [
                [jnp.cos(angle), -jnp.sin(angle), 0.0],
                [jnp.sin(angle), jnp.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )

    # Test the SDF at a few points
    assert box(jnp.array([0.0, 0.0, 0.0])) < 0  # in the center
    # on the face
    assert jnp.isclose(
        box(jnp.array([0.5 * jnp.cos(angle), 0.5 * jnp.sin(angle), 0.0])),
        0.0,
        atol=1e-7,
    )
    # on the corner
    assert jnp.isclose(
        box(jnp.array([0.5 * jnp.cos(angle), 0.5 * jnp.sin(angle), 0.5])),
        0.0,
        atol=1e-7,
    )
    # Outside
    assert box(jnp.array([2.0, 0.0, 0.0])) > 0.0
    assert box(jnp.array([-3.0, -3.0, 0.0])) > 0.0


def test_scene():
    """Test the combining multiple shapes into a single scene."""
    # Create some test shapes
    sphere1 = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(1.0))
    sphere2 = Sphere(center=jnp.array([0.0, 0.0, 2.0]), radius=jnp.array(1.0))
    box = Box(
        center=jnp.array([2.0, 0.0, 0.0]),
        extent=jnp.array([1.0, 1.0, 1.0]),
        rotation=jnp.eye(3),
    )

    # Combine them into a scene
    scene_sdf = Scene([sphere1, sphere2, box])

    # Test the scene SDF at a few points
    assert scene_sdf(jnp.array([0.0, 0.0, 0.0])) < 0  # inside sphere1
    assert scene_sdf(jnp.array([0.0, 0.0, 2.0])) < 0  # inside sphere2
    assert scene_sdf(jnp.array([2.0, 0.0, 0.0])) < 0  # inside box
    assert scene_sdf(jnp.array([0.0, 2.0, 0.0])) > 0  # outside all shapes
