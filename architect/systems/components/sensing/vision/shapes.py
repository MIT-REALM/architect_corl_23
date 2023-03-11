"""Represent various shapes using Signed Distance Functions (SDFs).

Note that these are intended for the purpose of rendering a view of a 3D scene, not
for contact simulation or collision detection. As a result, these functions may not
return a fully accurate SDF, but they should be sufficient for rendering.
"""
from abc import ABC, abstractmethod

import jax.numpy as jnp
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype
from beartype.typing import List
import equinox as eqx

from architect.utils import softmin


class SDFShape(ABC, eqx.Module):
    """Abstract base class for shapes defined via signed distance functions."""

    @abstractmethod
    def __call__(self, x: Float[Array, " 3"]) -> Float[Array, ""]:
        """Compute the SDF at a given point.

        Args:
            x: a point in world coordinates

        Returns:
            signed distance from the point to the shape
        """


class Scene(SDFShape):
    """Represent a scene using a SDF.

    Attributes:
        shapes: the shapes in the scene
        sharpness: the sharpness of the SDF
    """

    shapes: List[SDFShape]
    sharpness: float = 1.0

    def __call__(self, x: Float[Array, " 3"]) -> Float[Array, ""]:
        """Compute the SDF at a given point.

        Args:
            x: a point in world coordinates

        Returns:
            signed distance from the point to the scene (positive outside)
        """
        distances = jnp.array([shape(x) for shape in self.shapes])
        return softmin(distances, self.sharpness)


@jaxtyped
@beartype
class Sphere(SDFShape):
    """Represent a sphere using a SDF.

    Attributes:
        center: the center of the sphere in world coordinates
        radius: Radius of the sphere.
    """

    center: Float[Array, " 3"]
    radius: Float[Array, ""]

    def __call__(self, x: Float[Array, " 3"]) -> Float[Array, ""]:
        """Compute the SDF at a given point.

        Args:
            x: a point in world coordinates

        Returns:
            signed distance from the point to the sphere (positive outside)
        """
        return jnp.linalg.norm(x - self.center) - self.radius


class Halfspace(SDFShape):
    """Represent a halfspace using a SDF.

    Attributes:
        normal: the normal vector pointing to the exterior of the halfspace
        point: a point on the plane bounding the halfspace
    """

    normal: Float[Array, " 3"]
    point: Float[Array, ""]

    def __call__(self, x: Float[Array, " 3"]) -> Float[Array, ""]:
        """Compute the SDF at a given point.

        Args:
            x: a point in world coordinates

        Returns:
            signed distance from the point to the halfspace
        """
        return jnp.dot(x - self.point, self.normal)


class Box(SDFShape):
    """Represent a box using a SDF.

    Attributes:
        center: the center of the box in world coordinates
        extent: the extent of the box in each dimension
        R_to_world: the 3D rotation matrix from the box frame to the world frame
    """

    center: Float[Array, " 3"]
    extent: Float[Array, " 3"]
    rotation: Float[Array, "3 3"]

    def __call__(self, x: Float[Array, " 3"]) -> Float[Array, ""]:
        """Compute the SDF at a given point.

        Args:
            x: a point in world coordinates

        Returns:
            signed distance from the point to the box
        """
        # Get the offset from the box center to the point, and rotate it into the box
        # frame
        offset = self.rotation.T @ (x - self.center)

        # Compute the distance to the box in the box frame (which is axis-aligned)
        distance_to_box = jnp.abs(offset) - self.extent / 2.0

        sdf = jnp.max(distance_to_box)
        return sdf
