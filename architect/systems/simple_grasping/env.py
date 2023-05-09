"""Define a simple grasping environment."""
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, NamedTuple, Optional, Tuple
from jaxtyping import Array, Float, jaxtyped

from architect.systems.components.sensing.vision.render import (
    CameraExtrinsics,
    CameraIntrinsics,
    pinhole_camera_rays,
    raycast,
    render_color,
    render_depth,
    render_shadows,
)
from architect.systems.components.sensing.vision.shapes import (
    Box,
    Cylinder,
    Halfspace,
    Scene,
    SDFShape,
    Subtraction,
)
from architect.systems.components.sensing.vision.util import look_at


class MugHandle(SDFShape):
    """Represent a mug handle that is colored to highlight grasp affordances."""

    c: Float[Array, " 3"]
    R: Float[Array, "3 3"]
    sharpness: float = 1.0

    def __call__(self, x: Float[Array, " 3"]) -> Float[Array, ""]:
        """Return the SDF of the mug handle at the given point"""
        mug_handle = Box(
            center=self.c,
            extent=jnp.array([0.01, 0.17, 0.25]),
            rotation=self.R,
            c=jnp.array([1.0, 1.0, 1.0]),
            rounding=jnp.array(0.07),
        )
        mug_handle_hole = Box(
            center=self.c,
            extent=jnp.array([0.06, 0.08, 0.08]),
            rotation=self.R,
            c=jnp.zeros(3),
            rounding=jnp.array(0.05),
        )
        mug_handle = Subtraction(
            mug_handle,
            mug_handle_hole,
            sharpness=self.sharpness,
        )

        return mug_handle(x)

    def color(self, x: Float[Array, " 3"]) -> Float[Array, " 3"]:
        """Return the color of the mug handle at the given point"""
        # Highlight grasp affordances, which are on the sides of the handle.

        # Get the normal vector at the given point
        normal = jax.grad(self)(x)
        # Transform to the handle frame
        normal = self.R.T @ normal
        # Normalize
        normal = normal / jnp.sqrt((normal**2).sum() + 1e-3)

        # If the normal is pointing in the local +/- x direction, color the
        # handle white to indicate a grasp affordance
        normal_x = normal[0] ** 2
        color = jnp.ones(3) * normal_x

        # Also prefer the top half of the handle
        x_handle = self.R.T @ (x - self.c)
        color *= jax.nn.sigmoid(50 * x_handle[2])

        return color


def make_grasping_scene(
    mug_location: Float[Array, " 2"],
    mug_rotation: Float[Array, ""],
    sharpness: float = 1.0,
) -> SDFShape:
    """Make a scene with the object on a table"""
    # Make the tabletop
    tabletop = Halfspace(
        point=jnp.array([0.0, 0.0, -0.04]),
        normal=jnp.array([0.0, 0.0, 1.0]),
        c=jnp.zeros(3),
    )

    # Make the object (a mug!)
    mug_body = Cylinder(
        center=jnp.array([mug_location[0], mug_location[1], 0.25]),
        radius=jnp.array(0.3),
        height=jnp.array(0.5),
        rotation=jnp.eye(3),
        c=jnp.zeros(3),
    )
    mug_interior = Cylinder(
        center=jnp.array([mug_location[0], mug_location[1], 0.3]),
        radius=jnp.array(0.25),
        height=jnp.array(0.5),
        rotation=jnp.eye(3),
        c=jnp.zeros(3),
    )
    handle_rotation = jnp.array(
        [
            [jnp.cos(mug_rotation), -jnp.sin(mug_rotation), 0.0],
            [jnp.sin(mug_rotation), jnp.cos(mug_rotation), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    handle_center = jnp.array([mug_location[0], mug_location[1], 0.25])
    handle_center += handle_rotation @ jnp.array([0.0, 0.42, 0.0])
    mug_handle = MugHandle(
        c=handle_center,
        R=handle_rotation,
        sharpness=sharpness,
    )

    # Highlight grasp affordances on the handle by making the sides of the handle white
    # and the top and bottom black. This corresponds to good grasp affordances that
    # are in the local

    return Scene(
        shapes=[
            tabletop,
            Subtraction(
                mug_body,
                mug_interior,
                sharpness=sharpness,
            ),
            mug_handle,
        ],
        sharpness=sharpness,
    )


def normal_at_point(
    scene: SDFShape,
    contact_pt: Float[Array, "3"],
) -> Float[Array, "3"]:
    """Compute the contact normal on the mug at the given point.

    Args:
        scene: the scene to render
        contact_pt: the point of contact
    """
    # Compute the unit normal vector
    normal = jax.grad(scene)(contact_pt)
    normal = normal / jnp.sqrt(jnp.sum(normal**2) + 1e-3)
    return normal


def clamp_force_in_friction_cone(
    clamp_force: Float[Array, "3"],
    normal: Float[Array, "3"],
    mu: float = 0.5,
    sharpness: float = 50.0,
) -> tuple[Float[Array, "3"], Float[Array, "3"]]:
    """Check if the clamp force is in the friction cone.

    Args:
        clamp_force: the clamp force
        normal: the contact normal
        mu: the coefficient of friction
        sharpness: the sharpness of sticking/slipping transition

    Returns:
        The clamp force in the normal and tangential directions, where the force
        in the tangential direction is resisted by friction (non-zero clamp force
        indicates slipping).
    """
    # Compute the normal component of the clamp force
    normal_f = jnp.dot(clamp_force, normal)

    # Compute the tangential component of the clamp force
    tangential_f = clamp_force - normal_f * normal

    # Adjust for friction
    tangential_f = tangential_f * jax.nn.sigmoid(
        sharpness * (jnp.sqrt(jnp.sum(tangential_f**2) + 1e-3) - mu * normal_f)
    )

    # Check if the tangential component is in the friction cone
    return normal_f * normal, tangential_f


def render_rgbd(
    scene: SDFShape,
    camera_pos: Float[Array, "3"],
    max_dist: float = 2.0,
) -> Tuple[Float[Array, "H W"], Float[Array, "H W 3"]]:
    """Render the color and depth image of this scene from the given camera pose.

    The depth image is accurate, but the color image is used as an affordance mask
    that is white where there are good grasp affordances and black otherwise.

    Args:
        scene: the scene to render
        camera_pos: the position of the camera
        max_dist: the maximum distance to render

    Returns:
        The depth and color images of the scene
    """

    # Set the camera parameters
    intrinsics = CameraIntrinsics(
        focal_length=0.1,
        sensor_size=(0.1, 0.1),
        resolution=(64, 64),
    )
    R = look_at(camera_pos, jnp.zeros(3))
    extrinsics = CameraExtrinsics(
        camera_origin=camera_pos,
        camera_R_to_world=R,
    )

    # Render the scene
    rays = pinhole_camera_rays(intrinsics, extrinsics)
    hit_pts = jax.vmap(raycast, in_axes=(None, None, 0, None, None))(
        scene, extrinsics.camera_origin, rays, 100, 10 * max_dist
    )
    depth_image = render_depth(
        hit_pts, intrinsics, extrinsics, max_dist=max_dist
    ).reshape(intrinsics.resolution)
    color_image = render_color(hit_pts, scene).reshape(*intrinsics.resolution, 3)

    return depth_image, color_image


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from architect.systems.simple_grasping.policy import AffordancePredictor

    # Create the scene
    camera_pos = jnp.array([-1.0, 0.0, 1.0])
    mug_pos = jnp.array([0.0, 0.0])
    mug_rot = jnp.pi * 1 / 4
    scene = make_grasping_scene(
        mug_location=mug_pos,
        mug_rotation=mug_rot,
        sharpness=50.0,
    )
    depth_image, color_image = render_rgbd(scene, camera_pos)

    # Reduce color image to single channel
    color_image = jnp.mean(color_image, axis=-1)

    # Create the affordance predictor
    key = jax.random.PRNGKey(0)
    affordance_predictor = AffordancePredictor(key)
    prediction = affordance_predictor(depth_image)

    _, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].set_title("Depth")
    axs[0].imshow((depth_image).T)
    axs[1].set_title("GT Affordance")
    axs[1].imshow(color_image.T)
    axs[2].set_title("Predicted Affordance")
    axs[2].imshow((prediction).T)
    plt.savefig("test.png")

    contact_pt = jnp.array([0.15, 0.5, 0.2])
    normal = normal_at_point(scene, contact_pt)
    normal_f, tangential_f = clamp_force_in_friction_cone(
        clamp_force=jnp.array([-1.0, 0.0, 0.0]),
        normal=normal,
    )
    print("finger 1")
    print("normal: ", normal)
    print("normal_f: ", normal_f)
    print("tangential_f: ", tangential_f)
    print("||tangential_f||: ", jnp.sqrt(jnp.sum(tangential_f**2)))

    contact_pt = jnp.array([-0.15, 0.5, 0.2])
    normal = normal_at_point(scene, contact_pt)
    normal_f, tangential_f = clamp_force_in_friction_cone(
        clamp_force=jnp.array([1.0, 0.0, 0.0]),
        normal=normal,
    )
    print("finger 2")
    print("normal: ", normal)
    print("normal_f: ", normal_f)
    print("tangential_f: ", tangential_f)
    print("||tangential_f||: ", jnp.sqrt(jnp.sum(tangential_f**2)))
