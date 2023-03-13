"""Define a highway scene with a variable number of lanes and cars."""
from jaxtyping import Float, Array, jaxtyped
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, NamedTuple

from architect.systems.components.sensing.vision.shapes import (
    SDFShape,
    Halfspace,
    Box,
    Scene,
)
from architect.systems.components.sensing.vision.render import (
    pinhole_camera_rays,
    raycast,
    render_depth,
    CameraIntrinsics,
    CameraExtrinsics,
)


@beartype
class Car(NamedTuple):
    """Represent a car as a composite of primitive shapes.

    Attributes:
        height: the height of the car
        width: the width of the car
        length: the length of the car
    """

    height: Float[Array, " 1"] = jnp.array(2.0)
    width: Float[Array, " 1"] = jnp.array(3.0)
    length: Float[Array, " 1"] = jnp.array(4.5)

    @jaxtyped
    @beartype
    def get_shapes(self, state: Float[Array, " 3"]) -> List[SDFShape]:
        """Return a list of primitive shapes representing this car

        Args:
            state: the [x, y, heading] state of the car

        Returns:
            A list of SDF shapes representing the car
        """
        # Create the primitive shapes that make up the car
        body = Box(
            center=jnp.array([state[0], state[1], self.height / 2]),
            extent=jnp.array([self.length, self.width, self.height]),
            rotation=jnp.array(
                [
                    [jnp.cos(state[2]), -jnp.sin(state[2]), 0],
                    [jnp.sin(state[2]), jnp.cos(state[2]), 0],
                    [0, 0, 1],
                ]
            ),
        )
        return [body]


@beartype
class HighwayScene:
    """Represent a highway scene with multiple cars and lanes."""

    ground: Halfspace
    walls: List[Box]
    car: Car
    lane_width: float

    @beartype
    def __init__(
        self,
        num_lanes: int,
        lane_width: float = 3.7,
        segment_length: float = 100.0,
    ):
        """Initialize the highway scene.

        Args:
            num_lanes: the number of lanes in the scene
            lane_width: the width of each lane
            segment_length: the length of the highway segment to represent
        """
        # Create the ground plane and walls on each side of the highway
        self.ground = Halfspace(
            point=jnp.array([0.0, 0.0, 0.0]), normal=jnp.array([0.0, 0.0, 1.0])
        )
        self.walls = [
            Box(
                center=jnp.array([0.0, -lane_width * num_lanes / 2.0 - 0.5, 0.0]),
                extent=jnp.array([segment_length, 1.0, 10.0]),
                rotation=jnp.eye(3),
            ),
            Box(
                center=jnp.array([0.0, lane_width * num_lanes / 2 + 0.5, 0.0]),
                extent=jnp.array([segment_length, 1.0, 10.0]),
                rotation=jnp.eye(3),
            ),
        ]
        self.car = Car()  # re-used for all cars
        self.lane_width = lane_width

    @jaxtyped
    @beartype
    def _get_shapes(
        self, car_states: Float[Array, "n_car 3"], sharpness: float = 100.0
    ) -> SDFShape:
        """Return an SDF representation this scene."""
        car_shapes = [self.car.get_shapes(state) for state in car_states]
        shapes = (
            [self.ground]
            + self.walls
            + [shape for sublist in car_shapes for shape in sublist]
        )
        return Scene(shapes=shapes, sharpness=sharpness)

    @jaxtyped
    @beartype
    def render_depth(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        car_states: Float[Array, "n_car 3"],
        max_dist: float = 50.0,
        sharpness: float = 100.0,
    ) -> Float[Array, "res_x res_y"]:
        """Render the depth image of this scene from the given camera pose.

        Args:
            intrinsics: the camera intrinsics
            extrinsics: the camera extrinsics
            car_states: the [x, y, heading] state of each car
            max_dist: the maximum distance to render
            sharpness: the sharpness of the scene

        Returns:
            The depth image of the scene
        """
        # Make the scene (a composite of SDF shapes)
        scene = self._get_shapes(car_states, sharpness=sharpness)

        # Render the scene
        rays = pinhole_camera_rays(intrinsics, extrinsics)
        hit_pts = jax.vmap(raycast, in_axes=(None, None, 0))(
            scene, extrinsics.camera_origin, rays
        )
        depth_image = render_depth(
            hit_pts, intrinsics, extrinsics, max_dist=max_dist
        ).reshape(intrinsics.resolution)
        return depth_image


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from architect.systems.components.sensing.vision.util import look_at

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

    plt.imshow(depth_image.T, cmap="Greys")
    plt.colorbar()
    plt.show()
