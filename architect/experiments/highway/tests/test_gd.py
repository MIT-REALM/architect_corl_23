"""Test gradient descent on various potential functions."""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from architect.systems.components.sensing.vision.shapes import Scene, Sphere
from architect.systems.highway.highway_scene import HighwayScene

if __name__ == "__main__":
    # Can we optimize for a collision?
    scene = HighwayScene(num_lanes=3, lane_width=5.0, segment_length=200.0)

    def get_distance_to_collision(car_state) -> float:
        initial_non_ego_states = jnp.array(
            [
                [-90.0, -3.0, 0.0, 7.0],
                [-70, 3.0, 0.0, 8.0],
            ]
        )

        return scene.check_for_collision(
            car_state[:3],  # trim out speed; not needed for collision checking
            initial_non_ego_states[:, :3],
            include_ground=False,
        )

    # Can we optimize for a collision?
    lr = 1e-2
    steps = 500
    value_and_grad_fn = jax.jit(jax.value_and_grad(get_distance_to_collision))
    ep = jnp.array([-100.0, -3.0, 0.0, 10.0])
    values = []
    for i in range(steps):
        v, g = value_and_grad_fn(ep)
        values.append(v)
        print(f"Step {i}: potential: {v}, grad norm: {jnp.linalg.norm(g)}")
        ep = ep - lr * g  # gradient DESCENT to minimize distance

    plt.plot(values)
    plt.show()
