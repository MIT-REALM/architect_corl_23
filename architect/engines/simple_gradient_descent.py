import equinox as eqx
import jax
import jax.numpy as jnp
import time
import jax.tree_util
from matplotlib import pyplot as plt
from beartype import beartype
from jax.nn import logsumexp
from jaxtyping import Array, Float, jaxtyped, PyTree


"""
A simple gradient descent engine for optimizing the design parameter (a policy in most cases).
This engine was written with the intention of testing turtlebot and tellodrone source seeking.

args:
    iterations: the number of iterations in which to train the design parameter dp
    rate: learning rate for gradient based optimization
    loss_fn: the function/game for which dp is being optimized (loss_fn must return loss)
    eps: a tuple of exogenous parameters
    dp: design parameters (usually a Policy)
returns:
    dp: updated design parameter after gradient based optimization
"""
@jaxtyped
@beartype
def simple_grad_descent(N: int, T: float, iterations: int, rate: float, loss_fn, eps: PyTree[Float[Array, "..."]], dp: PyTree[Float[Array, "..."]]) -> PyTree[Float[Array, "..."]]:
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, allow_int = True))
    # Train
    iters = iterations
    learning_rate = rate 
    t_start = time.time()
    for i in range(iters):
        loss, grads = loss_and_grad(dp,eps) 
        dp = jax.tree_util.tree_map(lambda m, g: m - learning_rate * g, dp, grads) 
        if i % 1000 == 0 or i == iters - 1:
            print(f"Iter {i}, loss {loss}")
    t_end = time.time()
    print("Trained on {:,d} environmental interactions in {:.2f} seconds; final loss = {:.3f}".format(int(T * N * iters), t_end - t_start, loss))
    return dp
