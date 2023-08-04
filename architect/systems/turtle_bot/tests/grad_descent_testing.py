import equinox as eqx
import jax
import jax.numpy as jnp
import time
import jax.tree_util
from matplotlib import pyplot as plt
from beartype import beartype
from jax import tree_util
from jax.nn import logsumexp
from jaxtyping import Array, Float, jaxtyped, PyTree
from architect.systems.components.dynamics.dubins import dubins_next_state
from architect.systems.turtle_bot.turtle_bot_types import Policy, EnvironmentState
from architect.systems.turtle_bot.tests.grad_descent_game_test import (
    Game
)
from architect.engines.simple_gradient_descent import simple_grad_descent

# on the jupyter notebook, make a copy of turtle bots and use only single integrator

# This simulator tests the behavior of turtle bots in the source-seeking problem. 
# The control inputs are outputs of a neural network policy, which is trained using 
# simple gradient descent. The policy takes into consideration a history of positions 
# and a history of concentrations. The target positions and strengths (sigmas) of the sources 
# and the initial conditions/positions (of which there are N) are exogenous parameters.

# Initialization
game = Game(
    memory_length = 10,
    n_targets = 2
)
prng_key = jax.random.PRNGKey(1)
prng_key, subkey = jax.random.split(prng_key)
policy = Policy(subkey,10)


N=30
memory_length = 10
# Define exogenous parameters
prng_key = jax.random.PRNGKey(0)
x_inits = 1 * jax.random.normal(prng_key, shape=(N, 3)) 
target_pos = jnp.array([[-1.5,0.0],[0.5,0.0]])
num_targets = target_pos.shape[0]
logsigma = jnp.log(jnp.array([0.3, 0.2]))

# Define parameters fro gradient descent
T = game.duration
iterations = 6000
rate = 1e-2
loss_fn = game.loss_fn
target = target_pos
sig = logsigma
init = x_inits
dp = policy
eps = (target,sig,init)
new_dp = simple_grad_descent(N,T,iterations,rate,loss_fn,eps,dp)

# Define the Game to be solved using the optimized design parameter
e_sigma = jnp.exp(logsigma)
exogenous_env = EnvironmentState(target_pos, e_sigma, x_inits)
n_targets = num_targets
# initialize x_hists
def make_x_hist(x_init: Float[Array, "N 3"]) -> Float[Array, "N memory_length 3"]:
    x_hist = jnp.array([x_init for i in range(memory_length)])
    return x_hist
make_x_hists = jax.vmap(make_x_hist)
x_hists = make_x_hists(x_inits)
# initialize conc_hists
def make_conc_hist(x_init: Float[Array, "N 3"]) -> Float[Array, "N memory_length"]:
    conc = exogenous_env.get_concentration(x_init[0], x_init[1], n_targets)
    conc_hist = jnp.array([conc for i in range(memory_length)])
    return conc_hist
make_conc_hists = jax.vmap(make_conc_hist)
conc_hists = make_conc_hists(x_inits)

# Define a function to execute one step in the game
def step_policy(carry, dummy_input):
    x_hist, conc_hist, x_current = carry
    control_inputs = jax.nn.tanh(new_dp(x_hist, conc_hist))
    # finding the next state (x, y, theta)
    x_next = dubins_next_state(
        x_current,
        control_inputs,
        actuation_noise=jnp.array([0.0, 0.0, 0.0]),
        dt=0.1,
    )
    x_hist = jnp.delete(x_hist, -1, 0)
    conc_hist = jnp.delete(conc_hist, -1, 0)
    # insert the next state into the beginning of the list of position history
    x_hist = jnp.insert(x_hist, 0, x_next, axis=0)
    # insert the newest concentration found at x_next
    conc_hist = jnp.insert(
        conc_hist,
        0,
        exogenous_env.get_concentration(x_next[0], x_next[1], n_targets),
        axis=0,
    )
    return (x_hist, conc_hist, x_next), (x_next, control_inputs)

simulate_scan = lambda x_init, x_hist, conc_hist: jax.lax.scan(
    step_policy, (x_hist, conc_hist, x_init), xs=None, length=T
)[1]
simulate_batched = jax.vmap(simulate_scan)
trajectory, control = simulate_batched(x_inits, x_hists, conc_hists)
control = control**2

# Plotting
delta = 0.025
x = jnp.arange(-3.0, 3.0, delta)
y = jnp.arange(-2.0, 2.0, delta)
X, Y = jnp.meshgrid(x, y)
def get_concentration(x, y):
    sigmas = jnp.exp(logsigma)
    make_gaussian = lambda target, sig: 1/sig*jax.lax.exp(-1/sig* ((target[0]-x)**2 + (target[1]-y)**2))
    concs = [make_gaussian(target[n], sigmas[n]) for n in range(num_targets)]
    concentration = sum(concs)
    concentration = jnp.take(concentration, 0)
    return concentration
data = jax.vmap(jax.vmap(get_concentration))(X,Y)
print (data.shape)
plt.contour(X,Y, data)
xs=trajectory
print (jnp.shape(xs))
for i in range(N):
    plt.plot(xs[i,:,0], xs[i,:,1])
    plt.plot(xs[i,0,0], xs[i,0,1], "bo")
    plt.plot(xs[i,-1,0], xs[i,-1,1], "ro")
plt.title(f"Run for duration of {T} seconds, with {N} initial conditions. Final cost: {game.loss_fn(new_dp,eps)}")
plt.show()

