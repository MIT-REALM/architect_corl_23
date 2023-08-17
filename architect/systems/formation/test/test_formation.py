import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
import math
from jax.nn import logsumexp
from jaxtyping import Array, Float, jaxtyped
from matplotlib import pyplot as plt # this is just for testing, will get rid of this later

from architect.systems.formation.formation import (
    make_adj_matrix,
    init_bot_positions,
    current_dist_angle,
    velocity_vectors
)

# Note before testing: activate conda env first to have access to architect module
# >>> conda activate architect_private_env
# >>> cd architect/systems/formation

def test_pytest_is_working():
   num = 25
   assert math.sqrt(num) == 5

# --- current_dist_angle() ---

# def test_current_dist_angle_1():
#     current_positions = jnp.array([[4,6], [2,3], [1,-1], [-3,-5]])
#     adj_matrix = make_adj_matrix(4, 'single-chain')
#     current_r_theta = current_dist_angle(current_positions, adj_matrix)

#     answer = jnp.array([[0,0], 
#                          [3.605551275464,0.982793723247329],
#                          [4.123105625617661,1.325817663668],
#                          [1.5395564933646284,0.7853981633974483]])
#     isthisright = (current_r_theta == answer).all()
#     assert current_r_theta == jnp.all(isthisright) # not right values lol

# --- velocity_vectors() ---

# def test_velocity_vectors_1():
#     current2 = jnp.array([[0,0], [4,0], [4,0]])
#     desired2 = jnp.array([[0,0], [3.605551275464,0.982793723247329], [4.242640687119285,0.7853981633974483]])
#     adj_matrix = make_adj_matrix(3, 'single-chain')
#     vectors = velocity_vectors(current2, desired2, adj_matrix)

#     correct = jnp.array()
#     assert vectors == jnp.array([3, 4]) # not right values lol

# --- current_dist_angle() & velocity_vectors() ---

def formation_adjustment(current_pos, desired_pos, shape):
    n = jnp.shape(current_pos)[0]
    adj_matrix = make_adj_matrix(n, shape)
    current_r_theta = current_dist_angle(current_pos, adj_matrix)
    desired_r_theta = current_dist_angle(desired_pos, adj_matrix)
    return velocity_vectors(current_r_theta, desired_r_theta, adj_matrix)

# Single-chain testing

def test_adjustment_single_chain_1():
    jnp.set_printoptions(suppress=True)
    current_pos = jnp.array([[2,3],
                            [-2,3],
                            [-6,3]])
    desired_pos = jnp.array([[2,3],
                             [0,0],
                             [-3,-3]])
    result = formation_adjustment(current_pos, desired_pos, 'single-chain')
    expected = jnp.array([[0,0],
                          [2,-3],
                          [3,-6]])
    isitclose = jnp.allclose(result, expected, atol=1e-05)
    assert isitclose == True

def test_adjustment_single_chain_2():
    jnp.set_printoptions(suppress=True)
    current_pos = jnp.array([[4,6],
                             [2,3],
                             [1,-1],
                             [-3,-5]])
    desired_pos = jnp.array([[4,6],
                             [-1,5],
                             [-6,6],
                             [-8,3]])
    result = formation_adjustment(current_pos, desired_pos, 'single-chain')
    expected = jnp.array([[0,0],
                          [-3,2],
                          [-7,7],
                          [-5,8]])
    isitclose = jnp.allclose(result, expected, atol=1e-05)
    assert isitclose == True

def test_adjustment_single_chain_3():
    jnp.set_printoptions(suppress=True)
    current_pos = jnp.array([[4,6],
                             [5,0],
                             [8,-3],
                             [0,-5],
                             [12,-10]])
    desired_pos = jnp.array([[4,6],
                             [-1,5],
                             [-5,3],
                             [-9,4],
                             [-14,0]])
    result = formation_adjustment(current_pos, desired_pos, 'single-chain')
    expected = jnp.array([[0,0],
                          [-6,5],
                          [-13,6],
                          [-9,9],
                          [-26,10]])
    isitclose = jnp.allclose(result, expected, atol=1e-05)
    assert isitclose == True

def test_adjustment_single_chain_4():
    jnp.set_printoptions(suppress=True)
    current_pos = jnp.array([[20,25],
                             [5,20],
                             [-10,25],
                             [-12,10],
                             [-14,24],
                             [-25,15],
                             [-30,20]])
    desired_pos = jnp.array([[20,25],
                             [25,15],
                             [15,10],
                             [15,5],
                             [15,0],
                             [5,-10],
                             [-34,-10]])
    result = formation_adjustment(current_pos, desired_pos, 'single-chain')
    expected = jnp.array([[0,0],
                          [20,-5],
                          [25,-15],
                          [27,-5],
                          [29,-24],
                          [30,-25],
                          [-4,-30]])
    isitclose = jnp.allclose(result, expected, atol=1e-05)
    assert isitclose == True

# All-follow-leader testing

def test_adjustment_all_follow_1():
    jnp.set_printoptions(suppress=True)
    current_pos = jnp.array([[2,6],
                            [-2,4],
                            [0,3],
                            [3,3]])
    desired_pos = jnp.array([[2,6],
                             [-2,2],
                             [2,2],
                             [6,2]])
    result = formation_adjustment(current_pos, desired_pos, 'single-chain')
    expected = jnp.array([[0,0],
                          [0,-2],
                          [2,-1],
                          [3,-1]])
    print(result)
    print(expected)
    isitclose = jnp.allclose(result, expected, atol=1e-05)
    assert isitclose == True

def test_adjustment_all_follow_2():
    jnp.set_printoptions(suppress=True)
    current_pos = jnp.array([[1,1],
                             [1,8],
                             [-5,4],
                             [-3,-2],
                             [1,-3]])
    desired_pos = jnp.array([[1,1],
                             [-3,6],
                             [-10,0],
                             [6,1],
                             [6,7]])
    result = formation_adjustment(current_pos, desired_pos, 'single-chain')
    expected = jnp.array([[0,0],
                          [-4,-2],
                          [-5,-4],
                          [9,3],
                          [5,10]])
    isitclose = jnp.allclose(result, expected, atol=1e-05)
    assert isitclose == True
    
# V-Formation testing

def test_adjustment_v_formation_1():
    current_pos = jnp.array([[1,7],
                             [-2,5],
                             [1,4],
                             [-5,4],
                             [0,2]])
    desired_pos = jnp.array([[1,7],
                             [1,11],
                             [5,8],
                             [2,15],
                             [9,10]])
    result = formation_adjustment(current_pos, desired_pos, 'single-chain')
    expected = jnp.array([[0,0],
                          [3,6],
                          [4,4],
                          [7,11],
                          [9,8]])
    isitclose = jnp.allclose(result, expected, atol=1e-05)
    assert isitclose == True
    
def test_adjustment_v_formation_2(): # this test sucks, i'm right
    current_pos = jnp.array([[1,7],
                             [-2,5],
                             [1,4],
                             [-5,4],
                             [0,2],
                             [-4,-2],
                             [0,-6]])
    desired_pos = jnp.array([[1,7],
                             [4,0],
                             [9,5],
                             [6,-4],
                             [16,2],
                             [10,-8],
                             [16,-2]])
    result = formation_adjustment(current_pos, desired_pos, 'single-chain')
    expected = jnp.array([[0,0],
                          [6,-5],
                          [8,1],
                          [11,-8],
                          [16,0],
                          [14,-6],
                          [16,4]])
    isitclose = jnp.allclose(result, expected, atol=1e-05)
    assert isitclose == True
