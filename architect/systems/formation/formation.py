import equinox as eqx
import jax
import jax.numpy as jnp
# from beartype import beartype
from jax.nn import logsumexp
from jaxtyping import Array, Float, jaxtyped
from matplotlib import pyplot as plt # this is just for testing, will get rid of this later


# Adjacency matrix
def make_adj_matrix(n, shape):
    '''
    INPUT
    * n (int): number of robots/size of array
    * shape (str): 'all-follow-leader', 'single-chain', 'v-formation'

    OUTPUT
    * (n,n) array representing the adj matrix of the given formation shape
    '''
    base_matrix = jnp.zeros((n,n))

    shapes = {'all-follow-leader',
             'single-chain',
             'v-formation'}

    if shape not in shapes:
        return base_matrix
    elif shape == 'all-follow-leader':
        base_matrix = base_matrix.at[1:n,0].set(1)
    elif shape == 'single-chain':
        for i in range(n-1):
            base_matrix = base_matrix.at[i+1, i].set(1)
    elif shape == 'v-formation':
        base_matrix = base_matrix.at[1:3, 0].set(1)
        for i in range(n-2):
            base_matrix = base_matrix.at[i+2, i].set(1)
    return base_matrix


# Initial bot positions
def init_bot_positions(n):
    '''
    INPUT
    * n (int): number of robots

    OUTPUT
    * (n,2) array of x- and y- coordinates for each nth robot
    '''

    init_bot_positions = jnp.zeros((n,2))

    # arrange bots along a circle
    radius = 1.0
    theta = jnp.linspace(0, 2 * jnp.pi, n+1)
    x = radius * jnp.cos(theta)
    y = radius * jnp.sin(theta)
    x = jnp.round(x, decimals=2)
    y = jnp.round(y, decimals=2)
    jnp.set_printoptions(suppress=True)

    if n == 1:
        x = 0.5
        y = 0.5
    if n == 0:
        x = jnp.nan
        y = jnp.nan

    init_bot_positions = jnp.column_stack((x, y))
    # n > 1 gives a repeat of the first coordinate, removing that repeat
    if n > 1:
        init_bot_positions = init_bot_positions[:-1]

    return init_bot_positions


def current_dist_angle(bot_positions, adj_matrix):
    '''
    INPUT:
    * bot_positions: (n,2) array of (x,y) coordinates for each bot
    * adj_matrix: (n,n) database showing which bot follows who

    OUTPUT:
    * (n,2) array: the actual dist and angle between an nth bot and its local leader
    '''
    def current_dist_angle_helper(lead_pos, follow_pos):
        '''
        Helper function: Calculates the distance and angle (to the horizontal)
        between two points.

        INPUT:
        * pos1, pos2: (1,2) array representing vectors (x,y)
        OUTPUT:
        * [r, theta]: (1,2) array - r is the distance, theta is angle in radians
        '''
        delta_x = lead_pos[0]-follow_pos[0]
        delta_y = lead_pos[1]-follow_pos[1]
        r = jnp.hypot(abs(delta_x), abs(delta_y))
        theta = jnp.arctan2(delta_y, delta_x)
        return jnp.array([r, theta])
    
    n = jnp.shape(bot_positions)[0]
    current_dist_angles = jnp.zeros((n,2))

    # Get a list of positions for each bot's leader
    follow_idxs, lead_idxs = jax.numpy.nonzero(adj_matrix, size=n-1, fill_value=None)
    bot_followers = bot_positions[follow_idxs]
    bot_leaders = bot_positions[lead_idxs]
    current_dist_angles = jax.vmap(current_dist_angle_helper)(bot_leaders, bot_followers)
    return current_dist_angles


def velocity_vectors(current, desired, adj_matrix):
    '''
    Given the current distance-angles and desired distance-angles of each bot,
    return the velocity vectors that each bot must take to get to its desired position.

    INPUT:
    * current_dist_angle: (n,2) array - for each nth bot, shows its current distance and angle
    (to the horizontal) to its local leader
    * desired_dist_angle: (n,2) array - for each nth bot, shows its desired distance and angle
    (to the horizontal) to its local leader
    OUTPUT:
    * (n,2) array: each nth row is the [x,y] velocity vector for the nth bot
    '''
    def velocity_vectors_helper(current_r_theta, desired_r_theta):
        r, theta = current_r_theta
        r_d, theta_d = desired_r_theta
        v_c = jnp.array([r*jnp.cos(theta), r*jnp.sin(theta)])
        v_d = jnp.array([r_d*jnp.cos(theta_d), r_d*jnp.sin(theta_d)])
        v = jnp.subtract(v_c, v_d)
        return v

    vectors = jax.vmap(velocity_vectors_helper)(current, desired) # source of error i think
    return vectors


if __name__ == '__main__':

    # DEBUG TEST 2
    n = 3
    shape = 'single-chain'
    adj_matrix = make_adj_matrix(n, shape)
    init_states = jnp.array([[1,0],
                             [5,0],
                             [9,0]])
    current = current_dist_angle(init_states, adj_matrix)
    print(current)
    print('---')
    desired = jnp.array([[4, jnp.pi/2],
                         [4, jnp.pi/2]])
    print(desired)
    print('---')
    vectors = velocity_vectors(current, desired, adj_matrix)
    print(vectors)
    print('---')
