import argparse
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from tqdm import tqdm

from architect.systems.f16.simulator import (
    ResidualControl,
    initial_state_logprior,
    sample_initial_states,
    simulate,
)

if __name__ == "__main__":
    print("Training curves should be captured on WandB.")
