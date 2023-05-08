#!/bin/bash

conda activate architect_env

# # Run static "prediction" (just to give a sense of the prior behavior)
# EXPNAME="static" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 1 --num_steps_per_round 1 --num_chains 10 --disable_gradients --disable_stochasticity --savename highway_lqr &

# Run MALA prediction (with tempering)
EXPNAME="mala_t" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 200 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --savename highway_lqr &

# Run RMH prediction (no tempering)
EXPNAME="rmh" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 200 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_gradients --no-temper --savename highway_lqr &

# Run REINFORCE/L2C prediction (without tempering)
EXPNAME="l2c" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 200 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --savename highway_lqr &

# Run GD prediction (without tempering)
EXPNAME="gd" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 200 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --no-temper --savename highway_lqr &
