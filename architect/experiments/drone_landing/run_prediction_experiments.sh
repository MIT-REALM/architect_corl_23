#!/bin/bash

conda activate architect_env

# Run static "prediction" (just to give a sense of the prior behavior)
CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path tmp/oracle_32/policy_50.eqx --num_rounds 1 --num_steps_per_round 1 --num_chains 10 --T 70 --num_trees 5 --disable_gradients --disable_stochasticity --savename drone_landing_smooth &

# Run MALA prediction (with tempering)
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path tmp/oracle_32/policy_50.eqx --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-1 --dp_mcmc_step_size 1e-1 --num_trees 10 --savename drone_landing_smooth_10 &

# Run MALA prediction (no tempering)
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path tmp/oracle_32/policy_50.eqx --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-1 --dp_mcmc_step_size 1e-1 --num_trees 10 --no-temper --savename drone_landing_smooth_10 &

# Run RMH prediction (with tempering)
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path tmp/oracle_32/policy_50.eqx --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 10 --disable_gradients --savename drone_landing_smooth_10 &

# Run RMH prediction (no tempering)
CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path tmp/oracle_32/policy_50.eqx --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 10 --disable_gradients --no-temper --savename drone_landing_smooth_10 &

# Run ULA prediction (no tempering)
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path tmp/oracle_32/policy_50.eqx --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --disable_mh --no-temper --savename drone_landing_smooth &

# Run REINFORCE/L2C prediction (without tempering) TODO
CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path tmp/oracle_32/policy_50.eqx --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --num_trees 5 --reinforce --no-temper --savename drone_landing_smooth &

# Run REINFORCE/L2C prediction (with tempering)
CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path tmp/oracle_32/policy_50.eqx --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --num_trees 5 --reinforce --savename drone_landing_smooth &


# # Run GD prediction (without tempering) TODO once tuned
# CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path tmp/oracle_32/policy_50.eqx --num_rounds 50 --num_steps_per_round 10 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --num_trees 5 --disable_stochasticity --no-temper --savename drone_landing_smooth &

# # Run GD prediction (with tempering) TODO once tuned
# CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path tmp/oracle_32/policy_50.eqx --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --num_trees 5 --disable_stochasticity --savename drone_landing_smooth &
