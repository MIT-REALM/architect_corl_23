#!/bin/bash

conda activate architect_env

# DONE

# Run static "prediction" (just to give a sense of the prior behavior)
CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path tmp/oracle_32/policy_50.eqx --num_rounds 1 --num_steps_per_round 1 --num_chains 10 --T 60 --num_trees 10 --disable_gradients --disable_stochasticity --savename drone_landing &

# Run MALA prediction (with tempering)
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path tmp/oracle_32/policy_50.eqx --num_rounds 100 --num_steps_per_round 10 --num_chains 10 --T 60 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --num_trees 10 --savename drone_landing &

# Run GD prediction (with tempering)
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path tmp/oracle_32/policy_50.eqx --num_rounds 100 --num_steps_per_round 10 --num_chains 10 --T 60 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --num_trees 10 --disable_stochasticity --savename drone_landing &

# Run RMH prediction
CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path tmp/oracle_32/policy_50.eqx --num_rounds 100 --num_steps_per_round 10 --num_chains 10 --T 60 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --num_trees 10 --disable_gradients --savename drone_landing &

# Run ULA prediction
CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path tmp/oracle_32/policy_50.eqx --num_rounds 100 --num_steps_per_round 10 --num_chains 10 --T 60 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --num_trees 10 --disable_mh --savename drone_landing &

# Run REINFORCE/L2C prediction (with tempering)
CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path tmp/oracle_32/policy_50.eqx --num_rounds 100 --num_steps_per_round 10 --num_chains 10 --T 60 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --num_trees 10 --reinforce --savename drone_landing &

# Run REINFORCE/L2C prediction (without tempering)
CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path tmp/oracle_32/policy_50.eqx --num_rounds 100 --num_steps_per_round 10 --num_chains 10 --T 60 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --num_trees 10 --reinforce --no-temper --savename drone_landing &

# Run GD prediction (without tempering)
CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path tmp/oracle_32/policy_50.eqx --num_rounds 100 --num_steps_per_round 10 --num_chains 10 --T 60 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --num_trees 10 --disable_stochasticity --no-temper --savename drone_landing &
