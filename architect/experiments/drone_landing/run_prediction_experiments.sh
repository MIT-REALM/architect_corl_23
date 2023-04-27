#!/bin/bash

# Set CUDA/XLA environment

# Run MALA prediction
python architect/experiments/drone_landing/predict_and_mitigate.py \
    --model_path tmp/oracle_32/policy_50.eqx \
    --num_rounds 100 --num_steps_per_round 10 --num_chains 10 --T 80 \
    --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3  \
    --num_trees 10 \
    --savename drone_landing

# Run GD prediction
python architect/experiments/drone_landing/predict_and_mitigate.py \
    --model_path tmp/oracle_32/policy_50.eqx \
    --num_rounds 100 --num_steps_per_round 10 --num_chains 10 --T 80 \
    --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3  \
    --num_trees 10 \
    --disable_stochasticity \
    --savename drone_landing

# Run RMH prediction
python architect/experiments/drone_landing/predict_and_mitigate.py \
    --model_path tmp/oracle_32/policy_50.eqx \
    --num_rounds 100 --num_steps_per_round 10 --num_chains 10 --T 80 \
    --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3  \
    --num_trees 10 \
    --disable_gradients \
    --savename drone_landing


# Run ULA prediction
CUDA_VISIBLE_DEVICES=3, python architect/experiments/drone_landing/predict_and_mitigate.py \
    --model_path tmp/oracle_32/policy_50.eqx \
    --num_rounds 100 --num_steps_per_round 10 --num_chains 10 --T 80 \
    --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3  \
    --num_trees 10 \
    --disable_mh \
    --savename drone_landing