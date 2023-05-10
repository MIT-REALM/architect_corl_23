#!/bin/bash

conda activate architect_env

# # Run static "prediction" (just to give a sense of the prior behavior)
# EXPNAME="static" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 1 --num_steps_per_round 1 --num_chains 10 --disable_gradients --disable_stochasticity --dont_normalize_gradients --savename highway_lqr &

EXPNAME="mala_t" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 0 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --dont_normalize_gradients --savename highway_lqr &
EXPNAME="rmh" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 0 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_gradients --no-temper --dont_normalize_gradients --savename highway_lqr &
EXPNAME="l2c" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 0 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --dont_normalize_gradients --savename highway_lqr &
EXPNAME="gd" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 0 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --no-temper --dont_normalize_gradients --savename highway_lqr &

EXPNAME="mala_t" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 1 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --dont_normalize_gradients --savename highway_lqr &
EXPNAME="rmh" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 1 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_gradients --no-temper --dont_normalize_gradients --savename highway_lqr &
EXPNAME="l2c" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 1 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --dont_normalize_gradients --savename highway_lqr &
EXPNAME="gd" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 1 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --no-temper --dont_normalize_gradients --savename highway_lqr &

EXPNAME="mala_t" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 2 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --dont_normalize_gradients --savename highway_lqr &
EXPNAME="rmh" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 2 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_gradients --no-temper --dont_normalize_gradients --savename highway_lqr &
EXPNAME="l2c" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 2 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --dont_normalize_gradients --savename highway_lqr &
EXPNAME="gd" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 2 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --no-temper --dont_normalize_gradients --savename highway_lqr &

EXPNAME="mala_t" CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 3 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --dont_normalize_gradients --savename highway_lqr &
EXPNAME="rmh" CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 3 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_gradients --no-temper --dont_normalize_gradients --savename highway_lqr &
EXPNAME="l2c" CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 3 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --dont_normalize_gradients --savename highway_lqr &
EXPNAME="gd" CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 3 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --no-temper --dont_normalize_gradients --savename highway_lqr &

# 1e-4

EXPNAME="mala_t" CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 0 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --dont_normalize_gradients --savename highway_lqr &
EXPNAME="rmh" CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 0 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --disable_gradients --no-temper --dont_normalize_gradients --savename highway_lqr &
EXPNAME="l2c" CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 0 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --reinforce --no-temper --dont_normalize_gradients --savename highway_lqr &
EXPNAME="gd" CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 0 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --disable_stochasticity --no-temper --dont_normalize_gradients --savename highway_lqr &

EXPNAME="mala_t" CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 1 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --dont_normalize_gradients --savename highway_lqr &
EXPNAME="rmh" CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 1 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --disable_gradients --no-temper --dont_normalize_gradients --savename highway_lqr &
EXPNAME="l2c" CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 1 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --reinforce --no-temper --dont_normalize_gradients --savename highway_lqr &
EXPNAME="gd" CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 1 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --disable_stochasticity --no-temper --dont_normalize_gradients --savename highway_lqr &

### other instance

EXPNAME="mala_t" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 2 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --dont_normalize_gradients --savename highway_lqr &
EXPNAME="rmh" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 2 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --disable_gradients --no-temper --dont_normalize_gradients --savename highway_lqr &
EXPNAME="l2c" CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 2 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --reinforce --no-temper --dont_normalize_gradients --savename highway_lqr &
EXPNAME="gd" CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 2 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --disable_stochasticity --no-temper --dont_normalize_gradients --savename highway_lqr &

EXPNAME="mala_t" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 3 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --dont_normalize_gradients --savename highway_lqr &
EXPNAME="rmh" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 3 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --disable_gradients --no-temper --dont_normalize_gradients --savename highway_lqr &
EXPNAME="l2c" CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 3 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --reinforce --no-temper --dont_normalize_gradients --savename highway_lqr &
EXPNAME="gd" CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --seed 3 --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --disable_stochasticity --no-temper --dont_normalize_gradients --savename highway_lqr &
