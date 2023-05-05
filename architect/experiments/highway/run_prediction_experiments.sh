#!/bin/bash

conda activate architect_env

# Run static "prediction" (just to give a sense of the prior behavior)
EXPNAME="static" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 1 --num_steps_per_round 1 --num_chains 10 --disable_gradients --disable_stochasticity --savename highway_lqr &

# Run MALA prediction (with tempering)
EXPNAME="mala_t" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --savename highway_lqr &

# Run MALA prediction (no tempering)
EXPNAME="mala" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --no-temper --savename highway_lqr &

# Run RMH prediction (with tempering)
EXPNAME="rmh_t" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_gradients --savename highway_lqr &

# Run RMH prediction (no tempering)
EXPNAME="rmh" CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_gradients --no-temper --savename highway_lqr &

# Run ULA prediction (no tempering)
EXPNAME="ula" CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_mh --no-temper --savename highway_lqr &

# Run REINFORCE/L2C prediction (without tempering)
EXPNAME="l2c" CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --savename highway_lqr &

# Run REINFORCE/L2C prediction (with tempering)
EXPNAME="l2c_t" CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --savename highway_lqr &

# Run GD prediction (without tempering)
EXPNAME="gd" CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --no-temper --savename highway_lqr &

# Run GD prediction (with tempering)
EXPNAME="gd_t" CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 100 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --savename highway_lqr &
