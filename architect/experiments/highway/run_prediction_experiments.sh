#!/bin/bash

conda activate architect_env

# Run static "prediction" (just to give a sense of the prior behavior)
CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 1 --num_steps_per_round 1 --num_chains 10 --disable_gradients --disable_stochasticity --savename highway &

# Run MALA prediction (with tempering)
CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --savename highway &

# Run MALA prediction (no tempering)
CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --no-temper --savename highway &

# Run RMH prediction (with tempering)
CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --disable_gradients --savename highway &

# Run RMH prediction (no tempering)
CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --disable_gradients --no-temper --savename highway &

# Run ULA prediction (no tempering)
CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --disable_mh --no-temper --savename highway &

# Run REINFORCE/L2C prediction (without tempering)
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --savename highway &

# Run REINFORCE/L2C prediction (with tempering)
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --savename highway &

# Run GD prediction (without tempering)
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 3e-3 --dp_mcmc_step_size 3e-3 --disable_stochasticity --no-temper --savename highway &

# Run GD prediction (with tempering)
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/overtake_old/overtake_actions/initial_policy.eqx --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --ep_mcmc_step_size 3e-3 --dp_mcmc_step_size 3e-3 --disable_stochasticity --savename highway &
