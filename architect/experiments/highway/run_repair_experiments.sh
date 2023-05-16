#!/bin/bash

# # Run static "prediction" (just to give a sense of the prior behavior)
# EXPNAME="static" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway_lqr/initial_policy.eqx --num_rounds 1 --num_steps_per_round 1 --num_chains 10 --disable_gradients --disable_stochasticity --savename highway_lqr &

wait;

EXPNAME="mala_t" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway_lqr/initial_policy.eqx --seed 0 --num_rounds 10 --quench_rounds 5 --num_steps_per_round 5 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --repair --savename highway_lqr &
EXPNAME="rmh" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway_lqr/initial_policy.eqx --seed 0 --num_rounds 10 --num_steps_per_round 5 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_gradients --no-temper --repair --savename highway_lqr &
EXPNAME="l2c" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway_lqr/initial_policy.eqx --seed 0 --num_rounds 10 --num_steps_per_round 5 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --repair --savename highway_lqr &
EXPNAME="gd" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway_lqr/initial_policy.eqx --seed 0 --num_rounds 10 --num_steps_per_round 5 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --no-temper --repair --savename highway_lqr &

wait;

EXPNAME="mala_t" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway_lqr/initial_policy.eqx --seed 1 --num_rounds 10 --quench_rounds 5 --num_steps_per_round 5 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --repair --savename highway_lqr &
EXPNAME="rmh" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway_lqr/initial_policy.eqx --seed 1 --num_rounds 10 --num_steps_per_round 5 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_gradients --no-temper --repair --savename highway_lqr &
EXPNAME="l2c" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway_lqr/initial_policy.eqx --seed 1 --num_rounds 10 --num_steps_per_round 5 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --repair --savename highway_lqr &
EXPNAME="gd" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway_lqr/initial_policy.eqx --seed 1 --num_rounds 10 --num_steps_per_round 5 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --no-temper --repair --savename highway_lqr &

wait;

EXPNAME="mala_t" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway_lqr/initial_policy.eqx --seed 2 --num_rounds 10 --quench_rounds 5 --num_steps_per_round 5 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --repair --savename highway_lqr &
EXPNAME="rmh" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway_lqr/initial_policy.eqx --seed 2 --num_rounds 10 --num_steps_per_round 5 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_gradients --no-temper --repair --savename highway_lqr &
EXPNAME="l2c" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway_lqr/initial_policy.eqx --seed 2 --num_rounds 10 --num_steps_per_round 5 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --repair --savename highway_lqr &
EXPNAME="gd" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway_lqr/initial_policy.eqx --seed 2 --num_rounds 10 --num_steps_per_round 5 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --no-temper --repair --savename highway_lqr &

wait;

EXPNAME="mala_t" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway_lqr/initial_policy.eqx --seed 3 --num_rounds 10 --quench_rounds 5 --num_steps_per_round 5 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --repair --savename highway_lqr &
EXPNAME="rmh" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway_lqr/initial_policy.eqx --seed 3 --num_rounds 10 --num_steps_per_round 5 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_gradients --no-temper --repair --savename highway_lqr &
EXPNAME="l2c" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway_lqr/initial_policy.eqx --seed 3 --num_rounds 10 --num_steps_per_round 5 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --repair --savename highway_lqr &
EXPNAME="gd" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway_lqr/initial_policy.eqx --seed 3 --num_rounds 10 --num_steps_per_round 5 --num_chains 10 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --no-temper --repair --savename highway_lqr &
