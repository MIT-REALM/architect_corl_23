EXP="radium" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 0 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-3 --num_trees 5 --moving_obstacles --repair --savename drone_landing_smooth_dynamic &
EXP="rocus" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 0 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-3 --num_trees 5 --disable_gradients --no-temper --moving_obstacles --repair --savename drone_landing_smooth_dynamic &
EXP="l2c" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 0 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --num_trees 5 --reinforce --no-temper --moving_obstacles --repair --savename drone_landing_smooth_dynamic &
EXP="gd" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 0 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 3e-3 --dp_mcmc_step_size 1e-3 --num_trees 5 --disable_stochasticity --no-temper --moving_obstacles --repair --savename drone_landing_smooth_dynamic &

EXP="radium" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 1 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-3 --num_trees 5 --moving_obstacles --repair --savename drone_landing_smooth_dynamic &
EXP="rocus" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 1 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-3 --num_trees 5 --disable_gradients --no-temper --moving_obstacles --repair --savename drone_landing_smooth_dynamic &
EXP="l2c" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 1 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --num_trees 5 --reinforce --no-temper --moving_obstacles --repair --savename drone_landing_smooth_dynamic &
EXP="gd" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 1 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 3e-3 --dp_mcmc_step_size 1e-3 --num_trees 5 --disable_stochasticity --no-temper --moving_obstacles --repair --savename drone_landing_smooth_dynamic &

EXP="radium" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 2 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-3 --num_trees 5 --moving_obstacles --repair --savename drone_landing_smooth_dynamic &
EXP="rocus" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 2 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-3 --num_trees 5 --disable_gradients --no-temper --moving_obstacles --repair --savename drone_landing_smooth_dynamic &
EXP="l2c" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 2 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --num_trees 5 --reinforce --no-temper --moving_obstacles --repair --savename drone_landing_smooth_dynamic &
EXP="gd" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 2 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 3e-3 --dp_mcmc_step_size 1e-3 --num_trees 5 --disable_stochasticity --no-temper --moving_obstacles --repair --savename drone_landing_smooth_dynamic &

EXP="radium" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 3 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-3 --num_trees 5 --moving_obstacles --repair --savename drone_landing_smooth_dynamic &
EXP="rocus" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 3 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-3 --num_trees 5 --disable_gradients --no-temper --moving_obstacles --repair --savename drone_landing_smooth_dynamic &
EXP="l2c" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 3 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --num_trees 5 --reinforce --no-temper --moving_obstacles --repair --savename drone_landing_smooth_dynamic &
EXP="gd" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 3 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 3e-3 --dp_mcmc_step_size 1e-3 --num_trees 5 --disable_stochasticity --no-temper --moving_obstacles --repair --savename drone_landing_smooth_dynamic &


EXP="radium" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 0 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-3 --num_trees 5 --moving_obstacles --repair --savename drone_landing_smooth_dynamic &
EXP="radium" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 1 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-3 --num_trees 5 --moving_obstacles --repair --savename drone_landing_smooth_dynamic &
EXP="radium" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 2 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-3 --num_trees 5 --moving_obstacles --repair --savename drone_landing_smooth_dynamic &
EXP="radium" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/drone_landing_smooth_dynamic/initial_policy.eqx --seed 3 --num_rounds 30 --num_steps_per_round 1 --num_chains 10 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-3 --num_trees 5 --moving_obstacles --repair --savename drone_landing_smooth_dynamic &