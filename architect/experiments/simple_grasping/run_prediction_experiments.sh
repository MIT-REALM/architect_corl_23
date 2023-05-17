# Mug

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 0 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --object_type mug --savename grasping/mug &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 0 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --disable_gradients --no-temper --object_type mug --savename grasping/mug &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 0 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --reinforce --no-temper --no-repair --predict --object_type mug --savename grasping/mug &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 0 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --disable_stochasticity --no-repair --predict --no-temper --object_type mug --savename grasping/mug &

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 1 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --object_type mug --savename grasping/mug &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 1 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --disable_gradients --no-temper --object_type mug --savename grasping/mug &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 1 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --reinforce --no-temper --no-repair --predict --object_type mug --savename grasping/mug &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 1 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --disable_stochasticity --no-repair --predict --no-temper --object_type mug --savename grasping/mug &

wait;

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 2 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --object_type mug --savename grasping/mug &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 2 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --disable_gradients --no-temper --object_type mug --savename grasping/mug &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 2 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --reinforce --no-temper --no-repair --predict --object_type mug --savename grasping/mug &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 2 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --disable_stochasticity --no-repair --predict --no-temper --object_type mug --savename grasping/mug &

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 3 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --object_type mug --savename grasping/mug &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 3 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --disable_gradients --no-temper --object_type mug --savename grasping/mug &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 3 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --reinforce --no-temper --no-repair --predict --object_type mug --savename grasping/mug &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 3 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --disable_stochasticity --no-repair --predict --no-temper --object_type mug --savename grasping/mug &

# box
wait;

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 0 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --object_type box --savename grasping/box &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 0 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --disable_gradients --no-temper --object_type box --savename grasping/box &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 0 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --reinforce --no-temper --no-repair --predict --object_type box --savename grasping/box &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 0 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --disable_stochasticity --no-repair --predict --no-temper --object_type box --savename grasping/box &

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 1 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --object_type box --savename grasping/box &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 1 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --disable_gradients --no-temper --object_type box --savename grasping/box &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 1 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --reinforce --no-temper --no-repair --predict --object_type box --savename grasping/box &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 1 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --disable_stochasticity --no-repair --predict --no-temper --object_type box --savename grasping/box &

wait;

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 2 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --object_type box --savename grasping/box &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 2 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --disable_gradients --no-temper --object_type box --savename grasping/box &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 2 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --reinforce --no-temper --no-repair --predict --object_type box --savename grasping/box &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 2 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --disable_stochasticity --no-repair --predict --no-temper --object_type box --savename grasping/box &

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 3 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --object_type box --savename grasping/box &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 3 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --disable_gradients --no-temper --object_type box --savename grasping/box &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 3 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --reinforce --no-temper --no-repair --predict --object_type box --savename grasping/box &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 3 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --disable_stochasticity --no-repair --predict --no-temper --object_type box --savename grasping/box &

# bowl
wait;

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 0 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --object_type bowl --savename grasping/bowl &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 0 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --disable_gradients --no-temper --object_type bowl --savename grasping/bowl &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 0 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --reinforce --no-temper --no-repair --predict --object_type bowl --savename grasping/bowl &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 0 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --disable_stochasticity --no-repair --predict --no-temper --object_type bowl --savename grasping/bowl &

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 1 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --object_type bowl --savename grasping/bowl &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 1 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --disable_gradients --no-temper --object_type bowl --savename grasping/bowl &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 1 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --reinforce --no-temper --no-repair --predict --object_type bowl --savename grasping/bowl &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 1 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --disable_stochasticity --no-repair --predict --no-temper --object_type bowl --savename grasping/bowl &

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 2 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --object_type bowl --savename grasping/bowl &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 2 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --disable_gradients --no-temper --object_type bowl --savename grasping/bowl &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 2 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --reinforce --no-temper --no-repair --predict --object_type bowl --savename grasping/bowl &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 2 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --disable_stochasticity --no-repair --predict --no-temper --object_type bowl --savename grasping/bowl &

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 3 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --object_type bowl --savename grasping/bowl &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 3 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --L 10 --no-repair --predict --disable_gradients --no-temper --object_type bowl --savename grasping/bowl &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 3 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --reinforce --no-temper --no-repair --predict --object_type bowl --savename grasping/bowl &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping/initial_policy.eqx --seed 3 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --dont_normalize_gradients --disable_stochasticity --no-repair --predict --no-temper --object_type bowl --savename grasping/bowl &

