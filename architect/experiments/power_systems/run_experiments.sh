## Run different seeds

CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --seed 0 --disable_gradients --quench_rounds 0 &
CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --seed 0 &

CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --seed 1 --disable_gradients --quench_rounds 0 &
CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --seed 1 &

CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --seed 2 --disable_gradients --quench_rounds 0 &
CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --seed 2 &

CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --seed 3 --disable_gradients --quench_rounds 0 &
CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --seed 3 &

wait

CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --case_name case57 --seed 0 --disable_gradients --quench_rounds 0 &
CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --case_name case57 --seed 0 &

CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --case_name case57 --seed 1 --disable_gradients --quench_rounds 0 &
CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --case_name case57 --seed 1 &

CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --case_name case57 --seed 2 --disable_gradients --quench_rounds 0 &
CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --case_name case57 --seed 2 &

CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --case_name case57 --seed 3 --disable_gradients --quench_rounds 0 &
CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --case_name case57 --seed 3 &
