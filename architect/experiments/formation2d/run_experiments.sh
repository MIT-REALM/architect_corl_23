wait

CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/solve.py --disable_stochasticity --no-predict &
CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/solve.py --disable_stochasticity &
CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/solve.py --disable_gradients --quench_rounds 0 &
CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/solve.py &

wait

CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/stress_test.py --file_prefix results/formation2d_grad_norm/10/L_1.0e+00/3_T/250_samples/5_quench/no_temper5_chains/dp_1.0e-03/ep_1.0e-03/repair/predict/mala &
CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/stress_test.py --file_prefix results/formation2d_grad_norm/10/L_1.0e+00/3_T/250_samples/5_quench/no_temper5_chains/dp_1.0e-03/ep_1.0e-03/repair/predict/gd &
wait
CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/stress_test.py --file_prefix results/formation2d_grad_norm/10/L_1.0e+00/3_T/250_samples/5_quench/no_temper5_chains/dp_1.0e-03/ep_1.0e-03/repair/no_predict/gd &
CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/stress_test.py --file_prefix results/formation2d_grad_norm/10/L_1.0e+00/3_T/250_samples/0_quench/no_temper5_chains/dp_1.0e-03/ep_1.0e-03/repair/predict/rmh &

wait

CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/training_curves.py --file_prefix results/formation2d_grad_norm/10/L_1.0e+00/3_T/250_samples/5_quench/no_temper5_chains/dp_1.0e-03/ep_1.0e-03/repair/predict/mala &
CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/training_curves.py --file_prefix results/formation2d_grad_norm/10/L_1.0e+00/3_T/250_samples/5_quench/no_temper5_chains/dp_1.0e-03/ep_1.0e-03/repair/predict/gd &
wait
CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/training_curves.py --file_prefix results/formation2d_grad_norm/10/L_1.0e+00/3_T/250_samples/5_quench/no_temper5_chains/dp_1.0e-03/ep_1.0e-03/repair/no_predict/gd &
CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/training_curves.py --file_prefix results/formation2d_grad_norm/10/L_1.0e+00/3_T/250_samples/0_quench/no_temper5_chains/dp_1.0e-03/ep_1.0e-03/repair/predict/rmh &

wait

python architect/experiments/formation2d/plot_training_curves.py

python architect/experiments/formation2d/plot_stress_test_results.py 
