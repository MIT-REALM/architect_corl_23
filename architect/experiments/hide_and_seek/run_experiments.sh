CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --disable_stochasticity --no-predict &
CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --disable_stochasticity &
CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --disable_gradients --quench_rounds 0 &
CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py &

wait

CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/stress_test.py --filename results/hide_and_seek_disturbance/6_seekers_10_hiders/L_1.0e+01_5_T_1000_samples_25_quench_10_chains_step_dp_1.0e-02_ep_1.0e-02_gd_repair_predict.json &
CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/stress_test.py --filename results/hide_and_seek_disturbance/6_seekers_10_hiders/L_1.0e+01_5_T_1000_samples_0_quench_10_chains_step_dp_1.0e-02_ep_1.0e-02_rmh_repair_predict.json &
CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/stress_test.py --filename results/hide_and_seek_disturbance/6_seekers_10_hiders/L_1.0e+01_5_T_1000_samples_25_quench_10_chains_step_dp_1.0e-02_ep_1.0e-02_gd_repair.json &
CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/stress_test.py --filename results/hide_and_seek_disturbance/6_seekers_10_hiders/L_1.0e+01_5_T_1000_samples_25_quench_10_chains_step_dp_1.0e-02_ep_1.0e-02_mala_repair_predict.json &

wait

CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/training_curves.py --filename results/hide_and_seek_disturbance/6_seekers_10_hiders/L_1.0e+01_5_T_1000_samples_25_quench_10_chains_step_dp_1.0e-02_ep_1.0e-02_gd_repair_dp_trace.json &
CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/training_curves.py --filename results/hide_and_seek_disturbance/6_seekers_10_hiders/L_1.0e+01_5_T_1000_samples_25_quench_10_chains_step_dp_1.0e-02_ep_1.0e-02_gd_repair_predict_dp_trace.json &
CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/training_curves.py --filename results/hide_and_seek_disturbance/6_seekers_10_hiders/L_1.0e+01_5_T_1000_samples_25_quench_10_chains_step_dp_1.0e-02_ep_1.0e-02_mala_repair_predict_dp_trace.json &
CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/training_curves.py --filename results/hide_and_seek_disturbance/6_seekers_10_hiders/L_1.0e+01_5_T_1000_samples_0_quench_10_chains_step_dp_1.0e-02_ep_1.0e-02_rmh_repair_predict_dp_trace.json &

wait

python architect/experiments/hide_and_seek/plot_training_curves.py

python architect/experiments/hide_and_seek/plot_stress_test_results.py 


## Run different seeds

CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --seed 0 --disable_gradients --quench_rounds 0 &
CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --seed 0 &

CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --seed 1 --disable_gradients --quench_rounds 0 &
CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --seed 1 &

CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --seed 2 --disable_gradients --quench_rounds 0 &
CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --seed 2 &

CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --seed 3 --disable_gradients --quench_rounds 0 &
CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --seed 3 &