CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/solve.py --disable_stochasticity --no-predict &
CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/solve.py --disable_stochasticity &
CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/solve.py --disable_gradients --quench_rounds 0 &
CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/solve.py &

wait

CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/stress_test.py --filename results/formation2d/TODO/L_1.0e+01_5_T_1000_samples_25_quench_10_chains_step_dp_1.0e-02_ep_1.0e-02_gd_repair_predict.json &
CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/stress_test.py --filename results/formation2d/TODO/L_1.0e+01_5_T_1000_samples_0_quench_10_chains_step_dp_1.0e-02_ep_1.0e-02_rmh_repair_predict.json &
CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/stress_test.py --filename results/formation2d/TODO/L_1.0e+01_5_T_1000_samples_25_quench_10_chains_step_dp_1.0e-02_ep_1.0e-02_gd_repair.json &
CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/stress_test.py --filename results/formation2d/TODO/L_1.0e+01_5_T_1000_samples_25_quench_10_chains_step_dp_1.0e-02_ep_1.0e-02_mala_repair_predict.json &

CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/training_curves.py --filename results/formation2d/TODO/L_1.0e+01_5_T_1000_samples_25_quench_10_chains_step_dp_1.0e-02_ep_1.0e-02_gd_repair_dp_trace.json &
CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/training_curves.py --filename results/formation2d/TODO/L_1.0e+01_5_T_1000_samples_25_quench_10_chains_step_dp_1.0e-02_ep_1.0e-02_gd_repair_predict_dp_trace.json &
CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/training_curves.py --filename results/formation2d/TODO/L_1.0e+01_5_T_1000_samples_25_quench_10_chains_step_dp_1.0e-02_ep_1.0e-02_mala_repair_predict_dp_trace.json &
CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/training_curves.py --filename results/formation2d/TODO/L_1.0e+01_5_T_1000_samples_0_quench_10_chains_step_dp_1.0e-02_ep_1.0e-02_rmh_repair_predict_dp_trace.json &

wait

python architect/experiments/formation2d/plot_training_curves.py

python architect/experiments/formation2d/plot_stress_test_results.py 
