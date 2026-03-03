#!/bin/bash
source /home/shriramgeethanath/SNLP/HyperscaleES/.venv/bin/activate
cd /home/shriramgeethanath/SNLP/HyperscaleES

LOG_FILE="runs/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p runs

python -m llm_experiments.general_do_evolution \
  --model_choice l3.2-1B \
  --task qasper \
  --num_epochs 3 \
  --max_seq_len 4096 \
  --thinking_length 3968 \
  --answer_length 128 \
  --parallel_generations_per_gpu 32 \
  --parallel_validations 32 \
  --validation_iterations 2 \
  --validate_every 1 \
  --log_output_every 1 \
  --sigma 1e-3 \
  --lr_scale 1.0 \
  --track \
  --wandb_project HyperscaleExp \
  --wandb_name llama1b_qasper_timing_test \
  2>&1 | tee "$LOG_FILE"
