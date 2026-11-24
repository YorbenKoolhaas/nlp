#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1

date '+[%H:%M:%S-%d/%m/%y]'

PROJECT_HOME=${HOME}/nlp1
FOLD=0
MODEL_NAME="qwen-0.5b-instruct"
INPUT_PATH="/home/eilia/nlp1/part3/data/cross_validation_sets.pkl"
OUTPUT_DIR="/projects/0/gusr0466/eilia/lora-models"

echo "Activating environment"
#Add the path to the virtual environment you have created
source /home/eilia/nlp1-student/.venv/bin/activate

echo "Starting fine-tuning"
python3 /home/eilia/nlp1/part3/utils/fine_tuning.py \
  --model_name "${MODEL_NAME}" \
  --input_path "${INPUT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_bf16 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64

date '+[%H:%M:%S-%d/%m/%y]'
