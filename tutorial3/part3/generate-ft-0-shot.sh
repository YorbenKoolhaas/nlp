#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1

date '+[%H:%M:%S-%d/%m/%y]'

#Change the directories to correspond to the ones you have saved the project and data
PROJECT_HOME=${HOME}/nlp1 
INSTRUCTION_TYPE="0-shot"
MODEL_NAME="qwen-0.5b-instruct-ft-lora"
INPUT_ID_PATH="part3/data/cross_validation_sets.pkl"
OUTPUT_ID_PATH="part3/predictions"
ADAPTER_PATH="/projects/0/gusr0466/eilia/lora-models"

echo "Activating environment"
#Add the path to the virtual environment you have created
source /home/eilia/nlp1-student/.venv/bin/activate

echo "Executing python script"

# In order to generate the in-distribution predictions
python3 part3/utils/generate.py \
  --instruction_type "${INSTRUCTION_TYPE}" \
  --model_name "${MODEL_NAME}" \
  --input_path "${INPUT_ID_PATH}" \
  --output_path "${OUTPUT_ID_PATH}" \
  --adapter_path "${ADAPTER_PATH}"

date '+[%H:%M:%S-%d/%m/%y]'
