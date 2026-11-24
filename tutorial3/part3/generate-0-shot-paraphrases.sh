#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1

date '+[%H:%M:%S-%d/%m/%y]'

#Change the directories to correspond to the ones you have saved the project and data
PROJECT_HOME=${HOME}/nlp1 
INSTRUCTION_TYPE="0-shot"
MODEL_NAME="qwen-0.5b-instruct"
FOLD=0
INPUT_PARAPHRASED_PATH="part3/data/paraphrases/DipperParaphraser/0/paraphrases_for_generation.pkl"
OUTPUT_PARAPHRASED_PATH="part3/predictions/paraphrases"

echo "Activating environment"
source ${HOME}/unctune_env/bin/activate

echo "Executing python script"
# Generating the in-distribution paraphrased predictions
python3 part3/utils/generate.py \
  --instruction_type "${INSTRUCTION_TYPE}" \
  --model_name "${MODEL_NAME}" \
  --fold "${FOLD}" \
  --input_path "${INPUT_PARAPHRASED_PATH}" \
  --output_path "${OUTPUT_PARAPHRASED_PATH}"

date '+[%H:%M:%S-%d/%m/%y]'
