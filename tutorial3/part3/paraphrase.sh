#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1

date '+[%H:%M:%S-%d/%m/%y]'

PROJECT_HOME=${HOME}/nlp1 #Change to the directory you have saved the project
FOLD=0  
PARAPHRASE_MODEL="DipperParaphraser"
INPUT_PATH="part3/data/cross_validation_sets.pkl"
OUTPUT_PATH="part3/data/paraphrases"
NUM_SAMPLES_PER_CLASS=100

echo "Activating environment"
#Add the path to the virtual environment you have created
source /home/eilia/nlp1-student/.venv/bin/activate

export HF_HOME=/projects/0/gusr0466/eilia/hf_cache
export NLTK_DATA=/projects/0/gusr0466/eilia/nltk_data
huggingface-cli download kalpeshk2011/dipper-paraphraser-xxl --max-workers 1

echo "Starting paraphrasing"
python3 part3/utils/paraphrase.py \
  --paraphrase_model "${PARAPHRASE_MODEL}" \
  --fold "${FOLD}" \
  --input_path "${INPUT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --num_samples_per_class "${NUM_SAMPLES_PER_CLASS}"

date '+[%H:%M:%S-%d/%m/%y]'
