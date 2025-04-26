#!/bin/bash
# Change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "Failed to change directory to $SCRIPT_DIR"; exit 1; }
cd ..
source ./venv/bin/activate
export PYTHONPATH="./"

# Script
echo "This script trains an LLM using supervised fine tuining and a dataset of text."
accelerate config
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=1 
accelerate launch src/training/supervised_finetuning.py\
    --model_name openai-community/gpt2 --dataset "data\dataset\Huggingface\finetune_roman" \
    --dataset_offset 0 --dataset_percent 100 --save_interval 0.02 --dataset_num_proc 128 \
    --learning_rate 7e-5 --seq_len 1024 --per_device_batch 64 --gradient_accumulation_steps 1 --epochs 3 \
    --subdirectory sft/roman --save_model \
