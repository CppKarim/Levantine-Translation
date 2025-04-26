#!/bin/bash
# Change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "Failed to change directory to $SCRIPT_DIR"; exit 1; }
cd ..
export PYTHONPATH="./"

# Script
echo "This script trains an LLM using supervised fine tuining and a dataset of text."
accelerate config
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=1 
accelerate launch src/training/seq_2_seq.py\
    --model_name data/models/gpt2/sft/roman --dataset "data\dataset\Huggingface\translate_roman" \
    --dataset_offset 0 --dataset_percent 100 --save_interval 0.02 --dataset_num_proc 128 \
    --learning_rate 7e-5 --seq_len 512 --per_device_batch 32 --gradient_accumulation_steps 1 --epochs 3 \
    --subdirectory translate/roman --save_model \
