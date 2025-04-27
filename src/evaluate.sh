#!/bin/bash
# Change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "Failed to change directory to $SCRIPT_DIR"; exit 1; }
cd ../

accelerate config

accelerate launch ./src/evaluate/perplexity.py\
 --model data/models/openai-community/gpt2/sft/roman/final \
 --dataset "data\dataset\Huggingface\translate_roman" \
 --batch_size 16 --max_length 512\
 --wandb_project "CS512" --wandb_run_name "perplexity_roman"

accelerate launch ./src/evaluate/bleu.py\
 --model data/models/openai-community/gpt2/sft/roman/final \
 --dataset "data\dataset\Huggingface\translate_roman" \
 --batch_size 16 --max_length 512\
 --wandb_project "CS512" --wandb_run_name "bleu_roman" 


 
