#!/bin/bash
# Change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "Failed to change directory to $SCRIPT_DIR"; exit 1; }
cd ../

uv run ./src/evaluate/perplexity.py\
 --model "data/models/openai-community/gpt2/translate/roman" \
 --dataset "data\dataset\Huggingface\translate_roman" \
 --batch_size 16 --max_length 512\
 --wandb_project "CS512" --wandb_run_name "perplexity_roman"

uv run ./src/evaluate/bleu.py\
 --model "data/models/openai-community/gpt2/translate/roman" \
 --dataset "data\dataset\Huggingface\translate_roman" \
 --batch_size 16 --max_length 512\
 --wandb_project "CS512" --wandb_run_name "bleu_roman" 


 
