#!/bin/bash
# Change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "Failed to change directory to $SCRIPT_DIR"; exit 1; }
cd ../

./src/data_processing/download_data.sh

uv run ./src/data_processing/create_hf_dataset.py \
 --data_dir "data/dataset/UFAL Parallel Corpus of North Levantine 1.0"\
 --output_dir "data/dataset/Huggingface/UFAL Parallel Corpus of North Levantine 1.0 test"\
 --push_to_hub --repo_name UFAL


uv run .\src\data_processing\prepare_training_datasets.py\
 --dataset "data\dataset\Huggingface\UFAL Parallel Corpus of North Levantine 1.0"\
 --output_dir "data\dataset\Huggingface"\
 --push_to_hub --repo_name UFAL