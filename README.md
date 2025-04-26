# Arabic Dialect Translation

A framework for training language models to translate between dialects of Arabic with a focus on Levantine Arabic. This repository implements approaches for both direct translation and phoneticization-based translation inspired by the Romansetu paper.

## Features

- 🔤 **Phoneticization and Transliteration**: Convert Arabic script to Latin alphabet using the HSB (Habash-Soudi-Buckwalter) system via the CamelTools library
- 🔄 **Dialect Translation**: Train models to translate between Modern Standard Arabic (MSA) and Levantine Arabic dialects
- 📊 **Comprehensive Evaluation**: Measure model performance using perplexity, BLEU scores, and evaluation loss
- 🤗 **Hugging Face Integration**: Easily upload processed datasets and models to Hugging Face Hub
- 🧩 **Modular Architecture**: Separate scripts for dataset creation, model training, and evaluation

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone https://github.com/CppKarim/Levantine-Translation.git
cd Levantine-Translation

# Install dependencies using uv
uv sync
source .venv/bin/activate  
```

## Workflow

### 1. Dataset Preparation

The `create_dataset.py` script downloads and processes the dataset for training:

```bash
python create_dataset.py --data_dir data/ --output_dir processed_data/ --push_to_hub
```

This will:
- Download the North Levantine parallel corpus
- Process it into four dataset variants:
  - `finetune`: Direct dialect translation with empty prompts
  - `translate`: Translation with prompts indicating source and target languages
  - `finetune_roman`: Transliterated version of the finetune dataset
  - `translate_roman`: Transliterated version of the translate dataset
- Optionally upload the processed datasets to Hugging Face Hub

### 2. Model Training

#### Sequence-to-Sequence Translation

To train a model for dialect translation:

```bash
./seq_2_seq.sh
```

This script trains a sequence-to-sequence model to translate between Arabic dialects.

#### Fine-tuning

To fine-tune a pre-trained language model:

```bash
./finetune.sh
```

This script fine-tunes a model on the prepared dataset.

### 3. Evaluation & Results

To evaluate a trained model:

```bash
./evaluate.sh --model path/to/model --dataset path/to/test_data
```

The evaluation script calculates:
- BLEU scores
- Perplexity
- Evaluation loss

Results then logged to Weights & Biases for portability and visualization.
Results are availaible under the following project:
https://wandb.ai/khuss/CS512

## Phoneticization Approach

This repository implements phoneticization/latinization of Arabic text by transliterating it into Latin script using the HSB system. This approach is inspired by the Romansetu paper, which demonstrated improved performance for languages that don't use the Latin script. To do this, multiple tools were explored, and this is illustrated in the transliteration.ipynb notebook. 

Example of Arabic text transliteration:
```
Original: ذهبت إلى المكتبة
Transliterated: *habt <ilY Almktbp
```

The utils forlder contains a transliteration library which we adapted into python from javascript for these testing purposes.
## Dataset Structure

The processed datasets have the following structure:

### Standard Format
```python
{
  "prompt": ["", "", ""],
  "completion": ["English text", "MSA Arabic text", "Levantine Arabic text"]
}
```

### Translation Format
```python
{
  "prompt": ["english: Hello, how are you?\nclassical arabic:"],
  "completion": ["مرحبا كيف حالك؟"]
}
```

### Romanized Translation Format
```python
{
  "prompt": ["english: Hello, how are you?\nclassical arabic:"],
  "completion": ["mrHbA kyf HAlk?"]
}
```

## Requirements

- Python 3.8+
- uv package manager
- CamelTools
- PyTorch
- Transformers
- Datasets
- SacreBLEU
- wandb (for experiment tracking)
