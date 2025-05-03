# Levantine Translation

A framework for training language models to translate between dialects of Arabic with a focus on Levantine Arabic. This repository implements approaches for both direct translation and phoneticization-based translation inspired by the Romansetu paper.

## Features

- 🔤 **Transliteration**: Convert Arabic script to an approximate Latin alphabet using the HSB (Habash-Soudi-Buckwalter) system via the CamelTools library
- 🔄 **Dialect Translation**: Train models to translate between Modern Standard Arabic (MSA), Levantine Arabic dialects, and English
- 📊 **Comprehensive Evaluation**: Measure model performance using perplexity, BLEU scores, and evaluation loss
- 🤗 **Hugging Face Integration**: Easily upload processed datasets and models to Hugging Face Hub
- 🧩 **Modular Architecture**: Separate scripts for dataset creation, model training, and evaluation which can be customized using parameters

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

The `create_dataset..sh` script downloads and processes the dataset for training:

```bash
./src/create_dataset.py 
```

Under the hood, this will:
- Download the UFAL Parallel Corpus of North Levantine 1.0
- Perform the preprocessing needed to transform the downloaded files into a Huggingface dataset
- Process it into four dataset variants:
  - `finetune`: Text entries from english, classical arabic, and levantine arabic
  - `translate`: Translation data, with prompts indicating source and target languages
  - `finetune_roman`: Transliterated version of the finetune dataset
  - `translate_roman`: Transliterated version of the translate dataset
- Optionally upload the processed datasets to Hugging Face Hub

The script can be modified to push the datasets to Huggingface and choose the repo name, as well as changing the location of the dataset files on disk, or where the dataset will be saved.

### 2. Model Training

#### Fine-tuning

To fine-tune a pre-trained language model:

```bash
./src/finetune.sh
```

This script fine-tunes a model on the prepared dataset. This is performed before training the model on translation, to prepare the model for the distribution shift induced by the new language, since it is originally trained on english.

This script is very modular, and allows to changing the trained model, the dataset used, learning rate, epochs, gradient accumulation steps, model save location, as well as some other parameters. The script uses huggingface accelerate to simplify training on different kidns of hardware environments.

#### Sequence-to-Sequence Translation

To train a model for dialect translation:

```bash
./src/seq_2_seq.sh
```

This script trains a sequence-to-sequence model to translate between Arabic dialects. This will train the model on translatino betwwen every pair of languages in {english, classical arabic, and levantine arabic}. The reason behind this choice is that since the model is originally trained on english, this will serve as a "Rosetta stone" between the languages and allow transfer learning into the other languages. This is similar to the procedure used in the RomanSetu paper where their continual pretraining involved using both english corpora as well as coropra from their target language.

### 3. Evaluation & Results

To evaluate a trained model:

```bash
./src/evaluate.sh 
```

The evaluation script calculates:
- BLEU scores
- Perplexity
- Evaluation loss

Results then logged to Weights & Biases for portability and visualization. In the project workspace directory, multiple panels can be found which showcase the resuts. These panels include example translations for the models, Perplexity, BLEU scores, as well as Evaluation and Training losses recorded during training for multiple training runs used in hyperparameter tuning of the model
Results are availaible under the following project:
https://wandb.ai/khuss/CS512
The main results are also collected into a report, available here:
https://wandb.ai/khuss/CS512/reports/Dialect-Translation--VmlldzoxMjU4NjI3OQ



## Transliteration Approach

This repository implements phoneticization/latinization of Arabic text by transliterating it into Latin script using the HSB (Habash-Soudi-Buckwalter) transliteration system. This is a fully reversible transliteration system, which is build upon the original Buckwalter system by making it easy to read. 

Since HSB uses (mostly) latin characters, we use it as an approximate latinization. This approach builds upon the work in the Romansetu paper, which demonstrated improved performance for languages that don't use the Latin script. In the original paper, the authors use latinzation to improve performance on a target language, as well as imporve translation ability between that languge and english. We build upon this by using latinization to help with multi-way translation: between english, arabic, and levantine arabic, a dialect of arabic with some latin influences.

To do this, multiple tools were explored, and this is illustrated in the transliteration.ipynb notebook. 

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

### Romanized (HSB) Translation Format
```python
{
  "prompt": ["english: Hello, how are you?\nclassical arabic:"],
  "completion": ["mrHbA kyf HAlk؟"]
}
```

## Requirements

- Python 3.12+
- uv package manager
- CamelTools
- PyTorch
- Transformers
- Datasets
- SacreBLEU
- wandb (for experiment tracking)

## Citation

If you use this code, please cite:

```
@software{Levantine-Translation},
  author = {Karim El Husseini},
  title = {Levantine Translation},
  year = {2025},
  url = {https://github.com/CppKarim/Levantine-Translation.git}
}
```

This work is inspired by the RomanSetu paper:

```
@misc{husain2024romansetuefficientlyunlockingmultilingual,
      title={RomanSetu: Efficiently unlocking multilingual capabilities of Large Language Models via Romanization}, 
      author={Jaavid Aktar Husain and Raj Dabre and Aswanth Kumar and Jay Gala and Thanmay Jayakumar and Ratish Puduppully and Anoop Kunchukuttan},
      year={2024},
      eprint={2401.14280},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2401.14280}, 
}
```
