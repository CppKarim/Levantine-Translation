import os
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm

def load_text_file(file_path):
    """Load a text file and return a list of lines."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def load_ids_file(file_path):
    """
    Load an .ids file and return a list of mappings.
    
    Each line in the file is expected to have the format:
    source_file\ttarget_file\tsource_positions\ttarget_positions
    
    Returns:
        List of dictionaries with parsed mapping information
    """
    mappings = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            if len(parts) == 4:
                mapping = {
                    'source_file': parts[0],
                    'target_file': parts[1],
                    'source_positions': parts[2].split(),
                    'target_positions': parts[3].split()
                }
                mappings.append(mapping)
            else:
                print(f"Warning: Skipping malformed line in {file_path}: {line.strip()}")
    return mappings

def create_north_levantine_dataset(data_dir):
    """
    Create a Hugging Face dataset from the North Levantine parallel corpus.
    
    Args:
        data_dir: Directory containing the dataset files
    
    Returns:
        A DatasetDict containing the aligned parallel corpus
    """
    # ISO codes for all languages in the dataset
    languages = ['eng', 'arb', 'apc', 'deu', 'ell', 'fra', 'spa']
    
    # Step 1: Discover and load all language files
    text_data = {}
    
    # Find all language files in the directory
    language_files = {}
    for file in os.listdir(data_dir):
        if file.endswith(tuple(languages)) and not file.endswith(".ids"):
            lang_code = file.split(".")[-1]
            language_files[lang_code] = os.path.join(data_dir, file)
    
    # Load discovered language files
    for lang in tqdm(languages, desc="Loading language files"):
        if lang in language_files:
            text_data[lang] = load_text_file(language_files[lang])
            print(f"Loaded {lang} file with {len(text_data[lang])} lines")
        else:
            print(f"Warning: File for language '{lang}' not found in {data_dir}")
    
    # Step 2: Create individual datasets with index column
    language_datasets = {}
    for lang, lines in text_data.items():
        df = pd.DataFrame({
            'text': lines,
            'index': list(range(len(lines)))
        })
        language_datasets[lang] = Dataset.from_pandas(df)
    
    # Step 3: Create a merged dataset
    # Start with English dataset as base
    if 'eng' not in language_datasets:
        raise ValueError("English dataset is required as the base for alignment")
    
    # Get the number of examples in the English dataset
    num_examples = len(language_datasets['eng'])
    
    # Verify all languages have the same number of examples
    for lang, dataset in language_datasets.items():
        if len(dataset) != num_examples:
            print(f"Warning: {lang} dataset has {len(dataset)} examples, but English has {num_examples}")
    
    # Create a dictionary for the merged dataset
    merged_data = {
        'line_idx': list(range(num_examples)),
        'eng': language_datasets['eng']['text']
    }
    
    # Add other languages
    for lang in languages:
        if lang != 'eng' and lang in language_datasets:
            if len(language_datasets[lang]) == num_examples:
                merged_data[lang] = language_datasets[lang]['text']
            else:
                # Handle mismatched sizes by padding with empty strings
                padded_texts = language_datasets[lang]['text'] + [''] * (num_examples - len(language_datasets[lang]))
                merged_data[lang] = padded_texts[:num_examples]
    
    # Create the merged dataset
    merged_dataset = Dataset.from_dict(merged_data)
    
    # Create a dataset dictionary with train split
    dataset_dict = DatasetDict({
        'train': merged_dataset
    })
    
    return dataset_dict


def get_parallel_sentences(dataset, index, languages=None):
    """
    Get parallel sentences for a specific example across all or specified languages.
    
    Args:
        dataset: The dataset containing parallel sentences
        index: The index of the example to retrieve
        languages: Optional list of language codes to include (default: all available)
    
    Returns:
        Dictionary mapping language codes to sentences
    """
    if index >= len(dataset) or index < 0:
        raise ValueError(f"Index {index} out of range (0-{len(dataset)-1})")
    
    example = dataset[index]
    
    if languages is None:
        # Get all available languages excluding non-text columns
        languages = [col for col in example.keys() if col not in ['line_idx', 'index']]
    
    return {lang: example[lang] for lang in languages if lang in example}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a Hugging Face dataset from the North Levantine parallel corpus")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset files")
    parser.add_argument("--output_dir", type=str, default="north_levantine_dataset", help="Output directory for the dataset")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the dataset to the Hugging Face Hub")
    parser.add_argument("--repo_name", type=str, default="north_levantine_parallel", help="Name of the repository on Hugging Face Hub")
    parser.add_argument("--token", type=str, help="Hugging Face API token")
    parser.add_argument("--show_examples", type=int, default=3, help="Number of examples to display")
    
    args = parser.parse_args()
    
    # Create the dataset
    dataset = create_north_levantine_dataset(args.data_dir)["train"].train_test_split(test_size=0.15, seed=42)
    
    # Print some information about the dataset
    print("\nDataset Information:")
    print(f"Number of examples: {len(dataset['train'])}")
    print(f"Features: {dataset['train'].features}")
    
    # Show some examples
    print(f"\nShowing {args.show_examples} random examples:")
    import random
    sample_indices = random.sample(range(len(dataset['train'])), min(args.show_examples, len(dataset['train'])))
    
    for i, idx in enumerate(sample_indices):
        parallel_sentences = get_parallel_sentences(dataset['train'], idx)
        print(f"\nExample {i+1} (Index {idx}):")
        for lang, sentence in parallel_sentences.items():
            print(f"  {lang}: {sentence}")
    
    # Save the dataset locally

    dataset.save_to_disk(args.output_dir)
    print(f"Dataset saved to {args.output_dir}")

    
    # Push to Hugging Face Hub if requested
    if args.push_to_hub:
        dataset.push_to_hub(args.repo_name, token=args.token)
        print(f"Dataset pushed to Hugging Face Hub as {args.repo_name}")
    