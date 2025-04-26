import os
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from camel_tools.utils.charmap import CharMapper

# Initialize the Arabic to Buckwalter (HSB version) transliteration mapper
ar2bw = CharMapper.builtin_mapper('ar2hsb')
bw2ar = CharMapper.builtin_mapper('hsb2ar')

def load_text_file(file_path):
    """Load a text file and return a list of lines."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def format_finetune(row):
    """
    Format function for the finetune dataset.
    Creates empty prompts and completions with the text content.
    
    Args:
        row: A row from the dataset with eng, arb, and apc columns
        
    Returns:
        Dictionary with prompt and completion lists
    """
    # Check if the row values are lists or single strings
    if not isinstance(row["eng"], list):
        comp = [row["eng"], row["arb"], row["apc"]]
        promp = [""] * 3  # Empty prompts for each completion
    else:
        # Handle list case (though less likely in this context)
        comp = row["eng"] + row["arb"] + row["apc"]
        promp = [""] * len(comp)
    
    return {"prompt": promp, "completion": comp}

def format_finetune_roman(row):
    """
    Format function for the finetune_roman dataset.
    Creates empty prompts and completions with the text content,
    with Arabic texts transliterated to Latin script.
    
    Args:
        row: A row from the dataset with eng, arb, and apc columns
        
    Returns:
        Dictionary with prompt and completion lists
    """
    # Check if the row values are lists or single strings
    if not isinstance(row["eng"], list):
        comp = [
            row["eng"], 
            ar2bw(row["arb"]), 
            ar2bw(row["apc"])
        ]
        promp = [""] * 3  # Empty prompts for each completion
    else:
        # Handle list case
        comp = (
            row["eng"] + 
            [ar2bw(ar) for ar in row["arb"]] + 
            [ar2bw(ar) for ar in row["apc"]]
        )
        promp = [""] * len(comp)
    
    return {"prompt": promp, "completion": comp}

def format_translate(row):
    """
    Format function for the translate dataset.
    Creates prompts with source language text and completions with target language text.
    
    Args:
        row: A row from the dataset with eng, arb, and apc columns
        
    Returns:
        Dictionary with prompt and completion pairs for translation
    """
    prompts = []
    completions = []
    
    # Check if the row values are lists or single strings
    if not isinstance(row["eng"], list):
        # English to Arabic translations
        prompts.append(f"english: {row['eng']}\nclassical arabic:")
        completions.append(f"{row['arb']}")
        
        prompts.append(f"english: {row['eng']}\nlevantine arabic:")
        completions.append(f"{row['apc']}")
        
        # Classical Arabic to other languages
        prompts.append(f"classical arabic: {row['arb']}\nenglish:")
        completions.append(f"{row['eng']}")
        
        prompts.append(f"classical arabic: {row['arb']}\nlevantine arabic:")
        completions.append(f"{row['apc']}")
        
        # Levantine Arabic to other languages
        prompts.append(f"levantine arabic: {row['apc']}\nenglish:")
        completions.append(f"{row['eng']}")
        
        prompts.append(f"levantine arabic: {row['apc']}\nclassical arabic:")
        completions.append(f"{row['arb']}")
    else:
        # Handle list case (though less likely in this context)
        for i in range(len(row["eng"])):
            # English to Arabic translations
            prompts.append(f"english: {row['eng'][i]}\nclassical arabic:")
            completions.append(f"{row['arb'][i]}")
            
            prompts.append(f"english: {row['eng'][i]}\nlevantine arabic:")
            completions.append(f"{row['apc'][i]}")
            
            # Classical Arabic to other languages
            prompts.append(f"classical arabic: {row['arb'][i]}\nenglish:")
            completions.append(f"{row['eng'][i]}")
            
            prompts.append(f"classical arabic: {row['arb'][i]}\nlevantine arabic:")
            completions.append(f"{row['apc'][i]}")
            
            # Levantine Arabic to other languages
            prompts.append(f"levantine arabic: {row['apc'][i]}\nenglish:")
            completions.append(f"{row['eng'][i]}")
            
            prompts.append(f"levantine arabic: {row['apc'][i]}\nclassical arabic:")
            completions.append(f"{row['arb'][i]}")
    
    return {"prompt": prompts, "completion": completions}

def format_translate_roman(row):
    """
    Format function for the translate_roman dataset.
    Creates prompts with source language text and completions with target language text,
    with Arabic texts transliterated to Latin script.
    
    Args:
        row: A row from the dataset with eng, arb, and apc columns
        
    Returns:
        Dictionary with prompt and completion pairs for translation
    """
    prompts = []
    completions = []
    
    # Check if the row values are lists or single strings
    if not isinstance(row["eng"], list):
        # Transliterate Arabic texts
        arb_roman = ar2bw(row['arb'])
        apc_roman = ar2bw(row['apc'])
        
        # English to Arabic translations
        prompts.append(f"english: {row['eng']}\nclassical arabic:")
        completions.append(f"{arb_roman}")
        
        prompts.append(f"english: {row['eng']}\nlevantine arabic:")
        completions.append(f"{apc_roman}")
        
        # Classical Arabic to other languages
        prompts.append(f"classical arabic: {arb_roman}\nenglish:")
        completions.append(f"{row['eng']}")
        
        prompts.append(f"classical arabic: {arb_roman}\nlevantine arabic:")
        completions.append(f"{apc_roman}")
        
        # Levantine Arabic to other languages
        prompts.append(f"levantine arabic: {apc_roman}\nenglish:")
        completions.append(f"{row['eng']}")
        
        prompts.append(f"levantine arabic: {apc_roman}\nclassical arabic:")
        completions.append(f"{arb_roman}")
    else:
        # Handle list case
        for i in range(len(row["eng"])):
            # Transliterate Arabic texts
            arb_roman = ar2bw(row['arb'][i])
            apc_roman = ar2bw(row['apc'][i])
            
            # English to Arabic translations
            prompts.append(f"english: {row['eng'][i]}\nclassical arabic:")
            completions.append(f"{arb_roman}")
            
            prompts.append(f"english: {row['eng'][i]}\nlevantine arabic:")
            completions.append(f"{apc_roman}")
            
            # Classical Arabic to other languages
            prompts.append(f"classical arabic: {arb_roman}\nenglish:")
            completions.append(f"{row['eng'][i]}")
            
            prompts.append(f"classical arabic: {arb_roman}\nlevantine arabic:")
            completions.append(f"{apc_roman}")
            
            # Levantine Arabic to other languages
            prompts.append(f"levantine arabic: {apc_roman}\nenglish:")
            completions.append(f"{row['eng'][i]}")
            
            prompts.append(f"levantine arabic: {apc_roman}\nclassical arabic:")
            completions.append(f"{arb_roman}")
    
    return {"prompt": prompts, "completion": completions}

def load_and_process_dataset(dataset_name_or_path):
    """
    Load a dataset from Hugging Face Hub or local path and process it.
    
    Args:
        dataset_name_or_path: Name of the dataset on HF Hub or path to local dataset
    
    Returns:
        A dictionary of processed datasets
    """
    # Load the dataset
    if os.path.exists(dataset_name_or_path):
        dataset = DatasetDict.load_from_disk(dataset_name_or_path)
    else:
        from datasets import load_dataset
        dataset = load_dataset(dataset_name_or_path)
    
    print(f"Loaded dataset with {len(dataset['train'])} train examples")
    
    # Process the dataset for each format
    processed_datasets = {}
    
    processed_datasets["finetune"] = dataset.map(
        format_finetune,
        remove_columns=dataset["train"].column_names,
        desc="Processing finetune dataset",
        batched=True,
        batch_size=64,
    )
    
    processed_datasets["translate"] = dataset.map(
        format_translate,
        remove_columns=dataset["train"].column_names,
        desc="Processing translate dataset",
        batched=True,
        batch_size=64,
    )
    
    processed_datasets["finetune_roman"] = dataset.map(
        format_finetune_roman,
        remove_columns=dataset["train"].column_names,
        desc="Processing finetune_roman dataset",
        batched=True,
        batch_size=64,
    )
    
    processed_datasets["translate_roman"] = dataset.map(
        format_translate_roman,
        remove_columns=dataset["train"].column_names,
        desc="Processing translate_roman dataset",
        batched=True,
        batch_size=64,
    )
    return processed_datasets

def save_to_disk(dataset_dict, output_dir):
    """Save the datasets to disk in Hugging Face format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each processed dataset to a subdirectory
    for name, dataset in dataset_dict.items():
        dataset_path = os.path.join(output_dir, name)
        dataset.save_to_disk(dataset_path)
        print(f"Dataset '{name}' saved to {dataset_path}")

def push_to_hub(dataset_dict, repo_name, token=None):
    """Push the datasets to the Hugging Face Hub."""
    for name, dataset in dataset_dict.items():
        full_repo_name = f"{repo_name}_{name}"
        dataset.push_to_hub(full_repo_name, token=token)
        print(f"Dataset '{name}' pushed to Hugging Face Hub as {full_repo_name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process a parallel corpus dataset into multiple formats")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Path to the dataset on disk or huggingface name")
    parser.add_argument("--output_dir", type=str, default="processed_datasets", 
                        help="Output directory for the processed datasets")
    parser.add_argument("--push_to_hub", action="store_true", 
                        help="Whether to push the datasets to the Hugging Face Hub")
    parser.add_argument("--repo_name", type=str, default="UFAL", 
                        help="Base name of the repositories on Hugging Face Hub")
    parser.add_argument("--token", type=str, help="Hugging Face API token")
    parser.add_argument("--show_examples", type=int, default=3, 
                        help="Number of examples to show from each processed dataset")
    
    args = parser.parse_args()
    
    # Load and process the dataset
    processed_datasets = load_and_process_dataset(args.dataset)
    
    # Print information about the processed datasets
    print("\nProcessed Dataset Information:")
    for name, dataset in processed_datasets.items():
        print(f"\n{name} dataset:")
        print(f"  Number of examples: {len(dataset)}")
        print(f"  Features: {dataset["train"].features}")
        
        # Show some examples
        print(f"\n  Showing {min(args.show_examples, len(dataset))} examples from {name} dataset:")
        for i in range(min(args.show_examples, len(dataset))):
            print(f"    Example {i+1}:")
            print(f"      Prompt: {dataset["train"][i]['prompt']}")
            print(f"      Completion: {dataset["train"][i]['completion']}")

    # Save the datasets locally
    save_to_disk(processed_datasets, args.output_dir)
    
    # Push to Hugging Face Hub if requested
    if args.push_to_hub:
        push_to_hub(processed_datasets, args.repo_name, args.token)