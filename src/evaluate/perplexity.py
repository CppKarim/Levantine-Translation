#!/usr/bin/env python3
# perplexity_calculator.py
"""
Calculate perplexity of a language model on a prompt-completion dataset.
Supports different model types (causal LM, seq2seq) and formats.
"""

import argparse
import json
import os
import math
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    StoppingCriteriaList,
    GPT2LMHeadModel,
    T5ForConditionalGeneration
)
import wandb
from accelerate import PartialState
from accelerate.utils import gather_object


def validate_dataset_format(dataset):
    """
    Validate that the dataset has the required columns.
    
    Args:
        dataset: The dataset to validate
    
    Returns:
        True if the dataset format is valid, False otherwise
    """
    # Check for prompt-completion format
    if 'prompt' in dataset.column_names and 'completion' in dataset.column_names:
        return True
    
    return False

def calculate_perplexity(model, tokenizer, dataset, batch_size, state, max_length=512):
    """
    Calculate perplexity for a causal language model.
    
    Args:
        model: The causal language model
        tokenizer: The tokenizer
        dataloader: DataLoader containing batches with "prompt" and "completion" keys
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with perplexity results
    """
    import torch
    import math
    from tqdm import tqdm
    
    nlls = []
    token_counts = []

    with state.split_between_processes(dataset) as partial_dataset:
        dataloader = DataLoader(partial_dataset, batch_size=batch_size)
        # Calculate perplexity for each batch
        for batch in tqdm(dataloader, desc="Calculating Perplexity"):
            batch_prompts = batch["prompt"]
            batch_completions = batch["completion"]
            
            # Ensure prompts and completions are lists of strings
            if isinstance(batch_prompts, str):
                batch_prompts = [batch_prompts]
            if isinstance(batch_completions, str):
                batch_completions = [batch_completions]
            
            batch_size = len(batch_prompts)
            combined_texts = []
            prompt_lengths = []
            
            # Combine prompts and completions
            for prompt, completion in zip(batch_prompts, batch_completions):
                combined_texts.append(prompt + completion)
                # Calculate prompt length in tokens (subtract 1 to not count the last token twice)
                prompt_tokens = tokenizer.encode(prompt)
                prompt_lengths.append(len(prompt_tokens) - 1 if len(prompt_tokens) > 0 else 0)
            
            # Tokenize the combined texts with padding
            encodings = tokenizer(
                combined_texts, 
                padding=True, 
                return_tensors="pt", 
                truncation=True,
                max_length=max_length
            ).to(model.device)
            
            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask
            
            # Create labels by shifting input_ids right (standard causal LM approach)
            labels = input_ids.clone()
            
            # Mask out prompt tokens and padding tokens in labels
            for i in range(batch_size):
                # Mask prompt tokens
                labels[i, :prompt_lengths[i]] = -100
                # Also mask padding tokens
                padding_positions = (attention_mask[i] == 0)
                labels[i, padding_positions] = -100
            
            # Forward pass with labels
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                loss = outputs.loss
                
            # Calculate per-example stats
            for i in range(batch_size):
                # Count non-masked tokens in completion
                completion_token_count = (labels[i] != -100).sum().item()
                
                if completion_token_count > 0:
                    # For per-example loss, we need to calculate it separately
                    # This is a simplification - ideally we'd compute per-example loss directly
                    neg_log_likelihood = loss.item() * completion_token_count
                    nlls.append(neg_log_likelihood)
                    token_counts.append(completion_token_count)
    nlls, token_counts = gather_object(nlls), gather_object(token_counts)

    # Calculate aggregate perplexity
    if not token_counts:
        return {
            "perplexity": float('inf'),
            "total_tokens": 0,
            "average_nll": float('inf'),
        }
    
    total_nll = sum(nlls)
    total_tokens = sum(token_counts)
    
    # Calculate perplexity (exponential of average negative log-likelihood)
    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    
    return {
        "perplexity": ppl,
        "total_tokens": total_tokens,
        "average_nll": avg_nll,
    }

def evaluate_perplexity(model, tokenizer, dataset, state, batch_size=8, max_length=512):
    """
    Calculate perplexity of the model on the dataset.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        dataset: The dataset containing prompt and completion
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        n_examples: Optional number of examples to evaluate
    
    Returns:
        Dictionary with perplexity results
    """

    # Calculate perplexity based on model type
    results = calculate_perplexity(model, tokenizer, dataset, batch_size, state, max_length=max_length)
    
    # Print the results
    if state.is_main_process:
        print(f"Perplexity: {results['perplexity']:.4f}")
        print(f"Total tokens: {results['total_tokens']}")
        print(f"Average negative log-likelihood: {results['average_nll']:.4f}")
    
        # Log results to W&B if enabled
        wandb.log(results)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Calculate model perplexity on a dataset")
    parser.add_argument("--model", type=str, required=True,help="Path to the model or model identifier from huggingface.co/models")
    parser.add_argument("--dataset", type=str, required=True,help="Path to the dataset or dataset identifier from huggingface.co/datasets")
    parser.add_argument("--split", type=str, default="test",help="Dataset split to use")
    parser.add_argument("--batch_size", type=int, default=4,help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=512,help="Maximum sequence length")
    parser.add_argument("--n_examples", type=int, default=None,help="Number of examples to evaluate (default: all)")
    parser.add_argument("--wandb_project", type=str, default="CS512",help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,help="Weights & Biases run name")
    
    args = parser.parse_args()
    
    # Use accelerate for distirbuted inference of model
    state = PartialState()
    device = state.device

    # Initialize wandb if enabled
    if state.is_main_process:
        run_name = args.wandb_run_name or f"perplexity-{args.model.split('/')[-1]}-{dataset}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model).requires_grad_(False).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id==None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Load the dataset
    # Dataset should have "prompt" and "completion" columns
    if os.path.exists(args.dataset):
        dataset = DatasetDict.load_from_disk(args.dataset)[args.split]
    else:
        dataset = load_dataset(args.dataset)[args.split]
    
    if not validate_dataset_format(dataset):
        raise ValueError("Dataset format is not valid. It should contain 'prompt'/'completion' columns.")
    
    # Calculate perplexity
    evaluate_perplexity(
        model, tokenizer, dataset, 
        batch_size=args.batch_size, 
        max_length=args.max_length,
        state=state
    )
    
    # Finish the wandb run if initialized
    if state.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()