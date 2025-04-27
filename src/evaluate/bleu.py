import argparse
import json
import os
from typing import List, Dict, Union, Optional
from tqdm import tqdm
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from evaluate import load as load_metric
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import wandb
from accelerate import PartialState
from accelerate.utils import gather_object


def download_nltk_resources():
    """Download required NLTK resources."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')

def validate_dataset_format(dataset):
    """
    Validate that the dataset has the required columns.
    
    Args:
        dataset: The dataset to validate
    
    Returns:
        True if the dataset format is valid, False otherwise
    """
    required_columns = ['prompt', 'completion']
    
    # Check if the dataset has source and target columns
    if all(col in dataset.column_names for col in required_columns):
        return True
    return False
 
def translate_texts(model, tokenizer, dataloader, max_length: int = 128):
    """
    Translate a list of texts using the provided model and tokenizer.
    
    Args:
        model: The translation model
        tokenizer: The tokenizer
        dataloader: DataLoader containing batches with "prompt" and "completion" keys
        max_length: The maximum length of the translated texts
    
    Returns:
        List of translated texts
    """
    print(f"Translating {len(dataloader)} texts...")
    
    # Generation config to get translation
    gen_config = GenerationConfig(
        max_new_tokens=max_length,
        do_sample=True,
        temperature = 0.8,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    translations = []
    # Create translation pipeline
    for batch in tqdm(dataloader, desc="Translating sentences"):
        batch_prompts = batch["prompt"]
        
        # Ensure prompts and completions are lists of strings
        if isinstance(batch_prompts, str):
            batch_prompts = [batch_prompts]
        
        prompt_lengths = []
        
        # Calculate prompt length in tokens (subtract 1 to not count the last token twice)
        for prompt in batch_prompts:
            prompt_tokens = tokenizer.encode(prompt)
            prompt_lengths.append(len(prompt_tokens) - 1 if len(prompt_tokens) > 0 else 0)
        
        # Tokenize the combined texts with padding
        encodings = tokenizer(
            batch_prompts, 
            padding=True, 
            return_tensors="pt", 
            truncation=True,
            max_length=max_length
        ).to(model.device)
        
        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask
        
        # Forward pass with labels
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                generation_config=gen_config,
            )
            
            
        # Get the length of each input in the batch to properly trim the outputs
        for i, (prompt_length, output) in enumerate(zip(prompt_lengths, outputs)):
            # Slice the output to get only the generated part (skip the prompt tokens)
            generated_tokens = output[prompt_length:]
            
            # Decode the generated tokens without special tokens
            translation = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            translations.append(translation)
            
    return translations

def tokenize_text(text: str):
    """
    Tokenize a text string into a list of tokens.
    
    Args:
        text: The text to tokenize
    
    Returns:
        List of tokens
    """
    return nltk.word_tokenize(text.lower())

def calculate_sentence_bleu(reference: str, candidate: str):
    """
    Calculate the BLEU score for a single sentence.
    
    Args:
        reference: The reference translation
        candidate: The candidate translation
    
    Returns:
        BLEU score
    """
    # Tokenize the reference and candidate
    reference_tokens = [tokenize_text(reference)]
    candidate_tokens = tokenize_text(candidate)
    
    # Use smoothing to avoid zero scores for short sentences
    smoothing = SmoothingFunction().method1
    
    # Calculate the BLEU score
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing)

def calculate_corpus_bleu(references: List[str], candidates: List[str]):
    """
    Calculate the corpus BLEU score.
    
    Args:
        references: The list of reference translations
        candidates: The list of candidate translations
    
    Returns:
        Corpus BLEU score
    """
    # NLTK corpus_bleu expects references and candidates in this format:
    # references: list of lists where each inner list contains reference tokens for a single example
    # candidates: list where each item is a list of tokens for a candidate translation
    refs_tokens = [[tokenize_text(ref)] for ref in references]
    cand_tokens = [tokenize_text(cand) for cand in candidates]
    
    return corpus_bleu(refs_tokens, cand_tokens)

def calculate_bleu_by_length(references: List[str], candidates: List[str], max_length: int = 50, step: int = 10):
    """
    Calculate BLEU scores grouped by sentence length.
    
    Args:
        references: The list of reference translations
        candidates: The list of candidate translations
        max_length: The maximum sentence length to consider
        step: The step size for length buckets
    
    Returns:
        Dictionary mapping length buckets to BLEU scores
    """
    length_buckets = {}
    
    for i in range(0, max_length, step):
        bucket_refs = []
        bucket_cands = []
        
        for ref, cand in zip(references, candidates):
            ref_length = len(tokenize_text(ref))
            if i <= ref_length < i + step:
                bucket_refs.append(ref)
                bucket_cands.append(cand)
        
        if bucket_refs:
            # Convert to the format expected by corpus_bleu
            refs_tokens = [[tokenize_text(ref)] for ref in bucket_refs]
            cand_tokens = [tokenize_text(cand) for cand in bucket_cands]
            
            bucket_bleu = corpus_bleu(refs_tokens, cand_tokens)
            length_buckets[f"{i}-{i+step-1}"] = {
                "bleu": bucket_bleu,
                "count": len(bucket_refs)
            }
    
    # Add a bucket for very long sentences
    bucket_refs = []
    bucket_cands = []
    for ref, cand in zip(references, candidates):
        ref_length = len(tokenize_text(ref))
        if ref_length >= max_length:
            bucket_refs.append(ref)
            bucket_cands.append(cand)
    
    if bucket_refs:
        # Convert to the format expected by corpus_bleu
        refs_tokens = [[tokenize_text(ref)] for ref in bucket_refs]
        cand_tokens = [tokenize_text(cand) for cand in bucket_cands]
        
        bucket_bleu = corpus_bleu(refs_tokens, cand_tokens)
        length_buckets[f"{max_length}+"] = {
            "bleu": bucket_bleu,
            "count": len(bucket_refs)
        }
    
    return length_buckets


def evaluate_bleu(dataset, model, tokenizer, state, batch_size: int = 8, max_length: int = 128):
    """
    Evaluate translations using BLEU score.
    
    Args:
        dataset: The dataset containing source and target texts
        model: The translation model
        tokenizer: The tokenizer
        batch_size: The batch size for translation
        max_length: The maximum length of the translated texts
    
    Returns:
        Dictionary containing the evaluation results
    """
    
    # Translate the source texts
    with state.split_between_processes(dataset) as partial_dataset:
        dataloader = DataLoader(partial_dataset, batch_size=batch_size)
        candidate_texts = translate_texts(model, tokenizer, dataloader,max_length=max_length)
    candidate_texts = gather_object(candidate_texts)
    
    # Kill non main processes   
    if not state.is_main_process:
        return

    # Format dataset by removing languages from prompt
    def bleu_format(row):
        prompt = row["prompt"].split(':')[1].split('\n')[0]
        completion = row["completion"]
        return {"prompt":prompt , "completion":completion}
    
    dataset = dataset.map(bleu_format,
        remove_columns=dataset.column_names,
        desc="Preparing dataset for BLEU evaluation",
        batched=False,
        batch_size=1, 
        )

    # Load the BLEU metric from Hugging Face's evaluate library
    bleu_metric = load_metric("bleu")
    
    # Format references for the HuggingFace BLEU metric
    # HuggingFace's BLEU metric expects a list of lists for references
    formatted_references = [[ref] for ref in dataset["completion"]]
    
    # Calculate corpus BLEU score using HuggingFace's implementation
    corpus_results = bleu_metric.compute(predictions=candidate_texts, references=formatted_references)
    corpus_bleu_score = corpus_results["bleu"]
    print(f"Corpus BLEU score: {corpus_bleu_score:.2f}")
    
    # Calculate BLEU scores by sentence length
    bleu_by_length = calculate_bleu_by_length(dataset["completion"], candidate_texts)
    
    # Calculate individual sentence BLEU scores
    sentence_bleu_scores = []
    for ref, cand in zip(formatted_references, candidate_texts):
        # Calculate BLEU for a single sentence
        results = bleu_metric.compute(predictions=[cand], references=[ref])
        sentence_bleu_scores.append(results["bleu"])
    
    avg_sentence_bleu = np.mean(sentence_bleu_scores)
    print(f"Average sentence BLEU score: {avg_sentence_bleu:.4f}")
    
    # Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    # Note: HuggingFace's BLEU implementation allows setting max_order
    bleu_1 = bleu_metric.compute(predictions=candidate_texts, references=formatted_references, max_order=1)["bleu"]
    bleu_2 = bleu_metric.compute(predictions=candidate_texts, references=formatted_references, max_order=2)["bleu"]
    bleu_3 = bleu_metric.compute(predictions=candidate_texts, references=formatted_references, max_order=3)["bleu"]
    bleu_4 = bleu_metric.compute(predictions=candidate_texts, references=formatted_references, max_order=4)["bleu"]
    
    # Prepare the results
    results = {
        "corpus_bleu": corpus_bleu_score,
        "average_sentence_bleu": float(avg_sentence_bleu),
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_3": bleu_3,
        "bleu_4": bleu_4,
        "bleu_by_length": bleu_by_length,
        "examples": []
    }
    
    # Add examples 
    num_examples = min(20, len(dataset["prompt"]))
    for i in range(num_examples):
        results["examples"].append({
            "source": dataset["prompt"][i],  # Use the original prompt with language tags
            "reference": dataset["completion"][i],
            "candidate": candidate_texts[i],
            "bleu": sentence_bleu_scores[i]
        })

    # Log main metrics
    wandb.log({
        "corpus_bleu": corpus_bleu_score,
        "average_sentence_bleu": float(avg_sentence_bleu),
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_3": bleu_3,
        "bleu_4": bleu_4,
    })
    
    # Log BLEU scores by sentence length
    for length_range, data in bleu_by_length.items():
        wandb.log({
            f"bleu_by_length/{length_range}/score": data["bleu"],
            f"bleu_by_length/{length_range}/count": data["count"]
        })
    
    # Create a table with example translations
    examples_table = wandb.Table(columns=["Source", "Reference", "Translation", "BLEU"])
    for i in range(20):
        examples_table.add_data(
            results["examples"][i]["source"],
            results["examples"][i]["reference"],
            results["examples"][i]["candidate"],
            results["examples"][i]["bleu"]
        )
    wandb.log({"translation_examples": examples_table})
    
    # Create a histogram of sentence BLEU scores
    wandb.log({"sentence_bleu_histogram": wandb.Histogram(sentence_bleu_scores)})
    
    # Create a line chart for BLEU by sentence length
    length_data = [[k, v["bleu"], v["count"]] for k, v in bleu_by_length.items()]
    length_table = wandb.Table(data=length_data, columns=["Length Range", "BLEU", "Count"])
    wandb.log({"bleu_by_length_chart": wandb.plot.line(
        length_table, "Length Range", "BLEU", title="BLEU Score by Sentence Length")
    })
    
    return results
    
def main():
    # Get CLI arguments
    parser = argparse.ArgumentParser(description="Calculate BLEU scores for translation models")
    parser.add_argument("--model", type=str, required=True,help="Path or name of the translation model")
    parser.add_argument("--dataset", type=str, required=True,help="Path or name of the dataset")
    parser.add_argument("--split", type=str, default="test",help="Dataset split to use (default: test)")
    parser.add_argument("--source_lang", type=str, default="arb",help="Source language code")
    parser.add_argument("--target_lang", type=str, default="apc",help="Target language code")
    parser.add_argument("--batch_size", type=int, default=8,help="Batch size for translation")
    parser.add_argument("--max_length", type=int, default=256,help="Maximum length of the translated texts")
    parser.add_argument("--wandb_project", type=str, default="CS512",help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,help="Weights & Biases run name (default: auto-generated)")
    args = parser.parse_args()
    
    # Use accelerate for distirbuted inference of model
    state = PartialState()
    device = state.device

    # Initialize wandb if enabled
    if state.is_main_process:
        project = args.wandb_project
        run_name = args.wandb_run_name or f"BLEU-{args.model.split('/')[-1]}-{dataset}"
        wandb.init(
            project=project,
            name=run_name,
            config=vars(args)
        )

    # Download NLTK resources
    download_nltk_resources()
    

    # Load the model and tokenizer
    print(f"Loading model and tokenizer from: {args.model}")
    with state.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(args.model,padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(args.model).requires_grad_(False).eval().to(device)
        if tokenizer.pad_token_id==None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id

        # Load the dataset
        if os.path.exists(args.dataset):
            dataset = DatasetDict.load_from_disk(args.dataset)[args.split]
        else:
            dataset = load_dataset(args.dataset)[args.split]
        
        def language_format(row):
            split = row["prompt"].split(':',1) 
            source = split[0] 
            split = split[1].rsplit('\n',1)
            target = split[1]
            if args.source_lang==source and args.target_lang==target:
                return row

        dataset = dataset.map(language_format,
            remove_columns=dataset.column_names,
            desc="Removing rows of non-specified langugages",
            batched=False,
            batch_size=1, 
            )

        # Validate the dataset format
        if not validate_dataset_format(dataset):
            raise ValueError("Dataset format is not valid for translation evaluation. "
                            "It should contain 'prompt' and 'completion' columns or equivalent.")
    
    # Evaluate the translations
    evaluate_bleu(
        dataset, model, tokenizer, state,
        batch_size=args.batch_size, 
        max_length=args.max_length,
    )

    if state.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()