# seq_2_seq.py
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, setup_chat_format
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments
from accelerate import Accelerator, PartialState
import wandb
import os
import argparse
import evaluate
import numpy as np
  


def get_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action="store_true", default = False, help="Loads small chunk of data to check if script works.")

    parser.add_argument('--model_name', type=str, default ="meta-llama/Llama-3.2-1B", help="Pretrined model to be finetuned. Defaults to Llama 3.2 1B.")
    parser.add_argument('--bf16', action="store_true", default = False, help="Whether to use bf16.")
    parser.add_argument('--fp16', action="store_true", default = False, help="Whether to use fp16.")

    parser.add_argument('--dataset', type=str, default = "KHuss/hh-rlhf-formatted", help="Path to dataset, local or on hugginface.")
    parser.add_argument('--chat', action="store_true", default = False, help="Whether the dataset is in conversational format.")
    parser.add_argument('--dataset_offset', type=int, default = 0, help="What percentage of the dataset to ignore in the beginning.")
    parser.add_argument('--dataset_percent', type=int, default = 20, help="What percentage of the dataset will be used for finetuning.")
    parser.add_argument('--dataset_num_proc', type=int, default = 64, help="Number of cpu processes used for dataset processing.")
    parser.add_argument('--save_model', action="store_true", default = False, help="Whether to save model to disk.")

    parser.add_argument('--subdirectory', type=str, default = "sft", help="What directory under data/models/{model_name}/ to save the checkpoints.")
    parser.add_argument('--tag', type=str, default = "new_sft", help="Extra tag for wandb tracking.")

    parser.add_argument('--learning_rate', type=float, default = 0.00008387849149162703 , help="Initial learning rate of AdamW optimizer.")
    parser.add_argument('--weight_decay', type=float, default = 0.0 , help="What weight decay (L2 reg) value to use for the AdamW optimizer.")
    parser.add_argument('--lr_scheduler', type=str, default = "linear" ,choices=["linear","cosine","cosine_with_restarts"], help="Type of learnig rate schedule. Choose between linear and cosine.")
    parser.add_argument('--lr_decay', type=float, default = 0.0, help="Fraction of learning rate to decay to by end of training, only applies if schedule is cosine. Defaults to 0.")
    parser.add_argument('--num_restarts', type=int, default = 1, help="Number of times to restart learning, only applies if schedule is cosine with restarts. Defaults to 1.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default = 32, help="Gradient accumulation steps. Effective batch size = GAS*per_gpu_batch*num_gpus.")
    parser.add_argument('--per_device_batch', type=int, default = 2, help="Batch size per accelerator device")
    parser.add_argument('--seq_len', type=int, default = 512, help="To what length should the examples be truncated to.")
    parser.add_argument('--epochs', type=int, default = 3, help="Number of training epochs")
    parser.add_argument('--max_grad_norm', type=float, default = 1.0, help="Maximum gradient value for one optimization step.")

    parser.add_argument('--save_interval', type=float, default = 0.05, help="Saving periodicity/total training step.")
    parser.add_argument('--early_stopping', type=int, default = None, help="Patience for early stopping.")
    parser.add_argument('--grad_check', action="store_true", default = False, help="Whether to use gradient checkpointing")
    parser.add_argument('--ipex', action="store_true", default = False, help="Whether to optimize of Intel GPUs.")

    return parser.parse_args()


def main():
    #########
    # 0. OS environment, args and initialization
    #########
    args = get_cli()
    model_name = args.model_name
    if PartialState().is_main_process:
        wandb.init(project="CS512",tags=["seq_2_seq",args.model_name, args.tag])

    #########
    # 1. Load Model, Tokenizer 
    #########
    dtype = "float32" if not (args.bf16 and args.fp16) else "bfloat16" if args.bf16 else "float16"

    with PartialState().local_main_process_first():
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_size = "left", truncation_side= "left")
    tokenizer.pad_token_id = tokenizer.unk_token_id 
    model.config.pad_token_id = tokenizer.pad_token_id

    with PartialState().local_main_process_first():
        if args.debug:
            train_dataset = load_dataset(args.dataset, split=f"train[:200]")
            eval_dataset = load_dataset(args.dataset, split="test[:100]")
        else:
            train_dataset = load_dataset(args.dataset, split=f"train[{args.dataset_offset}%:{args.dataset_percent}%]")
            eval_dataset = load_dataset(args.dataset, split="test")

    #########
    # 2. Initialize Value trainer
    #########
    dir = f"./data/models/{model_name}/{args.subdirectory}"
    os.makedirs(f"{dir}/final",exist_ok=True)

    training_args = SFTConfig(
        output_dir=dir, save_total_limit=2, 
        load_best_model_at_end=True,metric_for_best_model="loss",greater_is_better=True,
        logging_steps=1e-3, eval_steps = args.save_interval, save_steps= args.save_interval, logging_strategy='steps', eval_strategy='steps', save_strategy='steps' if args.save_model else 'no', 
        max_grad_norm=args.max_grad_norm, learning_rate= args.learning_rate , gradient_accumulation_steps=args.gradient_accumulation_steps, auto_find_batch_size= False, num_train_epochs=args.epochs,weight_decay = args.weight_decay,
        max_seq_length=args.seq_len, 
        per_device_train_batch_size= args.per_device_batch, per_device_eval_batch_size=args.per_device_batch,eval_accumulation_steps=4,
        bf16= args.bf16, bf16_full_eval= args.bf16,
        fp16= args.fp16, fp16_full_eval= args.fp16,
        report_to= 'wandb', run_name = "sft_train",
        ddp_find_unused_parameters=False,
        gradient_checkpointing = args.grad_check,
        dataset_num_proc=args.dataset_num_proc,
        use_ipex=args.ipex,
        )
    callbacks = []
    if args.early_stopping:
        callbacks.append(EarlyStoppingCallback(args.early_stopping))

    def compute_metrics(eval_pred):
        bleu = evaluate.load("sacrebleu")
        
        predictions, labels = eval_pred
        predictions = predictions.cpu()
        labels = labels.cpu()
        
        # Replace -100 with the pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Convert ids to tokens
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {"bleu": result["score"]}

    trainer = SFTTrainer(
        model=model,
        args=training_args, 
        processing_class=tokenizer, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        #data_collator=data_collator,
        #compute_metrics=compute_metrics,
        callbacks = callbacks
        )
    
    
    #########
    # 3. Training Loop
    #########
    trainer.train()
    trainer.evaluate()
    if args.save_model and PartialState().is_local_main_process:
        trainer.save_model(f"{dir}/final")

if __name__ == "__main__":
    main()

