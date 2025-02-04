import os
import argparse
import time
import torch
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForQuestionAnswering,
                          DefaultDataCollator, TrainingArguments, Trainer)

# For DataParallel, just run the script i.e. python3 train.py
# For DistributedDataParallel, use torchrun i.e. torchrun nproc_per_node=<num_gpu> train.py

def main(args):
    # Check available GPUs
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("No GPUs available.")
    
    print(f"Using {available_gpus} GPU(s) for training.")

    # Load dataset
    squad = load_dataset("squad", split="train[:5000]").train_test_split(test_size=0.2)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions, end_positions = [], []
        
        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)
            
            # Find the start and end of the context
            idx = sequence_ids.index(1) if 1 in sequence_ids else 0
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1
            
            # Assign (0, 0) if answer is outside the context
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)
                
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
        
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    
    # Apply preprocessing
    tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
    data_collator = DefaultDataCollator()
    
    # Load model
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir="distilbert_qa_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        push_to_hub=False,
        fp16=args.amp  # Enable AMP if --amp is provided
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_squad["train"],
        eval_dataset=tokenized_squad["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    torch.cuda.synchronize()
    start_time = time.time()
    
    trainer.train()
    
    end_time = time.time()
    torch.cuda.synchronize()
    
    elapsed_time = (end_time - start_time) / 60
    print(f"Training completed in {elapsed_time:.2f} mins.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DistilBERT model for Question Answering with AMP.")
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision (AMP).')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per device.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train (default: 3)')
    args = parser.parse_args()
    
    main(args)