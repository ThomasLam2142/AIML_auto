import argparse
import time
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          DataCollatorWithPadding, TrainingArguments, Trainer)
import evaluate

# For DataParallel, just run the script i.e. python3 train.py
# For DistributedDataParallel, use torchrun i.e. torchrun --nproc_per_node=<num_gpu> train.py

def main(args):
    # Load dataset and tokenizer
    imdb = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define model
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )

    # Load evaluation metric
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="bert_tc_model",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        fp16=args.amp,
        ddp_find_unused_parameters=False  # Enable DDP
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
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
    parser = argparse.ArgumentParser(description="Train a DistilBERT model with multi-GPU support.")
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision (AMP).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per device.')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train (default: 2)')
    args = parser.parse_args()

    main(args)