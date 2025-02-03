import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from dataclasses import dataclass
from transformers import (AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import evaluate
import os

def main(args):
    # Set up device(s)
    available_gpus = torch.cuda.device_count()
    
    if available_gpus == 0:
        raise RuntimeError("No GPUs available.")
    
    if args.num_gpus > available_gpus:
        raise RuntimeError(f"Requested {args.num_gpus} GPUs, but only {available_gpus} are available.")
    
    num_gpus = min(args.num_gpus, available_gpus)
    
    # Restrict visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {num_gpus} GPU(s) for training.")

    # Load dataset and tokenizer
    swag = load_dataset("swag", "regular")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Define preprocess function
    ending_names = ["ending0", "ending1", "ending2", "ending3"]
    
    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples["sent1"]]
        question_headers = examples["sent2"]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]

        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
        return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    
    # Apply preprocessing to the dataset
    tokenized_swag = swag.map(preprocess_function, batched=True)

    # Debug: Print the structure of the first example in the dataset
    print("First example in the tokenized dataset:")
    print(tokenized_swag["train"][0])

    # Define data collator
    @dataclass
    class DataCollatorForMultipleChoice:
        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
    
        def __call__(self, features):
            # Debug: Print the features to check for 'input_ids'
            print("Features passed to the data collator:")
            print(features)
            
            # Ensure that 'input_ids' is present in the features
            if not features or "input_ids" not in features[0]:
                raise ValueError("The 'input_ids' key is missing from the features.")
            
            label_name = "label" if "label" in features[0] else "labels"
            labels = [feature.pop(label_name) for feature in features]
            batch_size = len(features)
            num_choices = len(features[0]["input_ids"])
            
            flattened_features = [
                [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
            ]
            flattened_features = sum(flattened_features, [])
    
            batch = self.tokenizer.pad(
                flattened_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
    
            batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
            return batch
    
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
    
    # Load model
    model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased")
    
    # Wrap model in DataParallel if using multiple GPUs
    if num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
    
    model.to(device)
    
    # Define evaluation function
    accuracy = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir="bert_mc_model",
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size // num_gpus if num_gpus > 0 else args.batch_size,
        per_device_eval_batch_size=args.batch_size // num_gpus if num_gpus > 0 else args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        fp16=args.amp
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_swag["train"],
        eval_dataset=tokenized_swag["validation"],
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
    parser = argparse.ArgumentParser(description="Train a BERT model for multiple choice with DataParallel multi-GPU support.")
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision (AMP).')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use (default: 1).')
    parser.add_argument('--batch_size', type=int, default=16, help='Total batch size (split across GPUs).')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train (default: 3)')
    args = parser.parse_args()
    
    main(args)