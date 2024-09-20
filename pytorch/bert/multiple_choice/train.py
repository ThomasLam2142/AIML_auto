import argparse
import torch
import time
from datasets import load_dataset
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import evaluate
import numpy as np

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Train a BERT model for multiple choice with optional mixed precision.")
parser.add_argument('--amp', action='store_true', help="Use automatic mixed precision (AMP) for training.")
args = parser.parse_args()

# Load SWAG dataset
swag = load_dataset("swag", "regular")

# Load BERT tokenizer to process the sentence starts and four possible endings
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

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

# Apply preprocessing function over the entire dataset
tokenized_swag = swag.map(preprocess_function, batched=True)

# Create examples - Transformers library doesn't offer data collating for multiple choice. 
@dataclass
class DataCollatorForMultipleChoice:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
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

# Define evaluation function
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Load model
model = AutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")

# Set up TrainingArguments
training_args = TrainingArguments(
    output_dir="bert_mc_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    fp16=args.amp,  # Enable mixed precision if --amp is provided
)

# Define Trainer class with mixed precision support
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if args.amp:
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                loss = outputs.loss
        else:
            outputs = model(**inputs)
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# Initialize Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_swag["train"],
    eval_dataset=tokenized_swag["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

# Synchronize and start the timer
if torch.cuda.is_available():
    torch.cuda.synchronize()

start_time = time.time()

# Perform training
trainer.train()

# Synchronize and end the timer
if torch.cuda.is_available():
    torch.cuda.synchronize()

end_time = time.time()

# Calculate and print the training time in minutes
training_duration = (end_time - start_time) / 60  # Convert seconds to minutes
print(f"Training Time = {training_duration:.2f} minutes")