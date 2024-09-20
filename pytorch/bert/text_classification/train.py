import argparse
import time
import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          DataCollatorWithPadding, TrainingArguments, Trainer)
import evaluate

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Train a DistilBERT model with optional mixed precision.")
parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision (AMP).')
args = parser.parse_args()

# Load and preprocess dataset
imdb = load_dataset("imdb")

# Load DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Preprocessing function to tokenize input text and truncate it to DistilBERT's max input length
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

# Apply preprocessing function to the entire dataset
tokenized_imdb = imdb.map(preprocess_function, batched=True)

# Create a batch of examples
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Evaluate
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Map model outputs to a readable label
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# Train
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

# Set fp16 in TrainingArguments based on command-line args
training_args = TrainingArguments(
    output_dir="bert_tc_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    fp16=args.amp  # Enable mixed precision if -amp is specified
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

# Train the model and record training time
torch.cuda.synchronize()
start_time = time.time()

trainer.train()

end_time = time.time()
torch.cuda.synchronize()

# Calculate elapsed time
elapsed_time = (end_time - start_time) / 60
print(f"Train Time: {elapsed_time:.2f} mins")