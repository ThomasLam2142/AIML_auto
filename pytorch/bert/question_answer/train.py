import argparse
import torch
import time
from datasets import load_dataset
from transformers import AutoTokenizer, DefaultDataCollator, AutoModelForQuestionAnswering, TrainingArguments, Trainer

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Train a model with different precisions.")
parser.add_argument('--amp', action='store_true', help="Use automatic mixed precision (AMP) for training")
args = parser.parse_args()

# Load a subset of the SQuAD dataset from the Datasets library and split the data into test and train groups
squad = load_dataset("squad", split="train[:5000]")
squad = squad.train_test_split(test_size=0.2)

# Preprocess
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
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"]  = end_positions
    return inputs

# Apply the preprocessing function over the entire dataset
tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

# Create a batch of examples
data_collator = DefaultDataCollator()

# Train model
model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")

# TrainingArguments configuration
training_args = TrainingArguments(
    output_dir="distilbert_qa_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
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

# Measure the training time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.cuda.synchronize()

start_time = time.time()

trainer.train()

if device.type == 'cuda':
    torch.cuda.synchronize()

end_time = time.time()

# Calculate and print the total training time in minutes
training_duration = (end_time - start_time) / 60
print(f"Training Time = {training_duration:.2f} minutes")