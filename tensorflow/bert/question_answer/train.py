import os
import time
import tensorflow as tf
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DefaultDataCollator,
    create_optimizer,
    TFAutoModelForQuestionAnswering,
    set_seed,
)

# Workaround for TensorFlow-Hub and Keras 3 compatibility issue
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Enable logging
logging.basicConfig(level=logging.INFO)

# Set seed for reproducibility
set_seed(42)

# Enable multi-GPU training with MirroredStrategy
strategy = tf.distribute.MirroredStrategy()
print(f"Using {strategy.num_replicas_in_sync} GPU(s) for training.")

# Load and preprocess dataset
squad = load_dataset("squad", split="train[:5000]").train_test_split(test_size=0.2)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


def preprocess_function(examples):
    """Tokenize input questions and contexts and align answer positions."""
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
    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_mapping):
        answer = examples["answers"][i]
        start_char, end_char = answer["answer_start"][0], answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Determine context range
        context_start = next(idx for idx, sid in enumerate(sequence_ids) if sid == 1)
        context_end = next(idx for idx, sid in enumerate(sequence_ids) if sid != 1 and idx > context_start) - 1

        # If answer is out of bounds, assign (0,0)
        if offsets[context_start][0] > end_char or offsets[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Locate answer token positions
            start_idx = next(idx for idx in range(context_start, context_end + 1) if offsets[idx][0] >= start_char)
            end_idx = next(idx for idx in range(context_end, context_start - 1, -1) if offsets[idx][1] <= end_char)
            start_positions.append(start_idx)
            end_positions.append(end_idx)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


# Tokenize dataset
tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

# Prepare data collator
data_collator = DefaultDataCollator(return_tensors="tf")

# Hyperparameters
batch_size = 32 * strategy.num_replicas_in_sync  # Scale batch size based on GPUs
num_epochs = 5
total_train_steps = (len(tokenized_squad["train"]) // batch_size) * num_epochs

# Create training strategy scope
with strategy.scope():
    # Optimizer
    optimizer, schedule = create_optimizer(
        init_lr=2e-5,
        num_warmup_steps=0,
        num_train_steps=total_train_steps,
    )

    # Load model inside the strategy scope
    model = TFAutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")

    # Compile model
    model.compile(optimizer=optimizer)

# Convert to TensorFlow dataset
tf_train_set = model.prepare_tf_dataset(
    tokenized_squad["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)

tf_validation_set = model.prepare_tf_dataset(
    tokenized_squad["test"],
    shuffle=False,
    batch_size=batch_size,
    collate_fn=data_collator,
)

# Track training time
start_time = time.time()

# Train model
history = model.fit(tf_train_set, validation_data=tf_validation_set, epochs=num_epochs)

# Calculate total training time
end_time = time.time()
total_time_minutes = (end_time - start_time) / 60
print(f"\nTotal Training Time: {total_time_minutes:.2f} minutes")

# Save the trained model and tokenizer
model.save_pretrained("./distilbert_qa")
tokenizer.save_pretrained("./distilbert_qa")