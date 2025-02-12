import jax
import jax.numpy as jnp
from transformers import FlaxBertForSequenceClassification, AutoTokenizer
import optax
from datasets import load_dataset
import numpy as np
import random
import time
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a BERT model on the IMDB dataset using JAX.")
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation.')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate for optimizer.')
args = parser.parse_args()

# Load dataset
dataset = load_dataset("imdb")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load model
model = FlaxBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
params = model.params

# Tokenization function
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128, return_tensors="np")

# Tokenize datasets
tokenized_datasets = dataset.map(tokenize, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format(type='numpy', columns=['input_ids', 'attention_mask', 'labels'])

# Define optimizer
optimizer = optax.adam(args.learning_rate)
opt_state = optimizer.init(params)

# Initialize PRNG key
rng = jax.random.PRNGKey(42)

# Training step
@jax.jit
def train_step(params, opt_state, batch, rng):
    rng, dropout_rng = jax.random.split(rng)  # Ensure dropout randomness per step

    def loss_fn(params):
        logits = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            params=params,
            dropout_rng=dropout_rng,  # Use updated dropout_rng
            train=True,
        ).logits
        one_hot_labels = jax.nn.one_hot(batch['labels'], num_classes=2)
        loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, rng  # Return updated RNG

# Evaluation step
@jax.jit
def evaluate_step(params, batch):
    logits = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        params=params,
        train=False,
    ).logits
    probabilities = jax.nn.softmax(logits, axis=-1)  # Convert logits to probabilities
    predictions = jnp.argmax(probabilities, axis=-1)
    accuracy = jnp.mean(predictions == batch['labels'])
    return accuracy

# Function to create batches with shuffling
def create_batches(dataset, batch_size):
    dataset_length = len(dataset['input_ids'])
    indices = list(range(dataset_length))
    random.shuffle(indices)  # Shuffle data before batching

    for i in range(0, dataset_length, batch_size):
        batch_indices = indices[i:i + batch_size]
        yield {
            key: dataset[key][batch_indices]
            for key in ['input_ids', 'attention_mask', 'labels']
        }

# Train and evaluate
total_start_time = time.time()

for epoch in range(args.epochs):
    epoch_start_time = time.time()
    print(f"Epoch {epoch + 1}/{args.epochs} started.")

    # Training loop
    train_loss = []
    for i, batch in enumerate(create_batches(tokenized_datasets['train'], args.batch_size)):
        rng, input_rng = jax.random.split(rng)
        params, opt_state, loss, rng = train_step(params, opt_state, batch, rng)
        train_loss.append(loss)

        if i % 50 == 0:
            print(f"Step {i}, Loss: {loss:.4f}")

    avg_train_loss = np.mean(train_loss)
    print(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}")

    # Evaluation loop
    accuracies = []
    for batch in create_batches(tokenized_datasets['test'], args.batch_size):
        accuracy = evaluate_step(params, batch)
        accuracies.append(float(accuracy))  # Convert JAX tensor to float for NumPy

    avg_accuracy = np.mean(np.array(accuracies))
    print(f"Epoch {epoch + 1} Validation Accuracy: {avg_accuracy * 100:.2f}%")

    epoch_end_time = time.time()
    print(f"Epoch {epoch + 1} took {(epoch_end_time - epoch_start_time) / 60:.2f} minutes.")

total_end_time = time.time()
print(f"Total training time: {(total_end_time - total_start_time) / 60:.2f} minutes.")

# Save model and tokenizer
model.save_pretrained("bert_tc", params=params)  # Ensure params are saved
tokenizer.save_pretrained("bert_tc")