import jax
import jax.numpy as jnp
from transformers import FlaxBertForSequenceClassification, AutoTokenizer
import optax
from datasets import load_dataset
import numpy as np
from flax import serialization
import time  # Importing the time module

# Load the dataset
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
learning_rate = 2e-5
weight_decay = 0.01
optimizer = optax.adam(learning_rate, weight_decay)
opt_state = optimizer.init(params)

# Initialize PRNG key
rng = jax.random.PRNGKey(42)

# Training step
@jax.jit
def train_step(params, opt_state, batch, rng):
    def loss_fn(params):
        logits = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            params=params,
            dropout_rng=rng,
            train=True,
        ).logits
        one_hot_labels = jax.nn.one_hot(batch['labels'], num_classes=2)
        loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Evaluation step
@jax.jit
def evaluate_step(params, batch):
    logits = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        params=params,
        train=False,
    ).logits
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == batch['labels'])
    return accuracy

# Function to create batches
def create_batches(dataset, batch_size):
    dataset_length = len(dataset['input_ids'])
    for i in range(0, dataset_length, batch_size):
        yield {
            key: dataset[key][i:i + batch_size]
            for key in ['input_ids', 'attention_mask', 'labels']
        }

# Train and evaluate
num_epochs = 3
batch_size = 16
total_start_time = time.time()  # Start the overall training timer

for epoch in range(num_epochs):
    epoch_start_time = time.time()  # Start the timer for this epoch
    print(f"Epoch {epoch + 1}/{num_epochs} started.")
    
    # Training loop
    train_loss = []
    for i, batch in enumerate(create_batches(tokenized_datasets['train'], batch_size)):
        rng, input_rng = jax.random.split(rng)  # Update RNG for each batch
        params, opt_state, loss = train_step(params, opt_state, batch, input_rng)
        train_loss.append(loss)
        if i % 50 == 0:
            print(f"Step {i}, Loss: {loss:.4f}")

    avg_train_loss = np.mean(train_loss)
    print(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}")

    # Evaluation loop
    accuracies = []
    for batch in create_batches(tokenized_datasets['test'], batch_size):
        accuracy = evaluate_step(params, batch)
        accuracies.append(accuracy)

    avg_accuracy = np.mean(np.array(accuracies))
    print(f"Epoch {epoch + 1} Validation Accuracy: {avg_accuracy * 100:.2f}%")
    
    epoch_end_time = time.time()  # End the timer for this epoch
    epoch_duration = (epoch_end_time - epoch_start_time) / 60  # Convert to minutes
    print(f"Epoch {epoch + 1} took {epoch_duration:.2f} minutes.")

total_end_time = time.time()  # End the overall training timer
total_duration = (total_end_time - total_start_time) / 60  # Convert to minutes
print(f"Total training time: {total_duration:.2f} minutes.")

# Save model
model.save_pretrained("bert_tc")