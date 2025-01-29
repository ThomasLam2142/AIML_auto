from transformers import BertTokenizer, FlaxBertForMultipleChoice
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from datasets import load_dataset

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = FlaxBertForMultipleChoice.from_pretrained('bert-base-uncased')

# Load dataset
dataset = load_dataset("swag", split="train")

# Preprocess dataset
def preprocess_function(examples):
    # Tokenize context and choices
    first_sentences = [[context] * 4 for context in examples["startphrase"]]
    second_sentences = [examples[ending] for ending in ["ending0", "ending1", "ending2", "ending3"]]

    # Flatten lists for tokenization
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    encodings = tokenizer(first_sentences, second_sentences, padding=True, truncation=True, return_tensors="np")
    
    # Reshape input tensors as required (for batching)
    # Tokenizing for the entire batch, without reshaping incorrectly
    return {key: encodings[key] for key in encodings}

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Define training step
class TrainState(TrainState):
    pass

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(**batch, params=params).logits
        labels = jax.nn.one_hot(batch["label"], num_classes=4)
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Define optimizer
optimizer = optax.adam(learning_rate=2e-5)
state = TrainState.create(
    apply_fn = model.__call__,
    params = model.params,
    tx = optimizer
)

# Training loop
for epoch in range(3):
    for batch in encoded_dataset:
        batch = {k: jnp.array(v, dtype=jnp.int32) for k, v in batch.items()}  # Ensure the batch contains numeric data
        state, loss = train_step(state, batch)
        print(f"Epoch {epoch}, Loss: {loss}")