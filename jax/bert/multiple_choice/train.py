from transformers import BertTokenizer, FlaxBertForMultipleChoice
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from datasets import load_dataset

# Load dataset, tokenizer, and model
dataset = load_dataset("swag", "regular")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = FlaxBertForMultipleChoice.from_pretrained('bert-base-uncased')

# Preprocess dataset
ending_names = ["ending0", "ending1", "ending2", "ending3"]

def preprocess_function(examples):
    # Tokenize context and choices
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    question_headers = examples["sent2"]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    # Flatten lists for tokenization
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    encodings = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in encodings.items()}

encoded_dataset = dataset.map(preprocess_function, batched=True)
print(encoded_dataset)