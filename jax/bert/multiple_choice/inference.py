import jax
import jax.numpy as jnp

from transformers import (
    FlaxBertForMultipleChoice,
    BertTokenizerFast
)

def prepare_input(tokenizer, context, choices, max_length=64):
    contexts = [context] * len(choices)

    # Tokenize
    encoding = tokenizer(
        contexts,
        choices,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='np'
    )
    
    batch = {k: v[None, :] for k, v in encoding.items()}
    return batch

def main():
    # 1. Load the Flax BERT model for multiple choice
    model_name = "bert-base-uncased"
    model = FlaxBertForMultipleChoice.from_pretrained(model_name)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    # 2. Example question and possible choices
    question = "Jonas went to the store to get some dairy, what did he buy?"
    choices = [
        "Bread",
        "Milk",
        "Eggs",
        "Sugar"
    ]

    # 3. Prepare inputs
    inputs = prepare_input(tokenizer, question, choices)

    # 4. Forward pass
    outputs = model(**inputs)
    logits = outputs.logits  # shape: (batch_size, num_choices)

    # 5. Use jnp.argmax to select the best choice
    predicted_choice = jnp.argmax(logits, axis=-1)
    predicted_index = int(predicted_choice[0])  # batch size is 1 here
    print(f"\nQuestion: {question}")
    print("Choices:")
    for i, c in enumerate(choices):
        print(f"  {i}. {c}")
    print(f"\nPredicted answer: {choices[predicted_index]}")

if __name__ == "__main__":
    main()