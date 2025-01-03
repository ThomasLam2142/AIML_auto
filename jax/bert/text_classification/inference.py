import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxBertForSequenceClassification
import numpy as np

# Load trained model and tokenizer
model = FlaxBertForSequenceClassification.from_pretrained("bert_tc")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize_input(text, max_length=128):
    return tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors="np")

# Inference function
@jax.jit
def infer(params, input_ids, attention_mask):
    # Perform inference
    logits = model(input_ids=input_ids, attention_mask=attention_mask, params=params, train=False).logits
    prediction = jnp.argmax(logits, axis=-1)
    return prediction

# Example inference
text = "This movie is great!"  
inputs = tokenize_input(text)  

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

params = model.params 

# Perform inference
prediction = infer(params, input_ids, attention_mask)

# Output prediction
label = prediction.item()
label_map = {0: "NEGATIVE", 1: "POSITIVE"}
readable_label = label_map[label]
print(f"Predicted label: {readable_label}")