import time
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxBertForSequenceClassification
import numpy as np

# Load trained model and tokenizer
model = FlaxBertForSequenceClassification.from_pretrained("bert_tc")
tokenizer = AutoTokenizer.from_pretrained("bert_tc")

# Tokenization function
def tokenize_input(text, max_length=128):
    return tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )

# Inference function
@jax.jit
def infer(params, input_ids, attention_mask):
    # Perform inference
    logits = model(input_ids=input_ids, attention_mask=attention_mask, params=params, train=False).logits
    probabilities = jax.nn.softmax(logits, axis=-1)  # Convert logits to probabilities
    prediction = jnp.argmax(probabilities, axis=-1)  # Get predicted label
    confidence = probabilities[0, prediction]  # Extract confidence score
    return prediction, confidence

# Example inference
text = "This movie is great!"  
inputs = tokenize_input(text)  

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

params = model.params 

# Warm-up call (optional)
_ = infer(params, input_ids, attention_mask)

# Measure inference time
start_time = time.time()
prediction, confidence = infer(params, input_ids, attention_mask)
end_time = time.time()
inference_time = end_time - start_time

# Output prediction
label = prediction.item()
label_map = {0: "NEGATIVE", 1: "POSITIVE"}
readable_label = label_map[label]
confidence_score = confidence.item()

print(f"Predicted label: {readable_label} ({prediction}) with confidence {confidence_score:.2f}")
print(f"Inference Time: {inference_time:.4f} seconds")