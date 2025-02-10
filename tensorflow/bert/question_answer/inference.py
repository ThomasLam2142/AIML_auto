import time
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering

# Load fine-tuned model and tokenizer
MODEL_PATH = "./distilbert_qa"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = TFAutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)

# Define question and context
question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 natural languages and 13 programming languages."

# Encode inputs
inputs = tokenizer(question, context, return_tensors="tf")

# Measure inference time
start_time = time.time()

# Run inference
outputs = model(**inputs)

# Get answer start and end indices
answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

# Decode predicted answer
predicted_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
predicted_answer = tokenizer.decode(predicted_answer_tokens)

# Measure end time
end_time = time.time()

# Print result and time taken
print(f"\nPredicted Answer: {predicted_answer}")
print(f"Inference Time: {end_time - start_time:.4f} seconds")