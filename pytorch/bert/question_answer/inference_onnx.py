import onnxruntime as ort
from transformers import AutoTokenizer

# Create a question and context
question = "What is my name?"
context = "My name is Mern and I live in Barcelona"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert_qa_model/checkpoint-750")

# Tokenize the input text and return tensors
inputs = tokenizer(question, context, return_tensors="pt")

# Load the ONNX model
onnx_path = "distilbert_qa_model.onnx"
ort_session = ort.InferenceSession(onnx_path)

# Run inference
ort_inputs = {
    'input_ids': inputs['input_ids'].numpy(),
    'attention_mask': inputs['attention_mask'].numpy()
}
ort_outputs = ort_session.run(None, ort_inputs)

# Get the start and end logits
start_logits = ort_outputs[0]
end_logits = ort_outputs[1]

# Get the most likely beginning and end of answer
answer_start_index = start_logits.argmax()
answer_end_index = end_logits.argmax()

# Get the answer tokens and decode them
predict_answer_tokens = inputs['input_ids'][0, answer_start_index : answer_end_index + 1]
answer = tokenizer.decode(predict_answer_tokens)

print(f"Question: {question}")
print(f"Answer: {answer}")
