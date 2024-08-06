import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer, AutoConfig

text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert_tc_model/checkpoint-3126")

# Tokenize the input text and return tensors
inputs = tokenizer(text, return_tensors="np", max_length=7, padding='max_length', truncation=True)

# Load the ONNX model
onnx_model_path = "bert_tc_model_mixed.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["ROCMExecutionProvider"])

# Run inference
inputs_onnx = {
    'input': inputs['input_ids']
}
outputs = session.run(None, inputs_onnx)

# Get the logits from the output
logits = outputs[0]

# Assuming logits shape is [batch_size, num_classes]
num_classes = logits.shape[-1]

# Load model configuration to get labels
config = AutoConfig.from_pretrained("bert_tc_model/checkpoint-3126")
id2label = config.id2label

# Get the class with the highest probability and map the output to a label
predicted_class_id = np.argmax(logits, axis=-1).item()
output_label = id2label[predicted_class_id]
print(output_label)