import onnxruntime as ort
import torch
from transformers import AutoTokenizer

# Create a prompt and two candidate answers
prompt = "France has a bread law, Le Decret Pain, with strict rules on what is allowed in a traditional baguette."
candidate1 = "The law applies to baguettes."
candidate2 = "The law applies to automotive manufacturing."

candidates = [candidate1, candidate2]

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

# Tokenize the input text and return tensors
inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)

# Prepare the inputs for ONNX Runtime
ort_inputs = {k: v.unsqueeze(0).numpy() for k, v in inputs.items()}

# Load the ONNX model
ort_session = ort.InferenceSession("bert_mc_model_fp16.onnx", providers=["ROCMExecutionProvider"])

# Run inference
ort_outs = ort_session.run(None, ort_inputs)

# Get the logits from the output
logits = torch.tensor(ort_outs[0])

# Get the class with the highest probability
predicted_class = logits.argmax().item()
print(f"Prompt: {prompt}")
print(f"Answer: {candidates[predicted_class]}")