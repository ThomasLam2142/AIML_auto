from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice
import torch

# Create a prompt and two candidate answers
prompt = "France has a bread law, Le Decret Pain, with strict rules on what is allowed in a traditional baguette."
candidate1 = "The law applies to baguettes."
candidate2 = "The law applies to automotive manufacturing."

candidates = [candidate1, candidate2]

# Tokenize each prompt-candidate pair and return PyTorch tensors
tokenzier = AutoTokenizer.from_pretrained("bert_mc_model/checkpoint-13791")
inputs = tokenzier([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)
labels = torch.tensor(0).unsqueeze(0)

# Pass the inputs and labels to the model and return the logits
model = AutoModelForMultipleChoice.from_pretrained("bert_mc_model/checkpoint-13791")
outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
logits = outputs.logits

# Get the class with the highest probability
predicted_class = logits.argmax().item()
print(f"Prompt: {prompt}")
print(f"Answer: {candidates[predicted_class]}")