from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

# Tokenize the text and return tensors
tokenizer = AutoTokenizer.from_pretrained("bert_tc_model/checkpoint-3126")
inputs = tokenizer(text, return_tensors="pt")

# Pass tokenized inputs into the model and return the logits
model = AutoModelForSequenceClassification.from_pretrained("bert_tc_model/checkpoint-3126")
with torch.no_grad():
    logits = model(**inputs).logits

# Get the class with the highest probability and map the output to a label
predicted_class_id = logits.argmax().item()
output = model.config.id2label[predicted_class_id]
print(output)