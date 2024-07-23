from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
import torch

question = "Where do I live?"
context  = "My name is Mern and I live in Barcelona"

tokenizer = AutoTokenizer.from_pretrained("distilbert_qa_model/checkpoint-750")
inputs = tokenizer(question, context, return_tensors="pt")

model = AutoModelForQuestionAnswering.from_pretrained("distilbert_qa_model/checkpoint-750")
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
answer = tokenizer.decode(predict_answer_tokens)
print(f"Question: {question}")
print(f"Answer: {answer}")