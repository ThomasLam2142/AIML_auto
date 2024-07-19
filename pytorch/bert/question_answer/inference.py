from transformers import pipeline

question = "Where do I live?"
context = "My name is Mern, and I live in Barcelona"

question_answerer = pipeline("question-answering", model="distilbert_qa_model/checkpoint-500")
output = question_answerer(question = question, context=context)
print(output)