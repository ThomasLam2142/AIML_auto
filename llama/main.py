import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/taccuser/llama3_converted")
model = AutoModelForCausalLM.from_pretrained("/home/taccuser/llama3_converted")

pipeline = transformers.pipeline("text-generation", model=model, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
pipeline("Hey how are you doing today?")