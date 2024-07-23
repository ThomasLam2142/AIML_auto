import onnx
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load the trained model and tokenizer
model_path = "distilbert_qa_model/checkpoint-750"
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Create dummy input
question = "What is the capital of France?"
context = "France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris is its capital and largest city."
inputs = tokenizer(question, context, return_tensors="pt")

# Convert to ONNX format
onnx_path = "distilbert_qa_model.onnx"
torch.onnx.export(
    model, 
    (inputs['input_ids'], inputs['attention_mask']),
    onnx_path, 
    input_names=['input_ids', 'attention_mask'],
    output_names=['start_logits', 'end_logits'],
    opset_version=11,
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'start_logits': {0: 'batch_size', 1: 'sequence_length'},
        'end_logits': {0: 'batch_size', 1: 'sequence_length'}
    }
)
# Verify the model's structure
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
