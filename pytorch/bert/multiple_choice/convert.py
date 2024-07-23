from transformers import AutoTokenizer, AutoModelForMultipleChoice
import torch
import onnx

# Load the trained model and tokenizer
model_path = "bert_mc_model/checkpoint-13791"
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModelForMultipleChoice.from_pretrained(model_path)

# Create dummy prompt and candidates
dummy_prompt = "This is a dummy prompt."
dummy_candidates = ["This is the first dummy candidate.", "This is the second dummy candidate.",
                    "This is the third dummy candidate.", "This is the fourth dummy candidate."]

# Tokenize the dummy inputs
inputs = tokenizer([[dummy_prompt, dummy_candidates[0]], [dummy_prompt, dummy_candidates[1]],
                    [dummy_prompt, dummy_candidates[2]], [dummy_prompt, dummy_candidates[3]]], 
                   return_tensors="pt", padding=True)

# Convert tokenized inputs to match the expected input format for the model
example = {
    "input_ids": inputs["input_ids"].unsqueeze(0), 
    "attention_mask": inputs["attention_mask"].unsqueeze(0), 
    "token_type_ids": inputs["token_type_ids"].unsqueeze(0)  
}

# Convert to ONNX format
onnx_model_path = "bert_mc_model.onnx"
torch.onnx.export(
    model,
    (example["input_ids"], example["attention_mask"], example["token_type_ids"]),
    onnx_model_path,
    opset_version=14,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "num_choices", 2: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "num_choices", 2: "sequence_length"},
        "token_type_ids": {0: "batch_size", 1: "num_choices", 2: "sequence_length"},
        "logits": {0: "batch_size", 1: "num_choices"}
    }
)

# Verify the model's structure
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
