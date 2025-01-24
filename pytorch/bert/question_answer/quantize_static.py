import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

# Load a subset of the SQuAD dataset for representative data
squad = load_dataset("squad", split="train[:500]")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Preprocess function to prepare the dataset
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_tensors="np",
        padding="max_length",
    )
    return inputs

# Apply the preprocessing function
tokenized_squad = squad.map(preprocess_function, batched=True)

# Create a CalibrationDataReader class
class SquadCalibrationDataReader(CalibrationDataReader):
    def __init__(self, tokenized_data):
        self.inputs = tokenized_data
        self.current_index = 0

    def get_next(self):
        if self.current_index < len(self.inputs["input_ids"]):
            input_ids = self.inputs["input_ids"][self.current_index : self.current_index + 1]
            attention_mask = self.inputs["attention_mask"][self.current_index : self.current_index + 1]
            self.current_index += 1
            return {
                "input_ids": np.array(input_ids, dtype=np.int64),
                "attention_mask": np.array(attention_mask, dtype=np.int64),
            }
        return None

# Initialize the CalibrationDataReader
calibration_reader = SquadCalibrationDataReader(tokenized_squad)

# Paths to the FP32 and INT8 models
fp32_model_path = "distilbert_qa_model.onnx"  # Path to your FP32 ONNX model
int8_model_path = "distilbert_qa_model_int8.onnx"  # Path to save the INT8 ONNX model

# Perform static quantization with calibration
quantize_static(
    model_input=fp32_model_path,
    model_output=int8_model_path,
    calibration_data_reader=calibration_reader,
    quant_format=QuantType.QInt8
)

print(f"Quantized model saved at: {int8_model_path}")