import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer

# Quantize model (dynamic)
quantize_dynamic(
    model_input="./distilbert_qa_model.onnx",
    model_output="./distilbert_qa_model_int8.onnx",
    weight_type=QuantType.QUInt8,
)

print("ONNX model has been successfully quantized to INT8.")

# Validate quantized model

session = ort.InferenceSession("./distilbert_qa_model_int8.onnx")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

dummy_input_ids = np.random.randint(0, tokenizer.vocab_size, (1, 384)).astype(np.int64)
dummy_attention_mask = np.ones((1, 384)).astype(np.int64)

inputs = {
    "input_ids": dummy_input_ids,
    "attention_mask": dummy_attention_mask
}

outputs = session.run(None, inputs)

print("Inference with quantized model successful.")
#print("Outputs:", outputs)