import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/taccuser/llama3_converted")
model = AutoModelForCausalLM.from_pretrained("/home/taccuser/llama3_converted")

# Function to generate a response from the model
def generate_response(prompt, max_length=100, num_return_sequences=1):
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate a response from the model
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,      # Enables sampling instead of greedy decoding
            top_k=50,            # Top-k sampling
            top_p=0.95,          # Top-p (nucleus) sampling
            temperature=0.7,     # Controls randomness of predictions
        )

    # Decode and return the generated text
    responses = [tokenizer.decode(out, skip_special_tokens=True) for out in output]
    return responses

# Example usage
prompt = "Once upon a time"
responses = generate_response(prompt)
for i, response in enumerate(responses):
    print(f"Response {i+1}: {response}")
