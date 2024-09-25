from diffusers import DiffusionPipeline
import torch

# Load the Stable Diffusion pipeline
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# Generate the image
prompt = "An astronaut riding a green horse"
images = pipe(prompt=prompt).images[0]

# Save the image
image_save_path = "image_fp16.png"
images.save(image_save_path)