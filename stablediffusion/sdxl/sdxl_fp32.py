from diffusers import DiffusionPipeline
import torch

# Load the Stable Diffusion pipeline in full precision (fp32)
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe.to("cuda")  # Move to GPU for faster performance

# Generate the image
prompt = "An astronaut riding a green horse"
images = pipe(prompt=prompt).images[0]

# Save the image
image_save_path = "image_fp32.png"
images.save(image_save_path)