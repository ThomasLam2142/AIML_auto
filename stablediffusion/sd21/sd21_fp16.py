import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Load the model in fp16 mode
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Move the model to GPU
pipe = pipe.to("cuda")

# Set parameters for image generation
prompt = "a photo of an astronaut riding a horse on mars"
negative_prompt = "blurry, low quality, text"  # Optional: avoid certain things
guidance_scale = 8.5  # Control how strictly the image follows the prompt
num_inference_steps = 75  # More steps lead to higher quality images
generator = torch.manual_seed(42)  # Set seed for reproducibility

# Generate the image
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    generator=generator
).images[0]

# Save the image to a file
image.save("image_sd21_fp16.png")