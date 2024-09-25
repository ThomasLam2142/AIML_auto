import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Load the model in fp32 mode (by omitting the torch_dtype argument)
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Move the model to GPU
pipe = pipe.to("cuda")

# Set parameters for image generation
prompt = "a photo of an astronaut riding a horse on mars"
negative_prompt = "blurry, low quality, text"  # Optional: avoid these elements
guidance_scale = 8.5  # Control prompt adherence (higher = more adherence)
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
image.save("image_sd21_fp32.png")