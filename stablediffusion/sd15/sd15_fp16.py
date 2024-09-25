from diffusers import StableDiffusionPipeline
import torch

# Load the model in fp16 mode (half-precision)
pipe = StableDiffusionPipeline.from_pretrained(
    "benjamin-paine/stable-diffusion-v1-5", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Prompt and other parameters
prompt = "a photo of an astronaut riding a horse on mars"
negative_prompt = "blurry, low quality, text"
guidance_scale = 8.5
num_inference_steps = 75
generator = torch.manual_seed(42)

# Generate the image
image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    generator=generator
).images[0]

# Save the final image
image.save("image_sd15_fp16.png")