from diffusers import StableDiffusionXLPipeline 
import torch
import os

lora_dir = "/root/code/diffusers/output/lora_alleta_3_pad/checkpoint-1000"
output_dir = os.path.join(lora_dir, "image/")
os.makedirs(output_dir, exist_ok=True)

pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
# pipe_id = "Linaqruf/animagine-xl"
pipe = StableDiffusionXLPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")

# pipe_id = "/root/code/diffusers/checkpoints/Pony_Diffusion_V6_XL.safetensors"
# pipe = StableDiffusionXLPipeline.from_single_file(pipe_id, torch_dtype=torch.float16, use_safetensors=True).to("cuda")

pipe.load_lora_weights(lora_dir)

prompt = "alleta, white background"

lora_scale = 0.9
image = pipe(
    prompt, num_inference_steps=30, guidance_scale=8.0,
    cross_attention_kwargs={"scale": lora_scale}, generator=torch.manual_seed(0),
).images[0]

image.save(f"{output_dir}output.png")
print("Saved to", f"{output_dir}output.png")