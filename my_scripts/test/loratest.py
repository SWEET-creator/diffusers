from diffusers import DiffusionPipeline, AutoPipelineForText2Image, StableDiffusionPipeline
import torch
import os

lora_dir = "/root/code/diffusers/output/stable-diffusion-v1-5-lora_alleta_3_pad"
output_dir = os.path.join(lora_dir, "image/")
os.makedirs(output_dir, exist_ok=True)

pipe_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
# pipe_id = "Disty0/LCM_SoteMix"
pipe = StableDiffusionPipeline.from_pretrained(pipe_id,
                                               safety_checker=None,
                                               torch_dtype=torch.float16).to("cuda")

pipe.load_lora_weights(lora_dir)

prompt = "alleta, full body, front, white background"

lora_scale = 0.5
image = pipe(
    prompt, num_inference_steps=30, guidance_scale=8.0,
    cross_attention_kwargs={"scale": lora_scale}, generator=torch.manual_seed(0)
).images[0]

image.save(f"{output_dir}output.png")