import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("aipicasso/emi-3", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    "anime style, 1girl, looking at viewer, serene expression, gentle smile, multicolored hair, rainbow gradient hair, wavy long hair, heterochromia, purple left eye, blue right eye, pastel color scheme, magical girl aesthetic, white text overlay \"ANIMINS\", centered text, modern typography, ethereal lighting, soft glow, fantasy atmosphere, rainbow gradient background, dreamy atmosphere, sparkles, light particles, magical effects, depth of field, bokeh effect",
    num_inference_steps=40,
    guidance_scale=4.5,
).images[0]
image.save("emi3.png")
