from django.core.management.base import BaseCommand
from diffusers import DiffusionPipeline
import torch

class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        print('Generating image from diffuser models.')
        model_id = "runwayml/stable-diffusion-v1-5"
        pipeline = DiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
        prompt = "portrait photo of a old warrior chief"
        generator = torch.Generator("cuda").manual_seed(0)
        image = pipeline(prompt, generator=generator).images[0]
        

