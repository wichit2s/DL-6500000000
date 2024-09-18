from django.core.management import BaseCommand

from diffusers import DiffusionPipeline
from PIL import Image
import torch


class Command(BaseCommand):
    def handle(self, *args, **options):
        print('references: https://github.com/huggingface/diffusers ')

        model_name = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
        pipeline = DiffusionPipeline.from_pretrained(model_name) #, torch_dtype=torch.float16)
        pipeline.to('cpu')
        image_tensor = pipeline('An image of a cute squirrel in cartoon style').images[0]
        image = (image_tensor / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image*255).round().astype("uint8"))
        image.show()
