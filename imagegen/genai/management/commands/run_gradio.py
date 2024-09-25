from django.core.management import BaseCommand

from diffusers import DiffusionPipeline
#from PIL import Image
import torch
# import gradio



class Command(BaseCommand):
    def handle(self, *args, **options):
        print('references: https://github.com/huggingface/diffusers ')

        model_name = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
        pipe = DiffusionPipeline.from_pretrained(model_name) #, torch_dtype=torch.float16)
        # gradio.Interface.from_pipeline(pipe).launch()

