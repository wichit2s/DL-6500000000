from django.core.management import BaseCommand

from diffusers import DiffusionPipeline
from PIL import Image
import torch


class Command(BaseCommand):
    def call_back(self, pipe, step_index, timestamp, kwargs):
        print(f'{step_index} {timestamp}')
        #print('pipe: ', pipe)
        #print('step_index: ', step_index)
        #print('timestamp: ', timestamp)
        #print('kwargs')
        #for k,v in kwargs.items():
        #    print('\t', k, v)
        return kwargs

    def handle(self, *args, **options):
        print('references: https://github.com/huggingface/diffusers ')

        # model_name = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
        model_name = 'CompVis/stable-diffusion-v1-4'
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        pipe.to('mps')
        images = pipe('Motorscooter Vespa on street in Los Angeles with palm trees', callback_on_step_end=self.call_back).images
        for i in range(len(images)):
            images[i].save(f'result-{i+1}.png')

