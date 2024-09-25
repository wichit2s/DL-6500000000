import base64
import json
import time
from io import BytesIO

import torch
from channels.generic.websocket import WebsocketConsumer
from diffusers import StableDiffusionPipeline
from django.template.loader import get_template


class ImageBotConsumer(WebsocketConsumer):

    def connect(self):
        self.accept()
        print('ImageBotConsumer: accept ws connection')

    def update(self, step_idx, t, latents):
        step = step_idx//5 + 1
        print(f' update step {step}')
        context = {
            'step': step,
            'steps': list(range(1, 7))
        }
        html = get_template('imagebot/update.html').render(context)
        self.send(text_data=html)

    def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)
        prompt = data['prompt']
        print(f'user input prompt: {prompt}')
        pipe = StableDiffusionPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4',
            #variant='fp16',
            #torch_dtype=torch.float16
        )
        pipe.to('cpu')
        # schedule = DDPMSScheduler()
        image = pipe(
            prompt,
            callback=self.update,
            callback_steps=5,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        img_str = base64.b64encode(buffered.getvalue())
        image_result_html = get_template('imagebot/result.html').render({
            'base64_image_str': img_str.decode('utf-8')
        })
        self.send(text_data=image_result_html)
        '''
        for i in range(1,8):
            print(f' update step {i}')
            context = {
                'step': i,
                'steps': list(range(1,7))
            }
            #self.send(text_data=json.dumps({
            #    'step': i,
            #    'steps': list(range(1,7))
            #}))
            html = get_template('imagebot/update.html').render(context)
            self.send(text_data=html)
            time.sleep(2)
        '''
    def disconnect(self, code):
        print('ImageBotConsumer: ws disconnect')