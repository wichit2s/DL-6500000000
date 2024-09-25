import json

from channels.generic.websocket import WebsocketConsumer


class ChatBotConsumer(WebsocketConsumer):

    def connect(self):
        self.accept()
        print('ImageBotConsumer: accept ws connection')

    def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)
        prompt = data['prompt']
        print(f'user input prompt: {prompt}')

    def disconnect(self, code):
        print('ImageBotConsumer: ws disconnect')