import json
from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync

class ChatGPTConsumer(WebsocketConsumer):

    def connect(self):
        self.accept()
        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': "สื่อสารไปกลับได้แล้วครับ"
        }))

    def receive(self, text_data=None, bytes_data=None):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        print('Message from client: ', message)
        #
        #self.send(text_data=json.dumps({
        #    'type': 'chat',
        #    'message': message
        #}))

    def disconnect(self, code):
        pass
