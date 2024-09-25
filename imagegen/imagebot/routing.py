from django.urls import re_path

from imagebot import consumers

websocket_urlpatterns = [
    re_path('ws/image', consumers.ImageBotConsumer.as_asgi()),
]