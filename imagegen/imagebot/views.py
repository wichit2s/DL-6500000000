from django.shortcuts import render

# Create your views here.
def index(request):
    context = {
        'step': 5,
        'steps': list(range(1,6))
    }
    return render(request, 'imagebot/index.html', context)