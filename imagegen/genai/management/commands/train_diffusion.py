from django.core.management import BaseCommand

class Command(BaseCommand):

    def handle(self, *args, **options):
        print('diffusion from scratch')
        print('https://medium.com/@mickael.boillaud/denoising-diffusion-model-from-scratch-using-pytorch-658805d293b4')
        print('https://github.com/pesser/pytorch_diffusion/blob/master/pytorch_diffusion/diffusion.py')