from rest_framework.response import Response 
from rest_framework.views import APIView
from rest_framework import status
from django.shortcuts import render
from .pred import get_front_pred, get_back_pred

# Create your views here.
class PredictionView(APIView): 
    def post(self,request):
        front_image = request.FILES.get('frontImage')
        back_image = request.FILES.get('backImage')
        # image = ImageModel.objects.create(front=front_image, back=back_image)
        front_res = get_front_pred(front_image)
        back_res = get_back_pred(back_image)
        data = {
            'front': front_res,
            'back': back_res
        }
        return Response(data, status=status.HTTP_200_OK)
    

def home(request):
    return render(request, 'home.html')