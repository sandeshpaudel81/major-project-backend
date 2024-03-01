from rest_framework.response import Response 
from rest_framework.views import APIView
from rest_framework import status
from django.shortcuts import render
from .pred import get_front_pred, get_back_pred
import base64
from django.core.files.base import ContentFile
from .postProcessing import postpro
import re

# Create your views here.
class PredictionView(APIView): 
    def post(self, request):
        # front_image = request.POST.get('front_image')
        # back_image = request.POST.get('back_image')
        data = request.data
        
        frontExt, frontimgstr = data['front_image'].split(';base64,')
        front_image_data = base64.b64decode(frontimgstr)
        frontImage=ContentFile(front_image_data, name=f'frontImage.{frontExt}')

        backExt, backimgstr = data['back_image'].split(';base64,')
        back_image_data = base64.b64decode(backimgstr)
        backImage=ContentFile(back_image_data, name=f'backImage.{backExt}')

        # # image = ImageModel.objects.create(front=front_image, back=back_image)
        front_res = get_front_pred(frontImage)
        back_res = get_back_pred(backImage)
        if('error' in front_res.keys()):
            return Response(front_res, status=status.HTTP_200_OK)
        elif('error' in back_res.keys()):
            return Response(back_res, status=status.HTTP_200_OK)
        else:
            front_res.update(back_res)
            json_data = {}
            for key, value in front_res.items():
                valueData = postpro(key, value)
                val = valueData.encode('unicode_escape').decode('utf-8')
                pattern = r'\\x..'
                output_string = re.sub(pattern, '', val)
                json_data[key] = output_string
            res_data = {'data': json_data}
            return Response(res_data, status=status.HTTP_200_OK)
    

class WebPredictionView(APIView): 
    def post(self,request):
        front_image = request.FILES.get('frontImage')
        back_image = request.FILES.get('backImage')
        # image = ImageModel.objects.create(front=front_image, back=back_image)
        front_res = get_front_pred(front_image)
        back_res = get_back_pred(back_image)
        if('error' in front_res.keys()):
            return Response(front_res, status=status.HTTP_400_BAD_REQUEST)
        elif('error' in back_res.keys()):
            return Response(back_res, status=status.HTTP_400_BAD_REQUEST)
        else:
            front_res.update(back_res)
            json_data = {}
            for key, value in front_res.items():
                valueData = postpro(key, value)
                val = valueData.encode('unicode_escape').decode('utf-8')
                pattern = r'\\x..'
                output_string = re.sub(pattern, '', val)
                json_data[key] = output_string
            res_data = {'data': json_data}
            return Response(res_data, status=status.HTTP_200_OK)
    

def home(request):
    line = "abcdef-ghi"

    # Split the line by colon or hyphen
    colonSplit = line.split(':')[0]
    print(colonSplit) 
    return render(request, 'home.html')