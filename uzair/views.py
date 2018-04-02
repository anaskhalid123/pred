from django.shortcuts import render
from rest_framework.decorators import api_view
from .serializers import testSerializer
from .models import Tests
from rest_framework.response import Response
from rest_framework import status
from freelancetask.post_project_v1 import run1
@api_view(['GET','POST'])
def ap(request):
    if request.method=='POST':
        print(request.data)
        # r=Test.objects.all()
        # d=testSerializer(r,many=True)

        try:
            d=run1(request.data)
        except Exception as e:
            print(e)




        return Response({'status': True, 'message': d}, status=status.HTTP_202_ACCEPTED)
    return Response({'status':'True','message':''},status=status.HTTP_200_OK)
    # if request.method == 'POST':
    #     r=testSerializer(data=request.data)
    #     r.is_valid(raise_exception=True)
    #     r.save()
    #
    #     return Response({'status': True, 'message': r.data}, status=status.HTTP_202_ACCEPTED)
    # else:
    #     return Response('lund')
#