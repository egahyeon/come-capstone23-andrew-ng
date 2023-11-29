from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.http import JsonResponse
from django.db.models import Max, Q

import datetime

from .models import pig_info
from .serializers import PigInfoSerializer, MainPageSerializer, GraphPageSerializer

from rest_framework.permissions import AllowAny

class JsonDataViewset(viewsets.ModelViewSet):
    queryset = pig_info.objects.all()
    serializer_class = PigInfoSerializer
    permission_classes = [AllowAny]  # 모든 사용자에게 접근 허용

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        
        pno, now_time = serializer.data['pNo'], serializer.data['now']
        json_item = {
            'pNo': pno, 
            'now': now_time,
            'act': serializer.data['act'], 
            'pred': serializer.data['pred']
        }
        
        now_time = datetime.datetime.strptime(now_time, '%Y-%m-%dT%H:%M:%S%z')
        
        latest_pig_info = pig_info.objects.filter(pNo=pno).order_by('-now')[7]
        if latest_pig_info and (now_time - latest_pig_info.now) > datetime.timedelta(days=1):
             print(f'Detele {pno}')
             pig_info.objects.filter(pNo=pno).exclude(now=now_time).delete()
        else:
             print(f'Not detele {pno}')
             print(now_time - latest_pig_info.now)
             print(now_time, latest_pig_info.now)

        return Response(serializer.data, status=status.HTTP_201_CREATED)

@api_view(['GET'])
def get_pig_info(request):
    if request.method == 'GET':
        # pig_info 모델의 모든 레코드를 가져옵니다.
        queryset = pig_info.objects.all()
        
        # 시리얼라이저를 사용하여 데이터를 직렬화합니다.
        serializer = PigInfoSerializer(queryset, many=True)
        print(serializer)
        # 직렬화된 데이터를 JSON 형식으로 반환합니다.
        return Response(serializer.data)

# Main Page GET
@api_view(['GET'])
def main_page_request(request):
    if request.method == 'GET':
        print('Main page request receive')
        # print(request)
        sending_list = []
        
        # start_of_day = datetime.datetime(2023, 11, 9)
        # end_of_day = datetime.datetime(2023, 11, 9, 23, 59, 59, 999999)
        # pig_info.objects.filter(now__range=(start_of_day, end_of_day)).delete()
        
        # pig_info 모델로부터 pNo의 고유한 값들을 가져옵니다.
        unique_pnos = pig_info.objects.values_list('pNo', flat=True).distinct()
        latest_entries = pig_info.objects.filter(pNo__in=unique_pnos, pred=True).annotate(latest_now=Max('now'))

        for entry in latest_entries:
            sending_list.append({
                'pNo': entry.pNo,
                'now': entry.latest_now,
                'pred': True
            })
        
        # Handle pNos without entries
        for pno in unique_pnos:
            if pno not in [entry['pNo'] for entry in sending_list]:
                sending_list.append({
                    'pNo': pno,
                    'now': None,
                    'pred': False
                })

        print('Sending Data: %s', sending_list)
        serializer = MainPageSerializer(data=sending_list, many=True)
        print(serializer.is_valid(raise_exception=True))
        return Response(serializer.data)

# Graph Page GET
@api_view(['GET'])
def graph_page_request(request):
    if request.method == 'GET':
        print('Graph Page request receive')
        print(request)
        request_pno = request.query_params.get('pNo')
        print(f'input_data {request_pno}')
        
        graph_data = pig_info.objects.filter(pNo=request_pno)
        
        print('Graph page Data')
        print(graph_data)
        main_serializer = GraphPageSerializer(graph_data, many=True)
        print('Sending Data')
        print(main_serializer.data)
        
        print('Graph page Sending')
        return Response(main_serializer.data)
