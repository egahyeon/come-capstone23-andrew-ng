from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import JsonDataViewset, get_pig_info, main_page_request, graph_page_request

router = DefaultRouter()
router.register('pig_data', JsonDataViewset)

urlpatterns = [
    path('gpu_server/', include(router.urls)),
    path('flutter_load/', get_pig_info, name='flutter_load'),
    path('main_page/', main_page_request, name='main_page_load'),
    path('graph_page/', graph_page_request, name='graph_page_load'),
]
