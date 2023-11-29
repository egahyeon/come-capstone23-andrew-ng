"""
URL configuration for django_backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('login_auth/', include('login_auth.urls')),  # Include URLs from the 'api' app
    path('pig_info/', include('pig_info.urls')),  # Include URLs from the 'db_save' app    
]


#localhost:8000/login_auth/login/
#localhost:8000/login_auth/signup/

#localhost:8000/pig_info/gpu_server/
#localhost:8000/pig_info/flutter_load/