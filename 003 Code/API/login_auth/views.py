from django.shortcuts import render

# Create your views here.
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import json

from rest_framework import generics

@csrf_exempt
def signup(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        print(data)
        user_id = data.get('user_id')
        user_pw = data.get('user_pw')
        user_email = data.get('user_email')

        # Perform any necessary validation here

        # Create and save the user instance
        user = User.objects.create_user(username=user_id, password=user_pw, email=user_email)

        return JsonResponse({'message': 'Signup successful'})

    return JsonResponse({'error': 'Invalid request method'})



@csrf_exempt
def login_view(request):
    print(request)
    if request.method == 'POST':
        data = json.loads(request.body)
        user_id = data.get('user_id')
        user_pw = data.get('user_pw')
        print(data)
        # Check if user exists
        user = User.objects.filter(username=user_id).first()
        print(user)
        if not user:
            return JsonResponse({'error': 'User does not exist'}, status=401)

        # Authenticate and login user
        authenticated_user = authenticate(request, username=user_id, password=user_pw)
        print(authenticated_user)
        if authenticated_user is None:
            return JsonResponse({'error': 'Invalid credentials'}, status=401)

        login(request, authenticated_user)
        return JsonResponse({'message': 'Login successful'})

    return JsonResponse({'error': 'Invalid request method'}, status=400)
