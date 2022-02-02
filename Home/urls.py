from django.contrib import admin
from django.urls import path
from Home import views

urlpatterns = [
    path('', views.index, name="home"),
    path('crop_form', views.crop_form, name='crop_form'),
    path('crop_form_result', views.crop_form_result, name='crop_form_result'),
    path('fertilizer_form', views.fertilizer_form, name='fertilizer_form'),
    path('index', views.index, name="home")
]