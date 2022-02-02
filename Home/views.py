from os import stat
from django.shortcuts import render
import pickle
import numpy as np

# Loading crop recommendation model

crop_recommendation_model_path = './models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

# Create your views here.
def index(request):
    return render(request, 'index.html')

def crop_form(request):
    return render(request, 'crop_form.html')

def crop_form_result(request):
    if request.method == "POST":
        N = request.POST.get('N')
        P = request.POST.get('P')
        K = request.POST.get('K')
        ph_level = request.POST.get('ph_level')
        rainfall = request.POST.get('rainfall')
        state = request.POST.get('state')
        city = request.POST.get('city')
        print("HElo")
        print(N, P, K, ph_level, rainfall, state)
    temperature, humidity = 21, 82
    data = np.array([[N, P, K, temperature, humidity, ph_level, rainfall]])
    my_prediction = crop_recommendation_model.predict(data)
    final_prediction = my_prediction[0]
    print(final_prediction)
    return render(request, 'crop_form.html')

def fertilizer_form(request):
    return render(request, 'fertilizer_form.html')


