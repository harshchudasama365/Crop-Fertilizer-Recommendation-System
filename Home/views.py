from os import stat
from urllib import response
from django.shortcuts import render
import pickle
import numpy as np
import pandas as pd
from .utils import fertilizer_dict
from markdown import markdown

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
    final_prediction = { 'prediction' : my_prediction[0]}
    # print(final_prediction.prediction)
    return render(request, 'crop_form_result.html', final_prediction)

def fertilizer_form(request):
    return render(request, 'fertilizer_form.html')

def fertilizer_form_result(request):
    if request.method == "POST":
        crop = request.POST.get('crop')
        N_filled = request.POST.get('N')
        P_filled = request.POST.get('P')
        K_filled = request.POST.get('K')
    df = pd.read_csv('.//Data/Crop_NPK.csv')
    print(df)
    # print(crop,N,P,K)

    N_desired = df[df['Crop'] == crop]['N'].iloc[0]
    P_desired = df[df['Crop'] == crop]['P'].iloc[0]
    K_desired = df[df['Crop'] == crop]['K'].iloc[0]

    print(N_desired, P_desired,K_desired)
    n = N_desired-int(N_filled)
    p = P_desired - int(P_filled)
    k = K_desired - int(K_filled)
    # print(n,p,k)
    if n < 0:
        key1 = "NHigh"
    elif n > 0:
        key1 = "Nlow"
    else:
        key1 = "NNo"

    if p < 0:
        key2 = "PHigh"
    elif p > 0:
        key2 = "Plow"
    else:
        key2 = "PNo"

    if k < 0:
        key3 = "KHigh"
    elif k > 0:
        key3 = "Klow"
    else:
        key3 = "KNo"

    abs_n = abs(n)
    abs_p = abs(p)
    abs_k = abs(k)

    response1 = markdown(str(fertilizer_dict[key1]))
    response2 = markdown(str(fertilizer_dict[key2]))
    response3 = markdown(str(fertilizer_dict[key3]))
    response_dict = {'response1':response1,'response2':response2,'response3':response3,'diff_n':abs_n, 'diff_p':abs_p,'diff_k': abs_k}
    return render(request, 'fertilizer_form_result.html', response_dict)
    # return render(request, 'fertilizer_form.html')


