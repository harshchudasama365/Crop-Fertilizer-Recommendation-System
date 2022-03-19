from os import stat
from urllib import response
from django.shortcuts import render
import pickle
import numpy as np
import pandas as pd
from .utils import fertilizer_dict, CNN_Model,disease_dic
from django.shortcuts import redirect
import torch
from torchvision import transforms
from PIL import Image
import io
from markupsafe import Markup
from django.templatetags.static import static
from django.utils.safestring import mark_safe
import requests
from datetime import date

disease_classes = ['Apple___Apple_scab',
'Apple___Black_rot',
'Apple___Cedar_apple_rust',
'Apple___healthy',
'Cherry_(including_sour)___Powdery_mildew',
'Cherry_(including_sour)___healthy',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
'Corn_(maize)___Common_rust_',
'Corn_(maize)___Northern_Leaf_Blight',
'Corn_(maize)___healthy',
'Grape___Black_rot',
'Grape___Esca_(Black_Measles)',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
'Grape___healthy',
'Peach___Bacterial_spot',
'Peach___healthy',
'Pepper,_bell___Bacterial_spot',
'Pepper,_bell___healthy',
'Potato___Early_blight',
'Potato___Late_blight',
'Potato___healthy',
'Strawberry___Leaf_scorch',
'Strawberry___healthy',
'Tomato___Bacterial_spot',
'Tomato___Early_blight',
'Tomato___Late_blight',
'Tomato___Leaf_Mold',
'Tomato___Septoria_leaf_spot',
'Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___healthy']
# Loading crop recommendation model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
disease_model_path = './models/plant_disease_model.pth'
disease_model = CNN_Model()
# disease_model.load_state_dict(torch.load(disease_model_path), map_location=torch.device('cpu'))
disease_model.load_state_dict(torch.load(disease_model_path))
# disease_model = disease_model.to(device)
disease_model.eval()


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


crop_recommendation_model_path = './models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

# Create your views here.
def index(request):
    
    return render(request, 'index.html')

def crop_form(request):
    
    # print(rainfall_value)
    return render(request, 'crop_form.html')

def crop_form_result(request):
    if request.method == "POST":
        N = request.POST.get('N')
        P = request.POST.get('P')
        K = request.POST.get('K')
        ph_level = request.POST.get('ph_level')
        # rainfall = request.POST.get('rainfall')
        # state = request.POST.get('state')
        city = request.POST.get('city')
        print("HElo"+ city )
        print(type(city))
        print(N, P, K, ph_level)
    df2 = pd.read_csv('.//Data/rainfall.csv')
    df2['rainfall'] = df2['rainfall'].astype(float)
    rainfall_value = df2[(df2['District'] == city.strip())]['rainfall'].mean()
    # rainfall_value = df2[(df2['District'] == city.strip()) & (df2['Month'] == str(date.today().month)) ]['rainfall'].iloc[0]
    # rainfall_value = "5.6"
    print(type(date.today().month))
    # temperature, humidity = weatherInfo(city)
    temperature, humidity = 22, 60
    print(temperature, humidity)
    data = np.array([[N, P, K, temperature, humidity, ph_level, rainfall_value]])

    print( "rainfall data is here for "+ city + "is "+ str(rainfall_value))
    my_prediction = crop_recommendation_model.predict(data)
    final_crop_prediction = my_prediction[0]
    pred='images/crop/'+final_crop_prediction+'.jpg'
    url = static(pred)
    img = mark_safe('<img src="{url}" width="780px" height="450px" alt={{prediction}}>'.format(url=url))
    final_prediction = { 'prediction' : final_crop_prediction, 'pred':pred,'img':img}

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

    response1 = Markup(str(fertilizer_dict[key1]))
    response2 = Markup(str(fertilizer_dict[key2]))
    response3 = Markup(str(fertilizer_dict[key3]))
    response_dict = {'response1':response1,'response2':response2,'response3':response3,'diff_n':abs_n, 'diff_p':abs_p,'diff_k': abs_k}
    return render(request, 'fertilizer_form_result.html', response_dict)
    # return render(request, 'fertilizer_form.html')

def plant_disease_form(request):
    return render(request, 'plant_disease_form.html')


def predict_image(img, model=disease_model):
    transform = transforms.ToTensor()
    transform = transforms.Compose([
        transforms.Resize([256,256]),
        # transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    yb = model(img_u)
    _, preds = torch.max(yb , dim=1)
    print(preds[0].item())
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction


def plant_disease_form_result(request):
    if 'file' not in request.FILES:
        return redirect("/")
    # prediction_dict = {}
    file = request.FILES["file"] 
    if not file:
        return render(request,'plant_disease_form.html')
    try:
        img = file.read()
        prediction = predict_image(img)
        prediction = Markup(str(disease_dic[prediction]))
        prediction_dict = {'prediction':prediction}
        # print(prediction_dict)
    except:
        pass
    
    return render(request, 'plant_disease_form_result.html', prediction_dict)

def contact(request):
    return render(request, 'contact.html')


def weatherInfo(city):
    api_key = "1f06ecea104a74bc101e05bd85c9bc81"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    return 0,0
    
