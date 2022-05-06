from os import stat
from unittest import result
from urllib import response
from cv2 import recoverPose
from django.shortcuts import render
import pickle
import numpy as np
import pandas as pd
from .utils import fertilizer_dict, CNN_Model,disease_dic, ResNet9
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
from requests_html import HTMLSession, AsyncHTMLSession
import requests, time, aiohttp,asyncio
import nest_asyncio
from django.http import JsonResponse
import concurrent.futures
from .models import *
from django.contrib import messages
import datetime

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
disease_model_path = './models/plant_disease_model1.pth'
disease_model = CNN_Model()
disease_model = ResNet9(3, 38)
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
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        service = request.POST.get('service')
        mobile_no = request.POST.get('mobile_no')      
        message = request.POST.get('message')
        contact = Contact(name=name, email=email, mobile_no=mobile_no, service=service,
                        message=message, date=datetime.date.today())
        contact.save()
        messages.success(request, 'Your Form Has Been Submitted.')
    
    
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


async def fetch(url):
    nest_asyncio.apply()
    session = AsyncHTMLSession()
    

    r = await session.get(url)


    r.html.render(scrolldown=5, timeout=20)

    articles = r.html.find('article')


    global newslist
    newslist = []


    for item in articles:
        try:
            newsitem = item.find('h3', first=True)
            # newsimg =  item.find('img')
            title = newsitem.text
            link = newsitem.absolute_links
            # img = newsimg.link

            newsarticle = {
                'title': title,
                'link': next(iter(link)) ,
                # 'img':img
            }
            newslist.append(newsarticle)
        except:
            pass

    r.close()
    session.close()
    print(newslist)
    return newslist
    # response_news = {'new_list' : newslist}
    # # response_news = {'news_list':[{'title': 'How Helena Agri-Enterprises is ready to grow', 'link': 'https://www.agriculture.com/news/business/how-helena-chemical-is-ready-to-grow'}, {'title': 'Manure And Cover Crops', 'link': 'https://www.agriculture.com/podcast/successful-farming-radio-podcast/manure-and-cover-crops'}, {'title': 'The future of weed management may be seed prevention technologies', 'link': 'https://www.agriculture.com/crops/crop-protection/the-future-of-weed-management-may-be-seed-prevention-technologies'}, {'title': '3 soil fertility strategies for 2022', 'link': 'https://www.agriculture.com/video/3-soil-fertility-strategies-for-2022'}, {'title': '5 strategies to overcome weed control issues', 'link': 'https://www.agriculture.com/video/5-strategies-to-overcome-weed-control-issues'}, {'title': 'CFAD releases new white paper', 'link': 'https://www.agriculture.com/crops/conservation/agree-cfad-releases-new-white-paper'}, {'title': 'Study shows consumers have limited understanding of the carbon farming market', 'link': 'https://www.agriculture.com/crops/study-shows-consumers-have-limited-understanding-of-the-carbon-farming-market'}, {'title': 'Soil Carbon Initiative launches new farm certification pilots', 'link': 'https://www.agriculture.com/crops/soil-health/soil-carbon-initiative-launches-new-farm-certification-pilots'}, {'title': 'USDA to invest $250 million to support American-made fertilizer', 'link': 'https://www.agriculture.com/crops/fertilizers/usda-to-invest-250-million-to-support-american-made-fertilizer'}]}
    # return render(request, 'news.html', response_news)


# async def news(request):
    

    

#     url='https://www.agriculture.com/crops'
#     async with aiohttp.ClientSession() as client:
#         tasks = []
#         task =  asyncio.ensure_future(fetch(url))
#         tasks.append(task)
#         results = await asyncio.gather(task)
#         response_news = {'new_list' : results}
    
#     # response_news = {'new_list' : response_news}
#     # # response_news = {'news_list':[{'title': 'How Helena Agri-Enterprises is ready to grow', 'link': 'https://www.agriculture.com/news/business/how-helena-chemical-is-ready-to-grow'}, {'title': 'Manure And Cover Crops', 'link': 'https://www.agriculture.com/podcast/successful-farming-radio-podcast/manure-and-cover-crops'}, {'title': 'The future of weed management may be seed prevention technologies', 'link': 'https://www.agriculture.com/crops/crop-protection/the-future-of-weed-management-may-be-seed-prevention-technologies'}, {'title': '3 soil fertility strategies for 2022', 'link': 'https://www.agriculture.com/video/3-soil-fertility-strategies-for-2022'}, {'title': '5 strategies to overcome weed control issues', 'link': 'https://www.agriculture.com/video/5-strategies-to-overcome-weed-control-issues'}, {'title': 'CFAD releases new white paper', 'link': 'https://www.agriculture.com/crops/conservation/agree-cfad-releases-new-white-paper'}, {'title': 'Study shows consumers have limited understanding of the carbon farming market', 'link': 'https://www.agriculture.com/crops/study-shows-consumers-have-limited-understanding-of-the-carbon-farming-market'}, {'title': 'Soil Carbon Initiative launches new farm certification pilots', 'link': 'https://www.agriculture.com/crops/soil-health/soil-carbon-initiative-launches-new-farm-certification-pilots'}, {'title': 'USDA to invest $250 million to support American-made fertilizer', 'link': 'https://www.agriculture.com/crops/fertilizers/usda-to-invest-250-million-to-support-american-made-fertilizer'}]}

#     return render(request, 'news.html', response_news)

        


# with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
#     url='https://www.agriculture.com/crops'
#     executor.submit(fetch, url)
        
def news(request):
    url = "https://newsapi.org/v2/everything?q=agriculture&from=2022-03-23&sortBy=popularity&apiKey=1bb563a6dc5e42458129668d861c58cd"
    
    agri_news = requests.get(url).json()
    a = agri_news['articles']
    desc=[]
    title=[]
    img=[]
    lst = []
    calender = {'01':'Jan','02':'Feb','03':'March','04':'April','05':'May',
    '06':'June','07':'July','08':'August','09':'Sept','10':'Oct','11':'Nov','12':'Dec'}
    for i in range(len(a)):
        f = a[i]
        temp_dict = dict()
        temp_dict['title'] = f['title']
        temp_dict['description'] = f['description']
        temp_dict['img'] = f['urlToImage']
        temp_dict['link'] = f['url']
        data = f['publishedAt']
        month = calender[data[5:7]]
        day = data[8:10]
        temp_dict['date'] = day+" "+month
        lst.append(temp_dict)
    # news_list = zip(title, desc, img)
    context = {'news_list':lst}
    print(lst)
    
    return render(request, 'news.html', context)