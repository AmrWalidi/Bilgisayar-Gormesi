from ultralytics import YOLO
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

model = YOLO('yolov8n.pt')

profile_url = pd.read_csv('data/facebook profile.csv')

raw_html = [requests.get(x).text for x in profile_url['website_url']]

parsed_html = [BeautifulSoup(x, 'html.parser') for x in raw_html]

image_url_list = [x.find('meta', {'property': 'og:image'}).attrs['content'] for x in parsed_html]

for i, image in enumerate(image_url_list):
    response = requests.get(image)
    with open(f'images/image{i + 1}.jpg', 'wb') as file:
        file.write(response.content)

images = os.listdir('images')

for i, image in enumerate(images):
    result = model(f'images/{image}', save=True)
