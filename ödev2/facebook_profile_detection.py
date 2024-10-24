from ultralytics import YOLO
import pandas as pd
from PIL import Image
from io import BytesIO
import requests
from bs4 import BeautifulSoup

#YOLO modeli tanımlanır
model = YOLO('yolov8n.pt')

#Facebook profil sayfaları dosyadan getirilir
profile_url = pd.read_csv('data/facebook profile.csv')

#Ham HTML kodu getirilir
raw_html = [requests.get(x).text for x in profile_url['website_url']]

#Ham HTML koduları ayrıştırılır
parsed_html = [BeautifulSoup(x, 'html.parser') for x in raw_html]

# og:image değerine sahip property özelliğine sahip meta element etiketlerinin content özelliğinin değerinden çıkarılan görüntü
image_url_list = [x.find('meta', {'property': 'og:image'}).attrs['content'] for x in parsed_html]

#Her görüntü okunup detect işlemi yapılır
for i, image in enumerate(image_url_list):
    response = requests.get(image)
    if 'image' in response.headers['Content-Type']:

        # URL görüntüye dönüştürüyor
        img = Image.open(BytesIO(response.content))

        result = model(img)
        detections = result[0].boxes

        # En yüksek güvenirlilik puanına sahip kutu seçilir
        highest_conf_box = max(detections, key=lambda box: box.conf)
        class_idx = int(highest_conf_box.cls)
        class_name = model.names[class_idx]

        # Veri setiye eklenir
        profile_url.loc[i, 'detection'] = 'insan' if class_name == 'person' else 'başka varlık'

# Data setiyi yazdırılır
print(profile_url)

