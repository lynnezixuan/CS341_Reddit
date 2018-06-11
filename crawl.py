import re
import os 
import requests
import math
from bs4 import BeautifulSoup
import pandas as pd
from io import BytesIO
from PIL import Image
df = pd.read_csv('/home/shared/CS341/Dataprocessing/imagelinks_submissions_2016_07.csv', header = None,usecols=[2,6])
pic_urls = df[6]
file_name = df[2]

# pic_urls = pic_urls[108130:]
# file_name=file_name[108130:]
names = ['nba','golf','dogpictures','guns','fishing','minecraft','pokemon','xray','streetwear','cats',\
         'overwatch','watches','starwars','lego','bicycling','pizza','beards']
# i = 108130
i = 0
count=0
for each in pic_urls:
    if file_name[i].lower() not in names:
        i+=1
        print (i)
        continue
    if i == len(pic_urls):
        break
    if each!=each:
        i+=1
        print (i)
        continue
    if each.endswith('.jpg') or each.endswith('png'):
        url = each
    else:
        try:
            html = requests.get(each).text
        except:
                print ('connection error')
                i+=1
                print (i)
                continue
        soup = BeautifulSoup(html,'lxml')
        img_ul = soup.find_all('link', {"rel": "image_src"})
        if len(img_ul)==0:
            i+=1
            print (i)
            continue
        url = img_ul[0]['href']
    try:
        pic= requests.get(url, timeout=100)
    except:
        print ('no image')
        i+=1
        print (i)
        continue
    try:
        im = Image.open(BytesIO(pic.content))    
        if im.size !=(130,60) or im.size!=(161,81):
            string = file_name[i].lower()+'-'+str(i) + '.jpg'
            if not os.path.exists('./newimages/'+file_name[i].lower()+'/'):
                os.makedirs('./newimages/'+file_name[i].lower()+'/')
            fp = open('./newimages/'+file_name[i].lower()+'/'+string,'wb')
            fp.write(pic.content)
            fp.close()
            i += 1
            count+=1
            print ('count:'+str(count))
            print (i)
        else:
            i+=1
            print (i)
    except IOError:
            print ('cannot access')
            i += 1
            print (i)
            continue