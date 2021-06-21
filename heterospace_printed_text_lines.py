# Author: Berat Kurar
# Date: 01/01/2021

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import shutil
import os
import numpy as np
import textwrap
import random

dataset_name='heterospace_dataset'
shutil.rmtree(dataset_name, ignore_errors=True)
os.mkdir(dataset_name)

file = open('text.txt',mode='r')
text= file.read()
file.close()

page_height=2500
page_width=1500

fonts= ['DejaVuSans-Bold.ttf',
        'DejaVuSans-BoldOblique.ttf',
        'DejaVuSans-ExtraLight.ttf',
        'DejaVuSans-Oblique.ttf',
        'DejaVuSans.ttf',
        'DejaVuSansCondensed-Bold.ttf',
        'DejaVuSansCondensed-BoldOblique.ttf',
        'DejaVuSansCondensed-Oblique.ttf',
        'DejaVuSansCondensed.ttf',
        'DejaVuSansMono-Bold.ttf',
        'DejaVuSansMono-BoldOblique.ttf',
        'DejaVuSansMono-Oblique.ttf',
        'DejaVuSansMono.ttf']

font_sizes=np.random.randint(15,50,30)

def text_press(text, fonts, page_width, page_height):
    margin=100
    max_word_width=300
    max_word_height=60
    space_heights=[60, 80, 100, 120, 140]
    
    
    page_img=Image.new(mode='L', size=(page_width, page_height), color='white')
    page_draw = ImageDraw.Draw(page_img)
    word_draw = ImageDraw.Draw(Image.new(mode='L', size=(100,100), color='white'))
    
    x=margin
    y=margin
    page_number=0
    for word in text.split(' '):
        word_font_name=random.choice(fonts)
        word_font_size=random.choice(font_sizes)
        word_font=ImageFont.truetype(word_font_name, word_font_size)
        word_width, word_height = word_draw.textsize(text=word+' ', font=word_font)
        mid_y=int(y-word_height//2)
        page_draw.text((x, mid_y), text=word, font=word_font)       
        if x<page_width-margin-max_word_width:
            x = x + word_width
        #new page
        elif y>page_height-margin-max_word_height:
            page_img.save(dataset_name+'/page'+str(page_number)+'.png')
            page_number=page_number+1
            x=margin
            y=margin
            page_img=Image.new(mode='L', size=(page_width, page_height), color='white')
            page_draw = ImageDraw.Draw(page_img)
            word_draw = ImageDraw.Draw(Image.new(mode='L', size=(100,100), color='white'))
        #new line
        else:
            x=margin
            space_height=random.choice(space_heights)
            y=y+space_height
    return page_number

page_number=text_press(text, fonts, page_width, page_height)
dataset_raw_name=dataset_name[:-8]
os.mkdir(dataset_name+'/'+dataset_raw_name+'_test')
os.mkdir(dataset_name+'/'+dataset_raw_name+'_train')
os.mkdir(dataset_name+'/'+dataset_raw_name+'_sample_different_pairs')
os.mkdir(dataset_name+'/'+dataset_raw_name+'_sample_similar_pairs')

for i in range(page_number):
    if i<2:
        shutil.copy(dataset_name+'/page'+str(i)+'.png',
                    dataset_name+'/'+dataset_raw_name+'_test/page'+str(i)+'.png')
        shutil.move(dataset_name+'/page'+str(i)+'.png',
                    dataset_name+'/'+dataset_raw_name+'_train/page'+str(i)+'.png')
    else:
        shutil.move(dataset_name+'/page'+str(i)+'.png',
                    dataset_name+'/'+dataset_raw_name+'_train/page'+str(i)+'.png')
