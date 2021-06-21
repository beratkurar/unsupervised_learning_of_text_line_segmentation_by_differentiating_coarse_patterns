# Author: Berat Kurar
# Date: 01/01/2021

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np
import textwrap
import random


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
        elif y>page_height-margin-max_word_height:
            page_img.save('printed_dataset/page'+str(page_number)+'.png')
            page_number=page_number+1
            x=margin
            y=margin
            page_img=Image.new(mode='L', size=(page_width, page_height), color='white')
            page_draw = ImageDraw.Draw(page_img)
            word_draw = ImageDraw.Draw(Image.new(mode='L', size=(100,100), color='white'))
        else:
            x=margin
            y=y+max_word_height

text_press(text, fonts, page_width, page_height)

