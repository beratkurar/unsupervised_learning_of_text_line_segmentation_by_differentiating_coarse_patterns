# Author: Berat Kurar
# Date: 01/01/2021

import math as m
from PIL import Image

Ro = 600.0
Ri = 520.0

cir = [[0 for x in range(int(Ro * 2))] for y in range(int(Ro * 2))]

image = Image.open('h_3600.png')
pixels = image.load()
width, height = image.size

def morph_img(img):
    list_image = [item for sublist in img for item in sublist]
    new_image = Image.new("L", (len(img[0]), len(img)))
    new_image.putdata(list_image)
    new_image.save("circled_text_image.png","PNG")

for i in range(int(Ro)):
    # outer_radius = Ro*m.cos(m.asin(i/Ro))
    outer_radius = m.sqrt(Ro*Ro - i*i)
    for j in range(-int(outer_radius),int(outer_radius)):
        if i < Ri:
            # inner_radius = Ri*m.cos(m.asin(i/Ri))
            inner_radius = m.sqrt(Ri*Ri - i*i)
        else:
            inner_radius = -1
        if j < -inner_radius or j > inner_radius:
            x = Ro+j
            y = Ro-i
            angle = m.atan2(y-Ro,x-Ro)/2
            distance = m.sqrt((y-Ro)*(y-Ro) + (x-Ro)*(x-Ro))
            distance = m.floor((distance-Ri+1)*(height-1)/(Ro-Ri))
            cir[int(y)][int(x)] = pixels[int(width*angle/m.pi) % width, height-distance-1]
            y = Ro+i
            angle = m.atan2(y-Ro,x-Ro)/2
            distance = m.sqrt((y-Ro)*(y-Ro) + (x-Ro)*(x-Ro))
            distance = m.floor((distance-Ri+1)*(height-1)/(Ro-Ri))
            cir[int(y)][int(x)] = pixels[int(width*angle/m.pi) % width, height-distance-1]

morph_img(cir)