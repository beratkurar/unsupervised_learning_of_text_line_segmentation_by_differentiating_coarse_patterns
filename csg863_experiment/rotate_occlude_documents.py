# Author: Berat Kurar
# Date: 01/01/2021

import cv2
import os
import numpy as np

def rotate_image(img, angle):
    height, width = img.shape[:2] 
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])
    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    # rotate image with the new bounds and translated rotation matrix
    rotated_img=cv2.warpAffine(img, rotation_mat, (bound_w, bound_h), borderMode=cv2.BORDER_REPLICATE)
    return rotated_img

def occlude_image(img, size):
    margin=5
    rows,cols=img.shape
    number_of_occlusions=int(((rows*cols)/(size*size))*0.1)
    for i in range(number_of_occlusions):
        offset=[np.random.randint(margin, rows-size-margin),np.random.randint(margin, cols-size-margin)]
        img[offset[0]:offset[0]+size,offset[1]:offset[1]+size]=255
    return img

dataset_name='ahte'
output_folder='../datasets/'+dataset_name+'_dataset/'+dataset_name+'_special'
os.makedirs(output_folder,exist_ok=True)

lines_0=cv2.imread('lines_rotated_0.png',0)
document_0=cv2.imread('document_rotated_0.png',0)
#rotate
angles=[0,45,90,135,180,225,270,315]
for angle in angles:
    rotated_lines=rotate_image(lines_0,angle)
    cv2.imwrite(output_folder+'/lines_rotated_'+str(angle)+'.png',rotated_lines)
    rotated_document=rotate_image(document_0,angle)
    cv2.imwrite(output_folder+'/document_rotated_'+str(angle)+'.png',rotated_document)

#occlude
sizes=[40,60,80,100]
for size in sizes:
    occluded_document=occlude_image(document_0,size)
    cv2.imwrite(output_folder+'/document_occluded_'+str(size)+'.png',occluded_document)