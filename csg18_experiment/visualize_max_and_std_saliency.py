# Author: Berat Kurar
# Date: 8/11/2020
# https://www.kaggle.com/zfturbo/visualisation-of-siamese-net
# This visualization assigns the pixels to pink as often as they have the max. Means that this pixel is important in all the activation maps.
# Assigns the pixels green as big as the standard deviation. Means that this pixel is considered.
# Assigns the pixels white as often as they have the max and as big as the standard deviation. Means that this pixel is important only in one activation map.

from tqdm import tqdm
import random
import numpy as np
import cv2
import os
import sys
from tensorflow.keras.models import load_model, Model
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def process(image):
    fimage=image.astype('float32')
    nimage=fimage/255.
    return nimage

def normalize_array(arr):
    arr = 255.0 * (arr - arr.min()) / (arr.max() - arr.min())
    return arr.astype(np.uint8)

def heatmap(vis_model, patch, patch_size):   
    #patch=process(patch)
    batch_patch = np.expand_dims(patch, axis=0)
    batch_patch = np.expand_dims(batch_patch, axis=3)
    activation_maps = vis_model.predict(batch_patch)[0]
    
    # Uncomment it to emulate RELU activation
    #activation_maps[activation_maps < 0] = 0

    ch0 = np.zeros_like(activation_maps[:, :, 0])
    ch1 = np.zeros_like(activation_maps[:, :, 0])
    ch2 = np.zeros_like(activation_maps[:, :, 0])

    # Find how often maximum is in each pixel.
    for k in range(activation_maps.shape[2]):
        p = activation_maps[:, :, k]
        mx = p.max()
        if mx == 0:
            continue
        for i in range(activation_maps.shape[0]):
            for j in range(activation_maps.shape[1]):
                if p[i, j] == mx:
                    ch0[i, j] += 1
                    ch2[i, j] += 1

    for i in range(activation_maps.shape[0]):
        for j in range(activation_maps.shape[1]):
            mn = activation_maps[i, j].min()
            mx = activation_maps[i, j].max()
            mean = activation_maps[i, j].mean()
            std = activation_maps[i, j].std()
            #ch1[i, j] = std
            #ch1[i, j] = mean
            ch1[i, j] = mx

    ch0 = normalize_array(ch0)
    ch1 = normalize_array(ch1)
    ch2 = normalize_array(ch2)
    ch = np.stack((ch2, ch1, ch0), axis=2)

    return ch

def max_and_std_saliency(patch_size, patches_set, max_and_std_saliency_results_folder):
    model=load_model('bestmodel')
    branch_model = model.layers[2]
    in_layer=branch_model.inputs
    #out_layer=branch_model.get_layer('act_5').output
    out_layer=branch_model.get_layer('conv_5').output
    vis_model=Model(inputs=in_layer, outputs=out_layer)

    for imgp in tqdm(os.listdir(patches_set)):
        patch = cv2.imread('{}/{}'.format(patches_set, imgp), 0)
        saliency_map = heatmap(vis_model, patch, patch_size)
        bgr = saliency_map.astype(np.uint8)
        
        upsampled_cubic=cv2.resize(bgr, (patch_size,patch_size), cv2.INTER_CUBIC)
        blended_upsampled_cubic=cv2.addWeighted(upsampled_cubic, 0.9, cv2.merge((patch, patch, patch)), 0.1, 0)
        cv2.imwrite(max_and_std_saliency_results_folder+'/'+imgp, blended_upsampled_cubic)
        
    del model
    del vis_model
