# Author: Berat Kurar
# Date: 8/11/2020
# Visualizing Deep Similarity Networks
# Highlights which parts of image are most important for the classification. Scaled from red to blue where red is most important.

from tqdm import tqdm
import random
import numpy as np
# matlib must be imported separately
from numpy import matlib as mb
import matplotlib.pyplot as plt
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

def blend(patch,patch_size, heatmap):
    cmap = plt.get_cmap('jet')
    heatmap = heatmap - np.min(heatmap)
    heatmap /= np.max(heatmap)
    heatmap = cmap(np.max(heatmap)-heatmap)
    if np.max(heatmap) < 255.:
        heatmap *= 255
    #remove alpha channel
    heatmap = heatmap[:,:,:3]
    heatmap = heatmap.astype('uint8')
    #convert to bgr
    heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR) 
    upsampled_heatmap = cv2.resize(heatmap_bgr,(patch_size, patch_size), cv2.INTER_CUBIC)   
    blended_upsampled_heatmap = cv2.addWeighted(upsampled_heatmap, 0.9,cv2.merge((patch,patch,patch)), 0.1, 0)
    return blended_upsampled_heatmap

def spatial_similarity(conv1,conv2):
    pool1 = np.mean(conv1,axis=0)
    pool2 = np.mean(conv2,axis=0)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
    conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
    im_similarity = np.zeros((conv1_normed.shape[0],conv1_normed.shape[0]))
    for zz in range(conv1_normed.shape[0]):
        repPx = mb.repmat(conv1_normed[zz,:],conv1_normed.shape[0],1)
        im_similarity[zz,:] = np.multiply(repPx,conv2_normed).sum(axis=1)
    similarity1 = np.reshape(np.sum(im_similarity,axis=1),out_sz)
    similarity2 = np.reshape(np.sum(im_similarity,axis=0),out_sz)
    return similarity1, similarity2

def pair_similarity(patch_size, patches_set, pair_similarity_results_folder):
    model=load_model('bestmodel')
    branch_model = model.layers[2]
    in_layer=branch_model.inputs
    #out_layer=branch_model.get_layer('act_5').output
    out_layer=branch_model.get_layer('conv_5').output
    vis_model=Model(inputs=in_layer, outputs=out_layer)

    for imgp in tqdm(os.listdir(patches_set)):
        s=imgp.split('_')
        n=s[0]
        k=s[1]
        r=s[2][1]
        if r=='1': 
            patch1 = cv2.imread('{}/{}'.format(patches_set, imgp), 0)
            patch2 = cv2.imread('{}/{}'.format(patches_set, n+'_'+k+'_p2.png'), 0)
            #patch1 = process(patch1)
            #patch2 = process(patch2)
            batch_patch1 = np.expand_dims(patch1, axis=0)
            batch_patch1 = np.expand_dims(batch_patch1, axis=3)
            batch_patch2 = np.expand_dims(patch2, axis=0)
            batch_patch2 = np.expand_dims(batch_patch2, axis=3)
            conv1 = vis_model.predict(batch_patch1)[0]
            conv2 = vis_model.predict(batch_patch2)[0]

            heatmap1, heatmap2 = spatial_similarity(conv1.reshape(-1,conv1.shape[-1]),
                                                    conv2.reshape(-1,conv2.shape[-1]))
            blended1 = blend(patch1,patch_size,heatmap1)
            blended2 = blend(patch2,patch_size,heatmap2)   
            concat = np.concatenate((blended1, blended2), axis=1)          
            cv2.imwrite(pair_similarity_results_folder+'/'+ imgp,concat)
     
    del model
    del vis_model
