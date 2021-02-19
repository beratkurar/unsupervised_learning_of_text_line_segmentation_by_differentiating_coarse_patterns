# Author: Ahmad Droby and Berat Kurar
# Date: 8/11/2020
# TraffickCam: Explainable Image Matching For Sex Trafficking Investigations
# Matching parts of the two images assigned to the same color.

from tqdm import tqdm
import numpy as np
import cv2
import os
import sys
from tensorflow.keras.models import load_model, Model
from sklearn.decomposition import PCA
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def process(image):
    fimage=image.astype('float32')
    nimage=fimage/255.
    return nimage

def normalize_array(arr):
    arr = 255.0 * (arr - arr.min()) / (arr.max() - arr.min())
    return arr.astype(np.uint8)

def patch_saliency(patch_size, patches_set, patch_saliency_results_folder):
    model=load_model('bestmodel')
    branch_model = model.layers[2]
    in_layer=branch_model.inputs
    #out_layer=branch_model.get_layer('act_5').output
    out_layer=branch_model.get_layer('conv_5').output
    vis_model=Model(inputs=in_layer, outputs=out_layer)
    
    for imgp in tqdm(os.listdir(patches_set)):
        patch = cv2.imread('{}/{}'.format(patches_set, imgp), 0)
        #patch=process(patch)
        batch_patch = np.expand_dims(patch, axis=0)
        batch_patch = np.expand_dims(batch_patch, axis=3)
        activation_maps = vis_model.predict(batch_patch)[0]
      
        pca = PCA(n_components=3)
        features = activation_maps.reshape(-1, activation_maps.shape[2])
        pca_t_features = pca.fit(features).transform(features)
        for i in range(pca_t_features.shape[1]):
            pca_t_features [:,i] = normalize_array(pca_t_features[:, i])
        rgb = np.asarray(pca_t_features, dtype=np.uint8).reshape((*activation_maps.shape[:2], 3))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) 
        
        upsampled_cubic=cv2.resize(bgr, (patch_size,patch_size), cv2.INTER_CUBIC)
        blended_upsampled_cubic=cv2.addWeighted(upsampled_cubic, 0.9, cv2.merge((patch, patch, patch)), 0.1, 0)
        cv2.imwrite(patch_saliency_results_folder+'/'+imgp, blended_upsampled_cubic)
    
    del model
    del vis_model
