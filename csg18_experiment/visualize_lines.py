# Author: Ahmad Droby and Berat Kurar
# Date: 8/11/2020

from tqdm import tqdm
import numpy as np
import cv2
import os
import sys
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def process(image):
    fimage = image.astype('float32')
    nimage = fimage/255.
    return nimage

def normalize_array(arr):
    arr = 255.0 * (arr - arr.min()) / (arr.max() - arr.min())
    return arr.astype(np.uint8)

def pca(outer_size, inner_size, test_set, results_folder_name):
    trim_size = int((outer_size-inner_size)//2)
    model = load_model('bestmodel')
    vis_model = model.layers[2]
    #vis_model = Model(inputs = vis_model.input, outputs = vis_model.get_layer('conv_4').output)

    os.makedirs(os.path.join(results_folder_name, 'upsampled_nearest'), exist_ok=True)
    os.makedirs(os.path.join(results_folder_name, 'upsampled_cubic'), exist_ok=True)
    os.makedirs(os.path.join(results_folder_name, 'blended_upsampled_nearest'), exist_ok=True)
    os.makedirs(os.path.join(results_folder_name, 'blended_upsampled_cubic'), exist_ok=True)

    for imgp in os.listdir(test_set): 
        page = cv2.imread('{}/{}'.format(test_set, imgp), 0)
        print(imgp)
        rows,cols = page.shape
        x = rows//inner_size
        y = cols//inner_size
        prows = (x+1)*inner_size+2*trim_size
        pcols = (y+1)*inner_size+2*trim_size
        #ppage=np.zeros([prows,pcols])
        ppage = np.ones([prows,pcols])*255
        ppage[trim_size:rows+trim_size,trim_size:cols+trim_size] = page[:,:]
        predicted_patch = vis_model.predict(np.zeros((1, outer_size, outer_size, 1)))
        predicted_img = np.zeros((x+1,y+1, predicted_patch.shape[1]), np.float32)

        for i in tqdm(range(0,x+1)):
            for j in range(0,y+1):
                patch = ppage[i*inner_size:i*inner_size+outer_size,
                            j*inner_size:j*inner_size+outer_size]
                #patch=process(patch)
                patch = np.expand_dims(patch, axis=0)
                patch = np.expand_dims(patch, axis=3)
                predicted_patch = vis_model.predict(patch)[0]
                predicted_img[i, j, :] = predicted_patch
        
        pca = PCA(n_components=3)
        features = predicted_img.reshape(-1, predicted_img.shape[2])
        pca_t_features = pca.fit(features).transform(features)
        
        for i in range(pca_t_features.shape[1]):
            pca_t_features [:,i] = normalize_array(pca_t_features[:, i])
        rgb = np.asarray(pca_t_features, dtype=np.uint8).reshape((*predicted_img.shape[:2], 3))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) 
        
        #Manually upsample nearest
        bgr_rows,bgr_cols,_= bgr.shape
        upsampled_nearest = np.zeros([rows,cols,3])
        for i in range(bgr_rows):
            for j in range(bgr_cols):
                pixel_value = bgr[i,j,:]
                upsampled_nearest[i*inner_size:i*inner_size+inner_size,
                                  j*inner_size:j*inner_size+inner_size,:] = pixel_value

        upsampled_nearest = np.asarray(upsampled_nearest, dtype = np.uint8)   
        blended_upsampled_nearest = cv2.addWeighted(upsampled_nearest, 0.9, cv2.merge((page, page, page)), 0.1, 0)
        
        upsampled_cubic = cv2.resize(bgr, (cols, rows), cv2.INTER_CUBIC)
        blended_upsampled_cubic = cv2.addWeighted(upsampled_cubic, 0.9, cv2.merge((page, page, page)), 0.1, 0)

        cv2.imwrite('{}/{}'.format(os.path.join(results_folder_name, 'upsampled_nearest'), imgp), 
                                                upsampled_nearest)
        cv2.imwrite('{}/{}'.format(os.path.join(results_folder_name, 'blended_upsampled_nearest'),imgp),
                                                blended_upsampled_nearest)
        cv2.imwrite('{}/{}'.format(os.path.join(results_folder_name, 'upsampled_cubic'), imgp), 
                                                upsampled_cubic)    
        cv2.imwrite('{}/{}'.format(os.path.join(results_folder_name, 'blended_upsampled_cubic'), imgp),
                                                blended_upsampled_cubic)
        
        #break    
    del model
    del vis_model
