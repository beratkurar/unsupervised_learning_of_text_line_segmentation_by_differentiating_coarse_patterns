# Author: Ahmad Droby and Berat Kurar
# Date: 8/11/2020

import numpy as np
import os
import cv2

c=0
grid = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
#grid = [(0, 1),   (0, -1)]

def pad_img(img, patch_size):
    rows, cols = img.shape
    padded_img = np.ones([rows+2*patch_size, cols+2*patch_size], dtype=np.uint8) * 255
    padded_img[patch_size:rows+patch_size , patch_size:cols+patch_size] = img 
    return padded_img

def valid_binary_patch(patch, alpha=0.03):
    # dark to black, light to white.
    ret, binary_patch = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return np.sum(np.sum(patch != 255)) > alpha*patch.shape[0]*patch.shape[1]

def valid_inv_binary_patch(patch, alpha=0.1):
    # light to black, dark to white.
    ret, binary_patch = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return np.sum(np.sum(patch != 0)) > alpha*patch.shape[0]*patch.shape[1]

def block(patch, fill=255, alpha=0.3):
    blocked_patch = patch.copy()
    block_size = int(patch.shape[0] * alpha)
    blocked_patch[(blocked_patch.shape[0] - block_size) // 2:(blocked_patch.shape[0] + block_size) // 2,
                  (blocked_patch.shape[1] - block_size) // 2:(blocked_patch.shape[1] + block_size) // 2] = fill  
    return blocked_patch

def h_flip(p):
    return cv2.flip(p, 1)

def v_flip(p):
    return cv2.flip(p, 0)

def rotate_180(p):
    return cv2.rotate(p, cv2.ROTATE_180)

def rotate_90(p):
    return cv2.rotate(p, cv2.ROTATE_90_COUNTERCLOCKWISE)

def identity(p):
    return p

def get_neighbouring_patches(img, patch_size):
    margin = 3
    gap = 2
    pos = [np.random.randint(margin + patch_size, img.shape[0] - 2*patch_size - margin),
           np.random.randint(margin + patch_size, img.shape[1] - 2*patch_size - margin)]
    p1_pos = pos
    i, j = grid[np.random.randint(0, len(grid))]
    #p2_pos = [pos[0] + np.random.randint(-gap, gap) + i * patch_size, pos[1] + np.random.randint(-gap, gap) + j *
              #patch_size]
    p2_pos = [pos[0] + i * patch_size, pos[1] + j * patch_size]
    p1 = img[p1_pos[0]:p1_pos[0] + patch_size, p1_pos[1]:p1_pos[1] + patch_size]
    p2 = img[p2_pos[0]:p2_pos[0] + patch_size, p2_pos[1]:p2_pos[1] + patch_size]
    return p1, p2

def get_same_patch(img, patch_size):
    margin = 3
    gap = 2
    pos = [np.random.randint(margin +  patch_size, img.shape[0] -  patch_size - margin),
           np.random.randint(margin + patch_size, img.shape[1] -  patch_size - margin)]
    p1_pos = pos
    #p2_pos = [pos[0] + np.random.randint(-gap, gap), pos[1] + np.random.randint(-gap, gap)]
    p1 = img[p1_pos[0]:p1_pos[0] + patch_size, p1_pos[1]:p1_pos[1] + patch_size]
    #p2 = img[p2_pos[0]:p2_pos[0] + patch_size, p2_pos[1]:p2_pos[1] + patch_size]
    p2 = p1.copy()  
    return p1, p2

def get_similar_patches(dataset_name, images_path, patch_size, SHOW_RESULTS, 
                        same_patch_assump, same_patch_aug):
    global c
    label=0
    images = os.listdir(images_path)
    i_p = np.random.choice(images)
    img = cv2.imread(os.path.join(images_path, i_p), 0)
    img = pad_img(img, patch_size)
    #img=255-img
    skip = 0
    while True:
        gen_func = np.random.choice(same_patch_assump)
        p1, p2 = gen_func(img, patch_size)
        if valid_binary_patch(p1) and valid_binary_patch(p2):
            break
        if skip == 15:
            images = os.listdir(images_path)
            i_p = np.random.choice(images)
            img = cv2.imread(os.path.join(images_path, i_p), 0)
            img = pad_img(img, patch_size)
            skip = 0
        skip = skip + 1
    aug_func = np.random.choice(same_patch_aug)
    #p2 = block(aug_func(p2))
    p2 = aug_func(p2)
    if SHOW_RESULTS:
        cv2.imwrite('../datasets/'+dataset_name+'_dataset/'+dataset_name+
                    '_sample_similar_pairs/'+str(c)+'_same_p1.png', p1)
        cv2.imwrite('../datasets/'+dataset_name+'_dataset/'+dataset_name+
                    '_sample_similar_pairs/'+str(c)+'_same_p2.png', p2) 
        c=c+1
    return p1, p2, label

def get_diff_patches(dataset_name, images_path, patch_size, SHOW_RESULTS, 
                     diff_patch_assump, diff_patch_aug):
    global c
    label=1
    images = os.listdir(images_path)
    i_p = np.random.choice(images)
    img = cv2.imread(os.path.join(images_path, i_p), 0)
    img = pad_img(img, patch_size)
    #img=255-img
    skip = 0
    while True:
        gen_func = np.random.choice(diff_patch_assump)
        p1, p2 = gen_func(img, patch_size)
        if valid_binary_patch(p1) and valid_binary_patch(p2):
            break
        if skip == 15:
            images = os.listdir(images_path)
            i_p = np.random.choice(images)
            img = cv2.imread(os.path.join(images_path, i_p), 0)
            img = pad_img(img, patch_size)
            skip = 0
        skip = skip + 1
    aug_func = np.random.choice(diff_patch_aug)
    #p2 = block(aug_func(rotate_90(p2)))
    p2 = aug_func(rotate_90(p2))
    if SHOW_RESULTS:
        cv2.imwrite('../datasets/'+dataset_name+'_dataset/'+dataset_name+
                    '_sample_different_pairs/'+str(c)+'_different_p1.png', p1)
        cv2.imwrite('../datasets/'+dataset_name+'_dataset/'+dataset_name+
                    '_sample_different_pairs/'+str(c)+'_different_p2.png', p2) 
        c=c+1
    return p1, p2, label

def get_random_pair(dataset_name, images_path, patch_size, SHOW_RESULTS, 
                    same_patch_assump, diff_patch_assump, same_patch_aug, diff_patch_aug):
    funcs = [get_similar_patches, get_diff_patches]
    args = [[dataset_name, images_path, patch_size, SHOW_RESULTS, same_patch_assump, same_patch_aug], 
            [dataset_name, images_path, patch_size, SHOW_RESULTS, diff_patch_assump, diff_patch_aug]]
    choice = np.random.randint(len(funcs))
    p1, p2, label = funcs[choice](*args[choice])
    return p1, p2, label

def avr_img_size(images_path):
    images = os.listdir(images_path)
    number_of_documents = len(images)
    total_rows=0
    total_cols=0
    for img in images:
        img = cv2.imread(os.path.join(images_path, img), 0)
        rows, cols = img.shape
        total_rows=total_rows+rows
        total_cols=total_cols+cols
    avr_rows=total_rows//number_of_documents
    avr_cols=total_cols//number_of_documents
    return avr_rows, avr_cols

def number_of_patches(images_path, patch_size, avr_rows, avr_cols):
    images = os.listdir(images_path)
    number_of_documents = len(images)              
    number_hor_patches = int(avr_cols/(patch_size))
    number_ver_patches = int(avr_rows/(patch_size))
    number_per_document = number_hor_patches * number_ver_patches
    number_of_patches = number_per_document * number_of_documents    
    return number_of_patches                 
