# Author: Berat Kurar
# Date: 01/01/2021

from tqdm import tqdm
import numpy as np
import cv2
import os
import shutil
import random
from random import shuffle
from sklearn.metrics import accuracy_score as accuracy
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten, Dropout, concatenate, BatchNormalization, Conv2D, Activation, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model, Sequential
from generate_patches import get_same_patch, get_neighbouring_patches, h_flip, v_flip, rotate_180, identity, rotate_90, get_random_pair
import generate_patches
import visualize_lines
import visualize_patch_saliency
import visualize_max_and_std_saliency
import visualize_pair_similarity
os.environ["CUDA_VISIBLE_DEVICES"]="0"

exp_name = '_exp'
dataset_name = 'csg18'
train_set = '../datasets/'+dataset_name+'_dataset/'+dataset_name+'_train'
test_set = '../datasets/'+dataset_name+'_dataset/'+dataset_name+'_test'
special_set = '../datasets/'+dataset_name+'_dataset/'+dataset_name+'_special'
similar_pairs_set = '../datasets/'+dataset_name+'_dataset/'+dataset_name+'_sample_similar_pairs'
different_pairs_set = '../datasets/'+dataset_name+'_dataset/'+dataset_name+'_sample_different_pairs'

same_patch_assump = [get_same_patch, get_neighbouring_patches]
diff_patch_assump = [get_same_patch, get_neighbouring_patches]
same_patch_aug = [h_flip, rotate_180]
diff_patch_aug =[v_flip, identity]

patch_size = 350
input_shape = (patch_size,patch_size,1)
inner_size = 20
avr_rows, avr_cols = generate_patches.avr_img_size(test_set)
train_set_size = generate_patches.number_of_patches(train_set, patch_size, avr_rows, avr_cols)
test_set_size = generate_patches.number_of_patches(test_set, patch_size, avr_rows, avr_cols)
sample_set_size = 90

#These values are to make a quick run for sanity check
# inner_size=100
# train_set_size=30
# test_set_size=30

learning_rate = 0.00001
epochs = 100
patient = 7
batch_size = 8
version = '1'
continue_from_best=False
continue_from_version = ''
branch = 3 #0-alex, 1-vgg, 2-xception 3-custom

#Save experimental setting
with open('parameters.txt', 'w') as f:
    print('dataset_name:', dataset_name, file=f)
    print('avr_rows:', avr_rows, file=f)
    print('avr_cols:', avr_cols, file=f)
    print('same_patch_assump:', same_patch_assump, file=f)
    print('diff_patch_assump:', diff_patch_assump, file=f)
    print('same_patch_aug:', same_patch_aug, file=f)
    print('diff_patch_aug:', diff_patch_aug, file=f)
    print('patch_size:', patch_size, file=f)
    print('inner_size:', inner_size, file=f)
    print('train_set_size:', train_set_size, file=f)
    print('test_set_size:', test_set_size, file=f)
    print('sample_set_size:', sample_set_size, file=f)
    print('learning_rate:', learning_rate, file=f)
    print('epochs:', epochs, file=f)
    print('patient:', patient, file=f)
    print('batch_size:', batch_size, file=f)
    print('branch_model 0-alex, 1-vgg, 2-xception, 3-custom:', branch, file=f)

def process(image):
    fimage=image.astype('float32')
    nimage=fimage/255.
    return nimage

class DataGenerator(Sequence):
    def __init__(self, epoch_size, batch_size, dataset_name, patch_size, folder_name, set_size, 
                 same_patch_assump, diff_patch_assump, same_patch_aug, diff_patch_aug):
        self.set_size = set_size
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.dataset_name = dataset_name
        self.patch_size = patch_size
        self.folder_name = folder_name
        self.same_patch_assump = same_patch_assump
        self.diff_patch_assump = diff_patch_assump
        self.same_patch_aug = same_patch_aug
        self.diff_patch_aug = diff_patch_aug

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.set_size / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        pairs = []
        labels = []
        for i in range(self.batch_size):
            p1, p2, label = generate_patches.get_random_pair(self.dataset_name, self.folder_name, self.patch_size, 
                                                             False,
                                                             self.same_patch_assump, self.diff_patch_assump, 
                                                             self.same_patch_aug, self.diff_patch_aug)
            #p1=process(p1)
            #p2=process(p2)
            p1=p1.reshape(p1.shape[0],p1.shape[1],1)
            p2=p2.reshape(p2.shape[0],p2.shape[1],1)
            pairs += [[p1, p2]]
            labels += [label]
        apairs = np.array(pairs)
        alabels = np.array(labels)
        return [apairs[:, 0], apairs[:, 1]], alabels

def unsupervised_loaddata(dataset_name, patch_size, folder_name, set_size, SHOW_RESULTS):
    pairs = []
    labels = []
    for i in tqdm(range(set_size)):
        p1, p2, label = generate_patches.get_random_pair(dataset_name, folder_name, patch_size,  
                                                         SHOW_RESULTS, same_patch_assump, diff_patch_assump, 
                                                         same_patch_aug, diff_patch_aug)
        #p1=process(p1)
        #p2=process(p2)
        p1=p1.reshape(p1.shape[0],p1.shape[1],1)
        p2=p2.reshape(p2.shape[0],p2.shape[1],1)
        pairs += [[p1, p2]]
        labels += [label]
    apairs=np.array(pairs)
    print(apairs.shape)
    alabels=np.array(labels)
    return apairs,alabels

#Generate sample pairs
shutil.rmtree('../datasets/'+dataset_name+'_dataset/'+dataset_name+'_sample_similar_pairs')
shutil.rmtree('../datasets/'+dataset_name+'_dataset/'+dataset_name+'_sample_different_pairs')
os.mkdir('../datasets/'+dataset_name+'_dataset/'+dataset_name+'_sample_similar_pairs')
os.mkdir('../datasets/'+dataset_name+'_dataset/'+dataset_name+'_sample_different_pairs')
unsupervised_loaddata(dataset_name, patch_size, test_set, sample_set_size, True)

#Generate train and test pairs batch by batch
training_generator = DataGenerator(epochs, batch_size, dataset_name, patch_size, train_set, train_set_size, 
                                   same_patch_assump, diff_patch_assump, same_patch_aug, diff_patch_aug)
validation_generator = DataGenerator(epochs, batch_size, dataset_name, patch_size, test_set, test_set_size, 
                                     same_patch_assump, diff_patch_assump, same_patch_aug, diff_patch_aug)

#Generate train and test pairs at once
# tr_pairs, tr_y=unsupervised_loaddata(dataset_name, patch_size, train_set, train_set_size, False)
te_pairs, te_y=unsupervised_loaddata(dataset_name, patch_size, test_set, test_set_size, False)
# print('number of train pairs:', len(tr_pairs))
# print('number of validation pairs:', len(te_pairs))

def base_model_xception(input_shape):
    base_model = Xception(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling='avg',)
    x = base_model.output
    x = Dense(512, activation="relu", name='fc_6')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def base_model_vgg(input_shape):
    base_model = VGG16(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling='avg',)
    x = base_model.output
    x = Dense(512, activation="relu", name='fc_6')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def base_model_alex(input_shape):
    alexnet=Sequential()
    alexnet.add(Conv2D(96, (7, 7), strides=(2,2), 
                padding='same', name='conv_1', input_shape=input_shape))
    #alexnet.add(BatchNormalization())
    alexnet.add(Activation(activation='relu', name='act_1'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    alexnet.add(Conv2D(256, (5, 5), strides=(2,2), padding='same', name='conv_2'))
    #alexnet.add(BatchNormalization())
    alexnet.add(Activation(activation='relu', name='act_2'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    alexnet.add(Conv2D(384, (3, 3), strides=(1,1), padding='same', name='conv_3'))
    #alexnet.add(BatchNormalization())
    alexnet.add(Activation(activation='relu', name='act_3'))

    alexnet.add(Conv2D(384, (3, 3), strides=(1,1), padding='same', name='conv_4'))
    #alexnet.add(BatchNormalization())
    alexnet.add(Activation(activation='relu', name='act_4'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    alexnet.add(Conv2D(256, (3, 3), strides=(1,1), padding='same', name='conv_5'))
    #alexnet.add(BatchNormalization())
    alexnet.add(Activation(activation='relu', name='act_5'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) 

    alexnet.add(Flatten())
    alexnet.add(Dense(512, name='fc_6'))
    #alexnet.add(BatchNormalization())
    alexnet.add(Activation(activation='relu', name='act_6'))
    #alexnet.add(Dropout(0.5))
    return alexnet

def base_model_berat(input_shape):
    inputs = Input(shape=input_shape)
    conv_1=Conv2D(64,(5,5),padding="same",activation='relu',name='conv_1')(inputs)
    conv_1=MaxPooling2D(pool_size=(2, 2))(conv_1)
    conv_2=Conv2D(128,(5,5),padding="same",activation='relu',name='conv_2')(conv_1)
    conv_2=MaxPooling2D(pool_size=(2, 2))(conv_2)
    conv_3=Conv2D(256,(3,3),padding="same",activation='relu',name='conv_3')(conv_2)
    conv_3=MaxPooling2D(pool_size=(2, 2))(conv_3)
    conv_4=Conv2D(512,(3,3),padding="same",activation='relu',name='conv_4')(conv_3)
    conv_5=Conv2D(512,(3,3),padding="same",activation='relu',name='conv_5')(conv_4)
    conv_5=MaxPooling2D(pool_size=(2, 2))(conv_5)

    dense_1=Flatten()(conv_5)
    dense_1=Dense(512,activation="relu", name='fc_5')(dense_1)
    dense_1=Dropout(0.5)(dense_1)
    dense_2=Dense(512,activation="relu", name='fc_6')(dense_1)
    dense_2=Dropout(0.5)(dense_2)
    return Model(inputs, dense_2)

models=[base_model_alex,base_model_vgg,base_model_xception,base_model_berat]
branch_model=models[branch]

if (continue_from_best):
    model=load_model('bestmodel'+continue_from_version)
else:
    base_network = branch_model(input_shape)
    base_network.summary()
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    x=concatenate([processed_a, processed_b], axis=1)
    x=Dense(1024, activation = 'relu', name='fc_7')(x)
    x=Dense(1024, activation = 'relu', name='fc_8')(x)
    x=Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input_a, input_b], outputs=x)

class MyThresholdCallback(Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        accuracy = logs['val_accuracy']
        if accuracy >= self.threshold:
            self.model.stop_training = True
    
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=patient)
thresh=MyThresholdCallback(threshold=1.0)
mcp = ModelCheckpoint('bestmodel', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
logs = CSVLogger('log')
adam = Adam(lr=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

#Loads all data together
# model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
#           callbacks=[es, mcp, logs])

#Loads data batch by batch
model.fit(training_generator,
          validation_data = validation_generator,
          epochs = epochs,
          callbacks = [es, thresh, mcp, logs])

del model
model=load_model('bestmodel')
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = accuracy(te_y, y_pred.round())
print('* Accuracy on test set: %0.4f%%' % (100 * te_acc))
del model

shutil.rmtree(dataset_name+'_results', ignore_errors=True)

os.makedirs(dataset_name+exp_name+'_results',exist_ok=True)
os.makedirs(dataset_name+exp_name+'_results/test_results',exist_ok=True)
os.makedirs(dataset_name+exp_name+'_results/special_results',exist_ok=True)
os.makedirs(dataset_name+exp_name+'_results/similar_patch_saliency',exist_ok=True)
os.makedirs(dataset_name+exp_name+'_results/different_patch_saliency',exist_ok=True)
os.makedirs(dataset_name+exp_name+'_results/similar_pair_similarity',exist_ok=True)
os.makedirs(dataset_name+exp_name+'_results/different_pair_similarity',exist_ok=True)
os.makedirs(dataset_name+exp_name+'_results/similar_max_and_std_saliency',exist_ok=True)
os.makedirs(dataset_name+exp_name+'_results/different_max_and_std_saliency',exist_ok=True)

test_results_folder=dataset_name+exp_name+'_results/test_results'
special_results_folder=dataset_name+exp_name+'_results/special_results'
similar_patch_saliency_results_folder=dataset_name+exp_name+'_results/similar_patch_saliency'
different_patch_saliency_results_folder=dataset_name+exp_name+'_results/different_patch_saliency'
similar_pair_similarity_results_folder=dataset_name+exp_name+'_results/similar_pair_similarity'
different_pair_similarity_results_folder=dataset_name+exp_name+'_results/different_pair_similarity'
similar_max_and_std_saliency_results_folder=dataset_name+exp_name+'_results/similar_max_and_std_saliency'
different_max_and_std_saliency_results_folder=dataset_name+exp_name+'_results/different_max_and_std_saliency'

visualize_lines.pca(patch_size, inner_size, test_set, test_results_folder)
visualize_lines.pca(patch_size, inner_size, special_set, special_results_folder)
visualize_patch_saliency.patch_saliency(patch_size, similar_pairs_set, similar_patch_saliency_results_folder)
visualize_patch_saliency.patch_saliency(patch_size, different_pairs_set, different_patch_saliency_results_folder)
visualize_pair_similarity.pair_similarity(patch_size, similar_pairs_set, similar_pair_similarity_results_folder)
visualize_pair_similarity.pair_similarity(patch_size, different_pairs_set, different_pair_similarity_results_folder)
visualize_max_and_std_saliency.max_and_std_saliency(patch_size, similar_pairs_set,
                                                    similar_max_and_std_saliency_results_folder)
visualize_max_and_std_saliency.max_and_std_saliency(patch_size, different_pairs_set,
                                                    different_max_and_std_saliency_results_folder)

