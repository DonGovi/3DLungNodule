# -*-coding:utf-8-*-

import h5py
import numpy as np
import pandas as pd
from keras.layers.convolutional import Convolution3D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianDropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import MaxPooling3D, AveragePooling3D, GlobalMaxPooling3D
from keras.layers.core import Dense
from keras.models import Model, load_model
from keras.optimizers import Nadam
from keras.regularizers import l2
from keras.layers import Input, Lambda, Dense, Flatten, Reshape, merge, Highway, Activation,Dropout
from keras import backend as K
from keras.utils import to_categorical
from keras import callbacks


def leakyCReLU(x):
    x_pos = K.relu(x, .0)
    x_neg = K.relu(-x, .0)
    return K.concatenate([x_pos, x_neg], axis=1)
    
def leakyCReLUShape(x_shape):
    shape = list(x_shape)
    shape[1] *= 2
    return tuple(shape)
    
def conv_block(x_input, num_filters,pool=True,activation='relu',init='orthogonal'):
    
    x1 = BatchNormalization(axis=1,momentum=0.995)(x_input)
    if activation == 'crelu':
        x1 = Lambda(leakyCReLU, output_shape=leakyCReLUShape)(x1)
    else:
        x1 = LeakyReLU(.01)(x1)
    x1 = Convolution3D(num_filters,3,3,3,border_mode='same',W_regularizer=l2(1e-4),init=init)(x1)
    if pool:
        x1 = MaxPooling3D()(x1)
    x_out = x1
    return x_out
    
    
    
def dense_branch(xstart, outsize=2,activation='sigmoid'):
    xdense_ = Dense(32,W_regularizer=l2(1e-4))(xstart)
    xdense_ = BatchNormalization(momentum=0.995)(xdense_)
    xdense_ = LeakyReLU(.01)(xdense_)
    xout = Dense(outsize,activation=activation,W_regularizer=l2(1e-4))(xdense_)
    return xout

def build_model(input_shape):

    xin = Input(input_shape)
    
    x1 = conv_block(xin,8,activation='crelu')
    x1_ident = AveragePooling3D()(xin)
    x1_merged = merge([x1, x1_ident],mode='concat', concat_axis=1)
    
    x2_1 = conv_block(x1_merged,24,activation='crelu',init='orthogonal') 
    x2_ident = AveragePooling3D()(x1_ident)
    x2_merged = merge([x2_1,x2_ident],mode='concat', concat_axis=1)
    
    #by branching we reduce the #params
    x3_1 = conv_block(x2_merged,36,activation='crelu',init='orthogonal') 
    x3_ident = AveragePooling3D()(x2_ident)
    x3_merged = merge([x3_1,x3_ident],mode='concat', concat_axis=1)

    x4_1 = conv_block(x3_merged,36,activation='crelu',init='orthogonal') 
    x4_ident = AveragePooling3D()(x3_ident)
    x4_merged = merge([x4_1,x4_ident],mode='concat', concat_axis=1)
    
    x5_1 = conv_block(x4_merged,64,pool=False,init='orthogonal') 
    
    xpool = BatchNormalization()(GlobalMaxPooling3D()(x5_1))
    
    xout = dense_branch(xpool,outsize=1,activation='sigmoid')
    
    
    model = Model(input=xin,output=xout)
    
    if input_shape[1] == 32:
        lr_start = 1e-5
    elif input_shape[1] == 64:
        lr_start = 1e-5
    elif input_shape[1] == 128:
        lr_start = 1e-4
    elif input_shape[1] == 96:
        lr_start = 5e-4
    
        
    opt = Nadam(lr_start,clipvalue=1.0)
    print('compiling model')

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def random_perturb(Xbatch): 
    #apply some random transformations...
    swaps = np.random.choice([-1,1],size=(Xbatch.shape[0],3))
    for i in range(Xbatch.shape[0]):
        #(1,32,32,32)
        #random 
        Xbatch[i] = Xbatch[i,:,::swaps[i,0],::swaps[i,1],::swaps[i,2]]
        txpose = np.random.permutation([1,2,3])
        Xbatch[i] = np.transpose(Xbatch[i], tuple([0] + list(txpose)))
    return Xbatch


def data_generator(nodule_arr, healthy_arr, batch_size, augment=True):

    #nodule_arr = np.load(nodule_npy)             结节样本数量*64*64*64
    #healthy_arr = np.load(healthy_npy)           健康样本数量*64*64*64
    nodule_num = nodule_arr.shape[0]           # 结节样本数量
    healthy_num = healthy_arr.shape[0]         # 健康样本数量

    train_arr = np.concatenate((nodule_arr, healthy_arr), axis=0)      # (结节+健康样本数量)*shape
    del nodule_arr, healthy_arr

    label = np.zeros(train_arr.shape[0])    # 样本标签，结节+健康样本数量
    label[:nodule_num] = 1                  # 结节样本标签设置为1

    #label = to_categorical(label, 2)


    while True:

        ixs = np.random.choice(range(train_arr.shape[0]),size=batch_size,replace=False)
        Xbatch, Ybatch = train_arr[ixs], label[ixs]
        Xbatch = np.expand_dims(Xbatch, 1)
        print(Xbatch.shape)
        if augment:
            Xbatch = random_perturb(Xbatch)
        Xbatch = Xbatch.astype('float32')
        Ybatch = Ybatch.astype('float32')
        Xbatch = (Xbatch + 1000.) / (600. + 1000.)
        Xbatch = np.clip(Xbatch, 0, 1)
        #print(Xbatch, Ybatch)
        yield (Xbatch, Ybatch)


def train_on_data(model, split=True):

    nodule_train_file = "E:/sample_patients/train_conv3d_model/all_nodule_32.npy"
    healthy_train_file = "E:/sample_patients/train_conv3d_model/small_healthy_32.npy"

    if split:   #split代表是否分割出val set
        #导入数组文件
        nodule_arr = np.load(nodule_train_file)
        healthy_arr = np.load(healthy_train_file)
        #划分train set，选取后split_num个为val set
        split_num = 150

        nodule_train = nodule_arr[:nodule_arr.shape[0]-split_num]
        healthy_train = healthy_arr[:healthy_arr.shape[0]-split_num]
        #选取val set
        nodule_val = nodule_arr[nodule_arr.shape[0]-split_num:nodule_arr.shape[0]]
        healthy_val = healthy_arr[healthy_arr.shape[0]-split_num:healthy_arr.shape[0]]

        del nodule_arr, healthy_arr

        train_generator = data_generator(nodule_train, healthy_train, batch_size=64)
        val_generator = data_generator(nodule_val, healthy_val, batch_size=64)

        csv_logger = callbacks.CSVLogger('E:/sample_patients/model_result/train_32_model_v3.log', append=True)

        model.fit_generator(train_generator, steps_per_epoch=50, epochs=300, validation_data=val_generator,
                callbacks=[csv_logger], validation_steps=5)
        return model

    else:
        nodule_arr = np.load(nodule_train_file)
        healthy_arr = np.load(healthy_train_file)

        train_generator = data_generator(nodule_arr, healthy_arr, batch_size=100)
    
        model.fit_generator(train_generator, steps_per_epoch=2000, epochs=20)

        return model  


model_32 = build_model((1, 32, 32, 32))
model_32.summary()

model_32 = train_on_data(model_32)

model_32.save("E:/sample_patients/model_result/train_32_model_v3.h5")

del model_32

