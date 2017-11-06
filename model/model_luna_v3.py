# -*-coding:utf-8-*-

import h5py
import numpy as np
import pandas as pd
from keras.layers.convolutional import Conv3D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianDropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import MaxPooling3D, AveragePooling3D, GlobalMaxPooling3D
from keras.layers.core import Dense
from keras.models import Model, load_model
from keras.optimizers import Nadam
from keras.regularizers import l2
from keras.layers import Input, Lambda, Dense, Flatten, Reshape, concatenate, Highway, Activation,Dropout
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

    x1 = Conv3D(num_filters,3,3,3,border_mode='same',W_regularizer=l2(1e-4),init=init)(x1)
    
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

def build_model(input_shape ,lr_start=1e-5):

    xin = Input(input_shape)

    x1 = conv_block(xin,8,activation='crelu')
    x1_ident = AveragePooling3D()(xin)
    x1_merged = concatenate([x1, x1_ident], axis=1)
    
    x2_1 = conv_block(x1_merged,24,activation='crelu',init='orthogonal') 
    x2_ident = AveragePooling3D()(x1_ident)
    x2_merged = concatenate([x2_1,x2_ident], axis=1)
    
    #by branching we reduce the #params
    x3_1 = conv_block(x2_merged,36,activation='crelu',init='orthogonal') 
    x3_ident = AveragePooling3D()(x2_ident)
    x3_merged = concatenate([x3_1,x3_ident], axis=1)

    x4_1 = conv_block(x3_merged,36,activation='crelu',init='orthogonal') 
    x4_ident = AveragePooling3D()(x3_ident)
    x4_merged = concatenate([x4_1,x4_ident], axis=1)
    
    x5_1 = conv_block(x4_merged,64,pool=False,init='orthogonal') 
    
    xpool = BatchNormalization()(GlobalMaxPooling3D()(x5_1))
    
    xout = dense_branch(xpool,outsize=1,activation='sigmoid')
    
    
    model = Model(input=xin,output=xout)
    
        
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


def data_generator(trainval_pos, trainval_neg, batch_size, augment=True):

    #nodule_arr = np.load(nodule_npy)             结节样本数量*64*64*64
    #healthy_arr = np.load(healthy_npy)           健康样本数量*64*64*64
    pos_num = trainval_pos.shape[0]           # 结节样本数量
    val_num = trainval_neg.shape[0]         # 健康样本数量

    trainval_arr = np.concatenate((trainval_pos, trainval_neg), axis=0)      # (结节+健康样本数量)*shape
    del trainval_pos, trainval_neg

    label = np.zeros(trainval_arr.shape[0])    # 样本标签，结节+健康样本数量
    label[:pos_num] = 1                  # 结节样本标签设置为1

    #label = to_categorical(label, 2)


    while True:

        ixs = np.random.choice(range(trainval_arr.shape[0]),size=batch_size,replace=False)
        Xbatch, Ybatch = trainval_arr[ixs], label[ixs]
        Xbatch = np.expand_dims(Xbatch, axis=1)
        #print(Xbatch.shape)
        if augment:
            Xbatch = random_perturb(Xbatch)
        Xbatch = Xbatch.astype('float32')
        Ybatch = Ybatch.astype('float32')
        Xbatch = (Xbatch + 1000.) / (600. + 1000.)
        Xbatch = np.clip(Xbatch, 0, 1)
        #print(Xbatch, Ybatch)
        yield (Xbatch, Ybatch)


def train_on_data(model, stage=1):

    pos_32_file = "E:/3d_lung_nodule/samples_32/pos_samples/trainval_pos.npy"
    neg_32_file = "E:/3d_lung_nodule/samples_32/neg_samples/trainval_neg.npy"
    pos_16_file = "E:/3d_lung_nodule/samples_16/pos_samples/pos_16_samples.npy"
    neg_16_file = "E:/3d_lung_nodule/samples_16/neg_samples/neg_16_samples.npy"

    if stage == 1:
        #导入数组文件
        pos_arr = np.load(pos_32_file)
        neg_arr = np.load(neg_32_file)
        #划分train set，选取后split_num个为val set
        split_num = 100

        pos_train = pos_arr[:pos_arr.shape[0]-split_num]
        neg_train = neg_arr[:neg_arr.shape[0]-split_num]
        #选取val set
        pos_val = pos_arr[pos_arr.shape[0]-split_num:pos_arr.shape[0]]
        neg_val = neg_arr[neg_arr.shape[0]-split_num:neg_arr.shape[0]]

        del pos_arr, neg_arr

        train_generator = data_generator(pos_train, neg_train, batch_size=64)
        val_generator = data_generator(pos_val, neg_val, batch_size=64)

        csv_logger = callbacks.CSVLogger('E:/3d_lung_nodule/model_results/train_32_model_v5.log', append=True)

        model.fit_generator(train_generator, steps_per_epoch=50, epochs=500, validation_data=val_generator,
                callbacks=[csv_logger], validation_steps=5)
        return model

    elif stage == 2:
        pos_arr = np.load(pos_16_file)
        neg_arr = np.load(neg_16_file)
        #划分train set，选取后split_num个为val set
        split_num = 100

        pos_train = pos_arr[:pos_arr.shape[0]-split_num]
        neg_train = neg_arr[:neg_arr.shape[0]-split_num]
        #选取val set
        pos_val = pos_arr[pos_arr.shape[0]-split_num:pos_arr.shape[0]]
        neg_val = neg_arr[neg_arr.shape[0]-split_num:neg_arr.shape[0]]

        del pos_arr, neg_arr

        train_generator = data_generator(pos_train, neg_train, batch_size=64)
        val_generator = data_generator(pos_val, neg_val, batch_size=64)

        csv_logger = callbacks.CSVLogger('E:/3d_lung_nodule/model_results/train_32_model_v5.log', append=True)

        model.fit_generator(train_generator, steps_per_epoch=40, epochs=300, validation_data=val_generator,
                callbacks=[csv_logger], validation_steps=5)
        return model

    elif stage == 3:
        pos_arr = np.load(pos_32_file)
        neg_arr = np.load(neg_32_file)
        #划分train set，选取后split_num个为val set
        split_num = 100

        pos_train = pos_arr[:pos_arr.shape[0]-split_num]
        neg_train = neg_arr[:neg_arr.shape[0]-split_num]
        #选取val set
        pos_val = pos_arr[pos_arr.shape[0]-split_num:pos_arr.shape[0]]
        neg_val = neg_arr[neg_arr.shape[0]-split_num:neg_arr.shape[0]]

        del pos_arr, neg_arr

        train_generator = data_generator(pos_train, neg_train, batch_size=64)
        val_generator = data_generator(pos_val, neg_val, batch_size=64)

        csv_logger = callbacks.CSVLogger('E:/3d_lung_nodule/model_results/train_32_model_v5.log', append=True)

        model.fit_generator(train_generator, steps_per_epoch=50, epochs=200, validation_data=val_generator,
                callbacks=[csv_logger], validation_steps=5)
        return model






model_32 = build_model((1, 32, 32, 32), lr_start=1e-5)
model_32.summary()
model_32 = train_on_data(model_32, stage=1)
model_32.save_weights("E:/3d_lung_nodule/model_results/train_32_weights_v3.h5")
del model_32

model_16 = build_model((1, 16, 16, 16), lr_start=1e-6)
model_16.summary()
model_16.load_weights("E:/3d_lung_nodule/model_results/train_32_weights_v3.h5")

model_16 = train_on_data(model_16, stage=2)
model_16.save_weights("E:/3d_lung_nodule/model_results/train_32_weights_v4.h5")
del model_16

model_32 = build_model((1, 32, 32, 32), lr_start=1e-6)
model_32.summary()
model_32.load_weights("E:/3d_lung_nodule/model_results/train_32_weights_v4.h5")
model_32 = train_on_data(model_32, stage=3)
model_32.save("E:/3d_lung_nodule/model_results/train_32_model_v5.h5")

print("Job Done")







