#-*-coding:utf-8-*-

import numpy as np
import os
import math
from keras.models import Model, load_model

path = "E:/3d_lung_nodule/"
pos_path = path + "samples_32/pos_samples/"
neg_path = path + "samples_32/neg_samples/"

pos_test_file = pos_path + "test_pos.npy"
neg_test_file = neg_path + "test_neg.npy"

model_path = path + "model_results/"
model_file = model_path + "train_32_model_v5.h5"

model = load_model(model_file)

pos_arr = np.load(pos_test_file)
neg_arr = np.load(neg_test_file)

threshold = 0.5
tp = 0
fp = 0
tn = 0
fn = 0

for i in range(pos_arr.shape[0]):
    pre_arr = pos_arr[i]
    pre_arr = (pre_arr + 1000.) / (600. + 1000.)
    pre_arr = np.clip(pre_arr, 0, 1)

    pre_arr = np.expand_dims(pre_arr, 0)
    pre_arr = np.expand_dims(pre_arr, 1)

    prob = model.predict_on_batch(pre_arr)
    if prob > threshold:
        tp += 1
    else:
        fn += 1


for j in range(neg_arr.shape[0]):
    pre_arr = neg_arr[j]
    pre_arr = (pre_arr + 1000.) / (600. + 1000.)
    pre_arr = np.clip(pre_arr, 0, 1)

    pre_arr = np.expand_dims(pre_arr, 0)
    pre_arr = np.expand_dims(pre_arr, 1)

    prob = model.predict_on_batch(pre_arr)
    print(prob)
    if prob < 1-threshold:
        tn += 1
    else:
        fp += 1

accur = tp / (tp + fp)
recall = tp / (tp + fn)

print("accur = %.4f" % accur)
print("recall = %.4f" % recall)