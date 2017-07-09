import numpy as np
import os

path = "E:/sample_patients/train_data/train_npy/"

file_list = os.listdir(path)


val_arr = np.zeros((1, 64, 64, 64))
print(val_arr.shape)

for filename in file_list:
    nodule_arr = np.load(path+filename)
    print("load %s" % filename)
    nodule_arr = np.expand_dims(nodule_arr, axis=0)
    val_arr = np.concatenate((val_arr, nodule_arr), axis=0)
    print(val_arr.shape)

val_arr = val_arr[1:]
print(val_arr.shape)

np.save("E:/sample_patients/train_set_one_array/all_nodule_set.npy", val_arr)