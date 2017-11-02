import numpy as np
import os
from random import sample

path = "E:/3d_lung_nodule/"
pos_path = path + "pos_samples/"
neg_path = path + "neg_samples/"

all_pos_list = os.listdir(pos_path)
all_neg_list = os.listdir(neg_path)


def sample_test(file_list, num=100):
    trainval_list = file_list.copy()

    test_list = sample(file_list, num)
    for testname in test_list:
        trainval_list.remove(testname)

    return trainval_list, test_list

def combination(file_list, file_path):

    comb_arr = np.zeros((1, 32, 32, 32))

    for filename in file_list:
        arr = np.load(file_path+filename)
        print("load %s" % filename)
        arr = np.expand_dims(arr, axis=0)
        comb_arr = np.concatenate((comb_arr, arr), axis=0)
        print(comb_arr.shape)

    comb_arr = comb_arr[1:]
    print(comb_arr.shape)

    return comb_arr

if __name__ == '__main__':
    trainval_pos_list, test_pos_list = sample_test(all_pos_list)
    trainval_neg_list, test_neg_list = sample_test(all_neg_list)
    '''
    print("Starting make trainval positive array...")
    trainval_pos = combination(trainval_pos_list, pos_path)
    np.save(pos_path+"trainval_pos.npy", trainval_pos)
    del trainval_pos
    print("Starting make test positive array...")
    test_pos = combination(test_pos_list, pos_path)
    np.save(pos_path+"test_pos.npy", test_pos)
    del test_pos
    '''
    print("Starting make trainval negative array...")
    trainval_neg = combination(trainval_neg_list, neg_path)
    np.save(neg_path+"trainval_neg.npy", trainval_neg)
    del trainval_neg
    print("Starting make test negative array...")
    test_neg = combination(test_neg_list, neg_path)
    np.save(neg_path+"test_neg.npy", test_neg)
    del test_neg

    print("Job done.")






