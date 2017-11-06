import pandas as pd
import numpy as np
import os
import math
from keras.models import Model, load_model
from sample_luna import lung_seg


luna_path = "E:/seg_luna16/"
luna_csv = luna_path + "csvfiles/"
luna_data = luna_path + "data/"
py_path = "E:/3d_lung_nodule/"
model_path = py_path + "model_results/"

candidates_file = "E:/LUNA16/csvfiles/candidates.csv"
luna_info = luna_csv + "luna_info.csv"
model_file = model_path + "train_32_model_v5.h5"

candidates_df = pd.read_csv(candidates_file, encoding='utf-8')
luna_info_df = pd.read_csv(luna_info, encoding='utf-8', index_col=0)
candidates_df['prob'] = 0

model_32 = load_model(model_file)


def set_window_width(image, MIN_BOUND=-1000.0):
    image[image < MIN_BOUND] = MIN_BOUND
    return image


def sample_cube(index, coord_voxel, lung, cube_size=32):
    zyx_1 = coord_voxel - cube_size//2
    zyx_2 = coord_voxel + cube_size//2

    for i in range(3):
        if zyx_1[i] < 0:
            zyx_1[i] = 0
            zyx_2[i] = zyx_1[i] + cube_size
        elif zyx_2[i] > lung.shape[i]:
            zyx_2[i] = lung.shape[i]
            zyx_1[i] = zyx_2[i] - lung.shape[i]


    sample_box = lung[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]
    sample_box = set_window_width(sample_box)
    sample_box = (sample_box + 1000.)/(600. + 1000.)
    sample_box = np.clip(sample_box, 0, 1)
    sample_box = np.expand_dims(sample_box, 0)
    sample_box = np.expand_dims(sample_box, 1)

    return sample_box



for i in range(candidates_df.shape[0]):
    if i == 0 or candidates_df.ix[i, 'seriesuid'] != candidates_df.ix[i-1, 'seriesuid']:
        lung_file = luna_data + candidates_df.ix[i, "seriesuid"] + '.npy'
        lung = np.load(lung_file)
        temp = luna_info_df[luna_info_df['seriesuid'] == candidates_df.ix[i, 'seriesuid']]
        origin_z = temp.iloc[0, 1]
        origin_y = temp.iloc[0, 2]
        origin_x = temp.iloc[0, 3]
        origin = np.array([origin_z, origin_y, origin_x])
        new_spacing = np.array([temp.iloc[0, 7], temp.iloc[0, 8], temp.iloc[0, 9]])

    coord_world = np.array([candidates_df.ix[i, 'coordZ'], 
                            candidates_df.ix[i, 'coordY'],
                            candidates_df.ix[i, 'coordX']])
    coord_voxel = np.rint((coord_world - origin) / new_spacing)
    coord_voxel = np.array(coord_voxel, dtype=int)

    candidate = sample_cube(i, coord_voxel, lung)
    if candidate.shape[2] != 32 or candidate.shape[3] != 32 or candidate.shape[4] != 32:
        continue

    prob = model_32.predict_on_batch(candidate)
    print("%d, %.8f" % (i, prob))
    candidates_df.ix[i, 'prob'] = prob[0, 0]



candidates_df.to_csv(model_path+"candidates_results.csv", index=False, encoding='utf-8')





