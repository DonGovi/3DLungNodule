import SimpleITK as sitk
import pandas as pd
import numpy as np
import os
import math
import lung_seg as ls
import random


path = "E:/LUNA16/"
csv_path = path + "csvfiles/"
data_path = path + "data/"
candidates_file = csv_path + "candidates.csv"
annotations_file = csv_path + "annotations.csv"
pos_save_path = "E:/3d_lung_nodule/samples_16/pos_samples/"
neg_save_path = "E:/3d_lung_nodule/samples_16/neg_samples/"

candidates_df = pd.read_csv(candidates_file)
annotations_df = pd.read_csv(annotations_file)
pos_df = annotations_df[annotations_df['diameter_mm']<=16]
neg_df = candidates_df[candidates_df['class']==0]

neg_slice = random.sample(neg_df.index.tolist(), 1000)
neg_slice = sorted(neg_slice)

neg_df = neg_df.loc[neg_slice]

del candidates_df, annotations_df
'''
print(pos_df.shape)
print(neg_df.shape)
'''
def set_window_width(image, MIN_BOUND=-1000.0):
    image[image < MIN_BOUND] = MIN_BOUND
    return image

def coord_trans(df, df_index, origin, new_spacing):

    nodule_z = df.iloc[df_index, 3]
    nodule_y = df.iloc[df_index, 2]
    nodule_x = df.iloc[df_index, 1]
    nodule_world = np.array([nodule_z, nodule_y, nodule_x])
    nodule_voxel = np.rint((nodule_world - origin) / new_spacing)
    nodule_voxel = np.array(nodule_voxel, dtype=int)

    return nodule_voxel

def sample_cube(df, df_index, origin, new_spacing, image, cls, cube_size=16):

    nodule_voxel = coord_trans(df, df_index, origin, new_spacing)
    zyx_1 = nodule_voxel - cube_size//2
    zyx_2 = nodule_voxel + cube_size//2

    for i in range(3):
        if zyx_1[i] < 0:
            zyx_1[i] = 0
            zyx_2[i] = zyx_1[i] + cube_size
        elif zyx_2[i] > image.shape[i]:
            zyx_2[i] = image.shape[i]
            zyx_1[i] = zyx_2[i] - image.shape[i]

    sample_box = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]
    sample_box = set_window_width(sample_box)
    print(sample_box.shape)
    if cls == 0:
        np.save(os.path.join(neg_save_path,"neg_sample_s16_%d.npy" % (df_index)), sample_box)
        print("Saved neg_sample_s16_%d.npy" % (df_index))
    else:
        np.save(os.path.join(pos_save_path,"pos_sample_s16_%d.npy" % (df_index)), sample_box)
        print("Saved pos_sample_s16_%d.npy" % (df_index))


def sample_all(pos_df, neg_df):
    print("Starting sample 16 positive...")

    for i in range(pos_df.shape[0]):
        print(i)
    # i is the row_num of dateframe, not the index, so use 'iloc' instead of 'loc' or 'ix'
        if i == 0 or pos_df.iloc[i, 0] != pos_df.iloc[i-1, 0]:
            filename = data_path + str(pos_df.iloc[i, 0]) + ".mhd"
            seg_lung, origin, new_spacing = ls.lung_seg(filename)
            sample_cube(pos_df, i, origin, new_spacing, seg_lung, 1)
        else:
            sample_cube(pos_df, i, origin, new_spacing, seg_lung, 1)

    print("Starting sample 16 negative...")

    for j in rang(neg_df.shape[0]):
        if j == 0 or neg_df.iloc[j, 0] != neg_df.iloc[j-1, 0]:
            filename = data_path + str(neg_df.iloc[j, 0]) + ".mhd"
            seg_lung, origin, new_spacing = ls.lung_seg(filename)
            sample_cube(neg_df, j, origin, new_spacing, seg_lung, 0)
        else:
            sample_cube(neg_df, j, origin, new_spacing, seg_lung, 0)

    print("Job done")



if __name__ == '__main__':
    sample_all(pos_df, neg_df)








