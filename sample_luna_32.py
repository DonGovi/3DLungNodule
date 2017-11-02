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
pos_save_path = "E:/3d_lung_nodule/samples_32/pos_samples/"
neg_save_path = "E:/3d_lung_nodule/samples_32/neg_samples/"

candidates_df = pd.read_csv(candidates_file)
pos_df = candidates_df[candidates_df['class']==1]
neg_df = candidates_df[candidates_df['class']==0]

pos_index = pos_df.index.tolist()
pos_index = sorted(pos_index)

neg_slice = random.sample(neg_df.index.tolist(), 1400)
neg_slice = sorted(neg_slice)

del pos_df, neg_df
'''
print(pos_df.shape)
print(neg_df.shape)
'''
def set_window_width(image, MIN_BOUND=-1000.0):
    image[image < MIN_BOUND] = MIN_BOUND
    return image

def coord_trans(df_index, origin, new_spacing):

    nodule_z = candidates_df.ix[df_index, 'coordZ']
    nodule_y = candidates_df.ix[df_index, 'coordY']
    nodule_x = candidates_df.ix[df_index, 'coordX']
    nodule_world = np.array([nodule_z, nodule_y, nodule_x])
    nodule_voxel = np.rint((nodule_world - origin) / new_spacing)
    nodule_voxel = np.array(nodule_voxel, dtype=int)

    return nodule_voxel

def sample_cube(num, df_index, origin, new_spacing, image, cube_size=32):

    nodule_voxel = coord_trans(df_index, origin, new_spacing)
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
    if candidates_df.ix[df_index, 'class'] == 0:
        np.save(os.path.join(neg_save_path,"neg_sample_s32_%d.npy" % (num)), sample_box)
        print("Saved neg_sample_s32_%d.npy" % (num))
    elif candidates_df.ix[df_index, 'class'] == 1:
        np.save(os.path.join(pos_save_path,"pos_sample_s32_%d.npy" % (num)), sample_box)
        print("Saved pos_sample_s32_%d.npy" % (num))


def sample_all(sample_list, cls=0):
    if cls == 0:
        print("Neg sample started...")
    elif cls == 1:
        print("Pos sample started...")

    for i in range(len(sample_list)):
        if i == 0 or candidates_df.ix[sample_list[i], 'seriesuid'] != candidates_df.ix[sample_list[i-1], 'seriesuid']:
            filename = data_path + str(candidates_df.ix[sample_list[i], 'seriesuid']) + '.mhd'
            seg_lung, origin, new_spacing = ls.lung_seg(filename)
            sample_cube(i, sample_list[i], origin, new_spacing, seg_lung)

        else:
            sample_cube(i, sample_list[i], origin, new_spacing, seg_lung)

    print("Job done.")


if __name__ == '__main__':
    sample_all(pos_index, cls=1)
    sample_all(neg_slice, cls=0)








