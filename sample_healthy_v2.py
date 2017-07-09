import SimpleITK as sitk
import pandas as pd
import numpy as np
import os
import csv
from glob import glob
import scipy.ndimage
import math

path = 'D:/sample_patients/'
healthy_output_path = path + 'train_32_data/healthy_32/'
healthy_csv = path + 'csv_files/single_nodule_case.csv'
image_path = path + 'train/'


def resample(image, old_spacing, new_spacing=[1, 1, 1]):
    resize_factor = old_spacing / new_spacing
    new_shape = image.shape * resize_factor
    new_shape = np.round(new_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing


def set_window_width(image, MIN_BOUND=-1000.0):
    image[image < MIN_BOUND] = MIN_BOUND
    return image


nodule_df = pd.read_csv(healthy_csv)

for i in range(nodule_df.shape[0]):        
    nodule_center = np.array([nodule_df.ix[i, 'coordZ'],
                              nodule_df.ix[i, 'coordY'],
                              nodule_df.ix[i, 'coordX']])
    nodule_diam = nodule_df.ix[i, 'diameter_mm']
    img_file = image_path + nodule_df.ix[i, 'seriesuid'] + ".mhd"
    print("Sample the %dth file: %s.mhd" % (i, nodule_df.ix[i, 'seriesuid']))
    full_image_info = sitk.ReadImage(img_file)                     # read mhd file
    full_scan = sitk.GetArrayFromImage(full_image_info)            # trans into numpy array, zyx
    origin = np.array(full_image_info.GetOrigin())[::-1]           # get origin's world coordinate, zyx
    old_spacing = np.array(full_image_info.GetSpacing())[::-1]     # get voxel spacing in image, zyx
    image, new_spacing = resample(full_scan, old_spacing)
    print(image.shape)          
    # resample the image and return new image and voxel spacing, near [1, 1, 1]
    v_center = np.rint((nodule_center - origin) / new_spacing)     
    # find the voxel coordinate of nodule center, also the index in numpy array
    v_center = np.array(v_center, dtype=int)     #trans it to int type
    print(v_center, nodule_diam)

    cube_size = 32
    # find the borders of nodule box
    dist = math.ceil(nodule_diam/2. + cube_size/2)

    for j in range(10):
        cube_center = np.zeros(3)      #init cube center (0, 0, 0), zyx
        cube_center[0] = np.random.choice(range(image.shape[0]), size=1, replace=False)           # random choose z
        print("init z = %d" % cube_center[0])
        cube_center[1] = np.random.choice(range(image.shape[1]), size=1, replace=False)           # random choose y
        print("init y = %d" % cube_center[1])
        cube_center[2] = np.random.choice(range(image.shape[2]), size=1, replace=False)           # random choose x
        print("init x = %d" % cube_center[2])
        while(abs(cube_center[0] - v_center[0]) <= dist):
            cube_center[0] = np.random.choice(range(image.shape[0]), size=1, replace=False)
            print("z = %d" % cube_center[0])

        while(abs(cube_center[1] - v_center[1]) <= dist):
            cube_center[1] = np.random.choice(range(image.shape[1]), size=1, replace=False)
            print("y = %d" % cube_center[1])

        while(abs(cube_center[2] - v_center[2]) <= dist):
            cube_center[2] = np.random.choice(range(image.shape[2]), size=1, replace=False)
            print("x = %d" % cube_center[2])

        z_min = int(cube_center[0] - cube_size/2)
        z_max = int(cube_center[0] + cube_size/2)
        y_min = int(cube_center[1] - cube_size/2)
        y_max = int(cube_center[1] + cube_size/2)
        x_min = int(cube_center[2] - cube_size/2)
        x_max = int(cube_center[2] + cube_size/2)
        print("%d:%d, %d:%d, %d:%d" % (z_min, z_max, y_min, y_max, x_min, x_max))

        healthy_cube = image[z_min:z_max, y_min:y_max, x_min:x_max]
        healthy_cube =set_window_width(healthy_cube)
        print(healthy_cube.shape)
        np.save(os.path.join(healthy_output_path,"train_healthy_%d_%d.npy" % (i,j)), healthy_cube)