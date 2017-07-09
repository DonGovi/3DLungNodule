import SimpleITK as sitk
import pandas as pd
import numpy as np
import os
import csv
from glob import glob
import scipy.ndimage
import math

#setting files path
path = "D:/sample_patients/"
files_path = path + "val/"
annot_path = path + "csv_files/val/"

#setting output path
output_path = path + 'val_32_data/nodule_32/'


#check its exist

'''    
if os.path.isdir(path + '/val_output'):
    pass
else:
    os.mkdir(path + '/val_output')
'''
#read annotations file
df_nodule = pd.read_csv(annot_path + 'annotations.csv')

#set the scan's spacing, reshape the image
def resample(image, old_spacing, new_spacing=[1, 1, 1]):
    resize_factor = old_spacing / new_spacing
    new_shape = image.shape * resize_factor
    new_shape = np.round(new_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing

#set the min HU -1000, any value <-1000 setted to -1000
def set_window_width(image, MIN_BOUND=-1000.0):
    image[image < MIN_BOUND] = MIN_BOUND
    return image


for i in range(df_nodule.shape[0]):
    img_file = files_path + df_nodule.ix[i,'seriesuid'] + '.mhd'
    if os.path.exists(img_file):
        print("Sample the %s.mhd" % df_nodule.ix[i,'seriesuid'])
        nodule_x = df_nodule.ix[i, 'coordX']
        nodule_y = df_nodule.ix[i, 'coordY']
        nodule_z = df_nodule.ix[i, 'coordZ']
        full_image_info = sitk.ReadImage(img_file)
        full_scan = sitk.GetArrayFromImage(full_image_info)   #transfer image into array, order:z,y,x
        origin = np.array(full_image_info.GetOrigin())[::-1]   #find the origin and set the order z,y,x
        old_spacing = np.array(full_image_info.GetSpacing())[::-1]  #find the old voxel spacing
        image, new_spacing = resample(full_scan, old_spacing)    #resample the array by voxel spacing almost [1,1,1]
        nodule_center = np.array([nodule_z, nodule_y, nodule_x])    #nodule center coord in world coordinates
        v_center = np.rint((nodule_center - origin) / new_spacing)    # transfer into voxel coordinates
        v_center = np.array(v_center, dtype=int)          # set its type to int
        '''
    #judge sample size by the nodule's diameter
        if df_nodule.ix[i, 'diameter_mm'] < 5:
            window_size = 7
        elif df_nodule.ix[i, 'diameter_mm'] < 10:
            window_size = 9
        elif df_nodule.ix[i, 'diameter_mm'] < 20:
            window_size = 15
        elif df_nodule.ix[i, 'diameter_mm'] < 25:
            window_size = 17
        elif df_nodule.ix[i, 'diameter_mm'] < 30:
            window_size = 20
        else:
            window_size = 22
        '''
        cube_size = 32                  # sample size
        zyx_1 = v_center - 16
        zyx_2 = v_center + 16

        nodule_box = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]      #select the nodule array 32*32*32
        nodule_box = set_window_width(nodule_box)
        print(nodule_box.shape)
        '''
        zyx_1 = v_center - window_size           #min border
        zyx_2 = v_center + window_size + 1        #max border
        nodule_box = np.zeros([cube_size, cube_size, cube_size],np.int16)    
        img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]   #print the nodule cube
        img_crop = set_window_width(img_crop)         #set all HU<-1000 to -1000
        zeros_fill = math.floor((cube_size - (2*window_size+1))/2)   # the border's width
        nodule_box[zeros_fill:cube_size-zeros_fill-1, zeros_fill:cube_size-zeros_fill-1,zeros_fill:cube_size-zeros_fill-1] = img_crop
    # put the nodule cube into nodule box, and the border is padding by 0

        nodule_box[nodule_box == 0] = -1000  # set the border to -1000 HU
        '''
        np.save(os.path.join(output_path,"val_nodule_s32_%d.npy" % (i)), nodule_box) #save npy