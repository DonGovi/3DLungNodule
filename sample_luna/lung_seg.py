#-*-coding:utf-8-*-

import SimpleITK as sitk
import pandas as pd
import numpy as np
import skimage
import os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening, convex_hull_image, disk
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc


#scan_file = "/home/donjuan/Luna16/data_set/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.mhd"
file_path = "E:/LUNA16/data/"
save_path = "E:/seg_luna16/"


def load_scan(scan_file):
    print("load %s" % scan_file)
    
    full_scan = sitk.ReadImage(scan_file)
    img_array = sitk.GetArrayFromImage(full_scan)  #numpy数组，z,y,x
    origin = np.array(full_scan.GetOrigin())[::-1]   #世界坐标原点 z,y,x
    old_spacing = np.array(full_scan.GetSpacing())[::-1]   #原体素间距
    return img_array, origin, old_spacing


def resample(image, old_spacing, new_spacing=[1, 1, 1]):
    '''
    将体素间距设为(1, 1, 1)
    '''
    resize_factor = old_spacing / new_spacing
    new_shape = image.shape * resize_factor
    new_shape = np.round(new_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing
    

def plot_3d(image, threshold=-600):
    p = image.transpose(2,1,0)

    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()


def largest_label_volume(image, bg=-1):
    vals, counts = np.unique(image, return_counts=True)    # 统计image中的值及频率
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    # 获得阈值图像
    labels = measure.label(binary_image, connectivity=1)
    # label()函数标记连通区域
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0,0,0]
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask

def extend_bounding(img):
    mask = np.copy(img)
    selem = disk(5)
    mask = binary_dilation(mask, selem)

    return mask


def extend_mask(image):
    mask = np.copy(image)
    for i_layer in range(mask.shape[0]):
        slice_img = mask[i_layer]
        mask_img = extend_bounding(slice_img)
        mask[i_layer] = mask_img

    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(mask,structure=struct,iterations=10) 
    return dilatedMask

def use_mask(image, lung_mask):
    cp_image = np.copy(image)

    return cp_image

def get_mhd_file(file_path):
    file_list = os.listdir(file_path)
    mhd_file_list = []
    for i in file_list:
        if os.path.splitext(i)[1] == '.mhd':
            mhd_file_list.append(i)

    return mhd_file_list

def lung_seg(filename):
    print("Starting %s segmentation" % filename)
    img_array, origin, old_spacing = load_scan(filename)
    image, new_spacing = resample(img_array, old_spacing)
    segmented_lungs_filled = segment_lung_mask(image, True)
    lung_mask = extend_mask(segmented_lungs_filled)
    seg_lung = image*lung_mask

    return seg_lung, origin, new_spacing


if __name__ == '__main__':
    file_list = get_mhd_file(file_path)

    for filename in file_list:
        seg_lung, origin, new_spacing = lung_seg(file_path+filename)

        print(seg_lung.shape)
        np.save(save_path+os.path.splitext(filename)[0]+'.npy', seg_lung)

    print("Job done.")


#processed_mask = process_mask(segmented_lungs_filled)
'''
vals, counts = np.unique(segmented_lungs_filled, return_counts=True)
print(vals)
print(counts)

img_array, origin, old_spacing = load_scan(scan_file)
image, new_spacing = resample(img_array, old_spacing)

print("Resample complete")
#segmented_lungs = segment_lung_mask(image, False)
segmented_lungs_filled = segment_lung_mask(image, True)
process_lung = extend_mask(segmented_lungs_filled)

print("Building mask complete")

fig,ax = plt.subplots(2,2,figsize=[8,8])
ax[0,0].imshow(image[150] ,cmap='gray')
ax[0,1].imshow(process_lung[150],cmap='gray')
ax[1,0].imshow(image[150]*process_lung[150],cmap='gray')
plt.show()
'''


#masked_image = use_mask(image, process_lung)

#print(masked_image.shape)


#plot_3d(process_lung, 0)
#plot_3d(image, -600)

'''
slice1 = segmented_lungs_filled[150, :, :]
chull1 = convex_hull_image(slice1)
dilate1 = extend_bounding(slice1)



slice2 = segmented_lungs_filled[160, :, :]
chull2 = convex_hull_image(slice2)
dilate2 = extend_bounding(slice2)


fig, axes = plt.subplots(2, 3, figsize=(10, 10))
ax0, ax1, ax2, ax3, ax4, ax5 = axes.ravel()

ax0.imshow(slice1, plt.cm.bone)
ax0.set_title('original image1')

ax1.imshow(chull1, plt.cm.bone)
ax1.set_title('chull image1')

ax2.imshow(dilate1, plt.cm.bone)
ax2.set_title('dilate image1')

ax3.imshow(slice2, plt.cm.bone)
ax3.set_title('original image2')

ax4.imshow(chull2, plt.cm.bone)
ax4.set_title('chull image2')

ax5.imshow(dilate2, plt.cm.bone)
ax5.set_title('dilate image2')


plt.show()
'''