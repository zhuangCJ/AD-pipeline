import os
from tqdm import trange
import SimpleITK as sitk
from scipy import ndimage
from PIL import Image
import numpy as np


# function of maximum projection of TL/FL in different views
def maximum_projection(in_path_, save_png_coronal, save_png_sagittal, save_png_transverse):

    label = sitk.ReadImage(in_path_)
    label_arr = sitk.GetArrayFromImage(label)

    re_size = 224

    # resize 512 to 224
    label_arr = ndimage.zoom(label_arr, (re_size/label_arr.shape[0], re_size/label_arr.shape[1], re_size/label_arr.shape[2]), order=0)
    new_arr_coronal = np.empty([re_size, re_size])
    new_arr_sagittal = np.empty([re_size, re_size])
    new_arr_transverse = np.empty([re_size, re_size])

    for x in range(re_size):
        for z in range(re_size):
            new_arr_coronal[x, z] = max(label_arr[x, z, :])
            new_arr_sagittal[x, z] = max(label_arr[x, :, z])
            new_arr_transverse[x, z] = max(label_arr[:, x, z])

    # save .png
    im_coronal = Image.fromarray(new_arr_coronal * 128 - 1)
    im_coronal = im_coronal.convert('RGB')
    im_png_coronal = save_png_coronal + '.png'
    im_coronal.save(im_png_coronal)

    im_sagittal = Image.fromarray(new_arr_sagittal * 128 - 1)
    im_sagittal = im_sagittal.convert('RGB')
    im_png_sagittal = save_png_sagittal + '.png'
    im_sagittal.save(im_png_sagittal)

    im_transverse = Image.fromarray(new_arr_transverse * 128 - 1)
    im_transverse = im_transverse.convert('RGB')
    im_png_transverse = save_png_transverse + '.png'
    im_transverse.save(im_png_transverse)


def mutil_view_maximum_projection(in_path, save_path):

    if not os.path.exists(os.path.join(save_path, 'coronal')):
        os.makedirs(os.path.join(save_path, 'coronal'))
        os.makedirs(os.path.join(save_path, 'sagittal'))
        os.makedirs(os.path.join(save_path, 'transverse'))

    path_coronal = os.path.join(save_path, 'coronal')
    path_sagittal = os.path.join(save_path, 'sagittal')
    path_transverse = os.path.join(save_path, 'transverse')

    name_list = os.listdir(in_path)
    for name in name_list:
        maximum_projection(os.path.join(in_path, name), os.path.join(path_coronal, name[0:-7]),
                           os.path.join(path_sagittal, name[0:-7]), os.path.join(path_transverse, name[0:-7]))









