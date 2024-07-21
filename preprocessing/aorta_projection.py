import os
from tqdm import trange
import SimpleITK as sitk
from scipy import ndimage
from PIL import Image
import numpy as np
from scipy.ndimage import rotate


# function of aorta rotation
def the_rotate(in_arr, angle_):

    x_0 = rotate(in_arr, angle=angle_)
    a, b = x_0.shape
    re_size = 224
    x_1 = x_0[int(a/2)-int(re_size/2):int(a/2)+int(re_size/2), int(a/2)-int(re_size/2):int(a/2)+int(re_size/2)]
    return x_1


# function of projection of aorta in different views
def projection(in_path_, save_png, angle):

    label = sitk.ReadImage(in_path_)
    label_arr = sitk.GetArrayFromImage(label)

    arr = ndimage.binary_dilation(label_arr, [[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                              [[1, 1, 1], [1, 1, 1], [1, 1, 1]]], iterations=1).astype(int)
    # get edge of aorta
    label_arr = arr - label_arr
    re_size = 224

    # resize 512 to 224
    label_arr = ndimage.zoom(label_arr, (re_size/label_arr.shape[0], re_size/label_arr.shape[1], re_size/label_arr.shape[2]), order=0)
    rotate_arr = np.empty([re_size, re_size, re_size])
    new_arr = np.empty([re_size, re_size])

    for i in trange(re_size):
        rotate_arr[i, :, :] = the_rotate(label_arr[i, :, :], angle)

    for x in range(re_size):
        for z in range(re_size):
            new_arr[x, z] = sum(rotate_arr[x, z, :])

    # save .png
    im = Image.fromarray(new_arr * new_arr * 2)
    im = im.convert('RGB')
    im_png = save_png + '.png'
    im.save(im_png)


def mutil_view_projection(in_path, save_path):

    if not os.path.exists(os.path.join(save_path, 'coronal')):
        os.makedirs(os.path.join(save_path, 'coronal'))
        os.makedirs(os.path.join(save_path, 'coronal-30'))
        os.makedirs(os.path.join(save_path, 'coronal-45'))

    path_coronal = os.path.join(save_path, 'coronal')
    path_coronal_30 = os.path.join(save_path, 'coronal-30')
    path_coronal_45 = os.path.join(save_path, 'coronal-45')

    name_list = os.listdir(in_path)
    for name in name_list:
        projection(os.path.join(in_path, name), os.path.join(path_coronal, name[0:-7]), 0)
        projection(os.path.join(in_path, name), os.path.join(path_coronal_30, name[0:-7]), -30)
        projection(os.path.join(in_path, name), os.path.join(path_coronal_45, name[0:-7]), -45)
