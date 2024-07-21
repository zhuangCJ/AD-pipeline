import SimpleITK as sitk
import os
import numpy as np


# ct value normalization (-1024 ~ 3071)
def process(in_path, save_path):
    # image
    sitk_img = sitk.ReadImage(in_path)
    img_arr = sitk.GetArrayFromImage(sitk_img)
    img_arr = np.int16(img_arr)

    # other information
    spacing = sitk_img.GetSpacing()
    direction = sitk_img.GetDirection()
    origin = sitk_img.GetOrigin()

    img_arr_2 = np.zeros(img_arr.shape)
    img_arr_2[img_arr > 1000] = 1
    c, _, _ = img_arr_2.shape
    t = img_arr_2.sum()/c

    if t > 10000:
        img_arr -= 1024

    img_arr[img_arr < -1024] = -1024
    img_arr[img_arr > 3071] = 3071

    img_arr = sitk.GetImageFromArray(img_arr)
    img_arr.SetSpacing(spacing)
    img_arr.SetDirection(direction)
    img_arr.SetOrigin(origin)

    sitk.WriteImage(img_arr, save_path)


# batch process
def data_process(in_path, save_path):
    name_list = os.listdir(in_path)
    remain_name = []
    for name in name_list:
        if name.endswith('.nii.gz'):
            remain_name.append(name)
    for name in remain_name:
        print(name)
        if not name.endswith('_0000.nii.gz'):
            process(os.path.join(in_path, name), os.path.join(save_path, name.replace('.nii.gz', '_0000.nii.gz')))
        else:
            process(os.path.join(in_path, name), os.path.join(save_path, name))
