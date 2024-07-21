import SimpleITK as sitk
import os
import tqdm
import numpy as np
import collections
import xlwt
import copy
import argparse
import preprocessing.data_batch_normal
import preprocessing.aorta_projection
import preprocessing.tfl_maximum_projection
import AD_Identification.predict_IFAD
import Stanford_Subtyping.predict_Subtyping


if __name__ == '__main__':

    # path of original CTA image
    parser = argparse.ArgumentParser(description='A pipeline of AD identification, segmentation and subtyping.')
    parser.add_argument('-i', '--img_path', default=" ", type=str, metavar='input_image',
                        help='input image directory path')
    args = parser.parse_args()

    # get root path
    current_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
    # create output path
    save_path = os.path.join(current_path, 'Output')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # make folder for processed data
    if not os.path.exists(os.path.join(save_path, 'Data_Processed')):
        os.makedirs(os.path.join(save_path, 'Data_Processed'))
    Data_processed_path = os.path.join(save_path, 'Data_Processed')

    # make folder for aorta segmentation result
    if not os.path.exists(os.path.join(save_path, 'Aorta_Segmentation')):
        os.makedirs(os.path.join(save_path, 'Aorta_Segmentation'))
    Aorta_segmentation_path = os.path.join(save_path, 'Aorta_Segmentation')

    # data preprocess
    print(' ')
    print('Start data processing.')
    preprocessing.data_batch_normal.data_process(in_path=args.img_path, save_path=Data_processed_path)
    print('Data processing completed.')

    # Aorta segmentation with nnunet
    print(' ')
    print('Start aorta segmentation.')
    os.system("nnUNet_predict -i {input_path} -o {output_path} -t 101 -m 3d_fullres -f 0".format(
        input_path=Data_processed_path, output_path=Aorta_segmentation_path))
    print('Aorta segmentation completed.')

    # make folder for aorta projection
    if not os.path.exists(os.path.join(save_path, 'Aorta_Projection')):
        os.makedirs(os.path.join(save_path, 'Aorta_Projection'))
    Aorta_projection_path = os.path.join(save_path, 'Aorta_Projection')

    # multi-view projection
    print(' ')
    print('Start multi-view projection.')
    preprocessing.aorta_projection.mutil_view_projection(in_path=Aorta_segmentation_path,
                                                         save_path=Aorta_projection_path)
    print('Multi-view projection completed.')

    # AD identification
    print(' ')
    print('Start AD identification.')
    data_list, type_list = AD_Identification.predict_IFAD.batch_AD_Iden(in_path=Aorta_projection_path,
                                                                        save_path=save_path)
    print('AD identification completed.')

    # make folder for TL/FL segmentation
    if not os.path.exists(os.path.join(save_path, 'Data_Processed_TFL')):
        os.makedirs(os.path.join(save_path, 'Data_Processed_TFL'))
    Data_processed_TFL_path = os.path.join(save_path, 'Data_Processed_TFL')

    # make folder for TL/FL segmentation result
    if not os.path.exists(os.path.join(save_path, 'TFL_Segmentation')):
        os.makedirs(os.path.join(save_path, 'TFL_Segmentation'))
    TFL_segmentation_path = os.path.join(save_path, 'TFL_Segmentation')

    # select AD data
    AD_list = []
    for data, the_type in zip(data_list, type_list):
        if the_type == 'AD':
            AD_list.append(data)

    for name in AD_list:
        ct_name = name.replace('.nii.gz', '_0000.nii.gz')

        ct_image = sitk.ReadImage(os.path.join(Data_processed_path, ct_name))
        aorta_label = sitk.ReadImage(os.path.join(Aorta_segmentation_path, name))

        spacing = ct_image.GetSpacing()
        direction = ct_image.GetDirection()
        origin = ct_image.GetOrigin()

        ct_arr = sitk.GetArrayFromImage(ct_image)
        label_arr = sitk.GetArrayFromImage(aorta_label)
        roi_arr = ct_arr*label_arr

        roi_image = sitk.GetImageFromArray(roi_arr)
        roi_image.SetSpacing(spacing)
        roi_image.SetDirection(direction)
        roi_image.SetOrigin(origin)

        sitk.WriteImage(roi_image, os.path.join(Data_processed_TFL_path, ct_name))

    # TL/FL segmentation with nnunet
    print(' ')
    print('Start TL/FL segmentation.')
    os.system("nnUNet_predict -i {input_path} -o {output_path} -t 102 -m 3d_fullres -f 0".format(
        input_path=Data_processed_TFL_path, output_path=TFL_segmentation_path))
    print('TL/FL segmentation completed.')

    # make folder for TL/FL maximum projection
    if not os.path.exists(os.path.join(save_path, 'TFL_Maximum_Projection')):
        os.makedirs(os.path.join(save_path, 'TFL_Maximum_Projection'))
    TFL_Maximum_projection_path = os.path.join(save_path, 'TFL_Maximum_Projection')

    # multi-view maximum projection
    print(' ')
    print('Start multi-view projection.')
    preprocessing.tfl_maximum_projection.mutil_view_maximum_projection(in_path=TFL_segmentation_path,
                                                                       save_path=TFL_Maximum_projection_path)
    print('Multi-view maximum projection completed.')

    # AD Stanford subtyping
    print(' ')
    print('Start AD Stanford subtyping.')
    data_list, type_list = Stanford_Subtyping.predict_Subtyping.batch_Stanford_Sub(in_path=TFL_Maximum_projection_path,
                                                                                   save_path=save_path)
    print('AD Stanford Subtyping completed.')
    print('Pipeline finished.')
