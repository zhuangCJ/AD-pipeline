import os
import json
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import xlwt
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "."))
import model_feature
import model_class


def Stanford_Sub(img_path_0, img_path_1, img_path_2):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    assert os.path.exists(img_path_0), "file: '{}' dose not exist.".format(img_path_0)
    img_0 = Image.open(img_path_0)
    img_0 = data_transform(img_0)
    img_0 = torch.unsqueeze(img_0, dim=0)

    assert os.path.exists(img_path_1), "file: '{}' dose not exist.".format(img_path_1)
    img_1 = Image.open(img_path_1)
    img_1 = data_transform(img_1)
    img_1 = torch.unsqueeze(img_1, dim=0)

    assert os.path.exists(img_path_2), "file: '{}' dose not exist.".format(img_path_2)
    img_2 = Image.open(img_path_2)
    img_2 = data_transform(img_2)
    img_2 = torch.unsqueeze(img_2, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model_0 = model_feature.resnet34(pretrained=True).to(device)
    model_0.aux_logits = False
    model_1 = model_feature.resnet34(pretrained=True).to(device)
    model_1.aux_logits = False
    model_2 = model_feature.resnet34(pretrained=True).to(device)
    model_2.aux_logits = False

    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0, std=0.2)

    model_3 = model_class.NeuralNet(510, 256, 128, 64, 32, 2).to(device)
    model_3.apply(init_normal)

    best_model_weight_0 = './weights_/best_model_0.pth'
    best_model_weight_1 = './weights_/best_model_1.pth'
    best_model_weight_2 = './weights_/best_model_2.pth'
    best_model_weight_3 = './weights_/best_model_3.pth'

    model_0.load_state_dict(torch.load(best_model_weight_0, map_location=device))
    model_1.load_state_dict(torch.load(best_model_weight_1, map_location=device))
    model_2.load_state_dict(torch.load(best_model_weight_2, map_location=device))
    model_3.load_state_dict(torch.load(best_model_weight_3, map_location=device))

    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()

    with torch.no_grad():
        # predict class
        pred_0 = model_0(img_0.to(device))
        pred_1 = model_1(img_1.to(device))
        pred_2 = model_2(img_2.to(device))
        pred_3 = torch.squeeze(model_3(pred_0, pred_1, pred_2)).cpu()

        predict = torch.softmax(pred_3, dim=0)
        predict_cla = torch.argmax(predict, dim=0).numpy()

        if not predict_cla:
            the_type = 'TAAD'
        else:
            the_type = 'TBAD'

    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], predict[i].numpy()))

    return the_type


def batch_Stanford_Sub(in_path, save_path):

    path_coronal = os.path.join(in_path, 'coronal')
    path_sagittal = os.path.join(in_path, 'sagittal')
    path_transverse = os.path.join(in_path, 'transverse')

    type_list = []
    name_list = os.listdir(path_coronal)
    name_list_2 = []
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet('Sheet1')

    for i, name in enumerate(name_list):
        the_type = Stanford_Sub(os.path.join(path_coronal, name), os.path.join(path_sagittal, name),
                                os.path.join(path_transverse, name))
        name_2 = name.replace('.png', '.nii.gz')

        type_list.append(the_type)
        name_list_2.append(name_2)

        worksheet.write(i, 0, name_2)
        worksheet.write(i, 1, the_type)

    workbook.save(os.path.join(save_path, 'AD_Stanford_Subtyping.xls'))

    return name_list_2, type_list
