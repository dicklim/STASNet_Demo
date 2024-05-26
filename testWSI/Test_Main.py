import os,glob,time
import sys
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
# from model import ResNet
# import staintools
# from Pre01_getWSI import get_WSI
from getWSI import cut_WSI, read_Geojson
# from Pre02_color_norm import color_norm
# from Pre03_trim_fig import cut_data
# from Pre04_getdatase import Getdata
from getdataset import Getdata
# from Predict_visi import predict_to_csv, percentage, merge_picture
from Predict_visi import predict_to_csv, merge_picture, predict_to_WSI

def test_WSI(WSIpath, geojson_path, target_head):
    name2label = {
        'Normal': 0,
        'STAS': 1,
        'Tumor': 998,
        'White': 999
    }
    color_map = {1: [255, 0, 0],  # 深紫色LPA
                 0: [255, 255, 255],
                 998: [0, 255, 127],
                 999: [255, 255, 255]}

    device = torch.device('cuda')
    torch.cuda.set_device(0)
    print('Use: ', torch.cuda.get_device_name())

    model_root = '/home/ssd_1T/Demo4_Classify_history/0204_MobileNet_V3_K_fold_128/Models/Round-2_Epo014_acc0.9434.mdl'
    # model = torchvision.models.resnet18(pretrained=True)
    # model.fc = nn.Sequential(
    #     nn.Linear(512, 1024),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(0.5),
    #     nn.Linear(1024, 256),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(0.5),
    #     nn.Linear(256, 2)
    # )
    model = torchvision.models.mobilenet_v3_large()
    model.classifier = nn.Sequential(
        nn.Linear(960, 1280),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(1280, 256),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(256, 2)
    )
    model = model.to(device)

    print('Using:', model_root.split(os.sep))

    checkpoint = torch.load(model_root)
    # print(checkpoint.keys())
    # sys.exit()
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})

    model.eval()

    Slice, fig_list, white_list, tumor_list, fig_shape = cut_WSI(WSIpath, geojson_path)

    # white_list, normal_list, tumor_list, fig_shape = cut_data(Slice, geojson_path)

    dataset = Getdata(fig_list)
    print('fig shape:', fig_shape, "tiles = ", fig_shape[0] * fig_shape[1])
    print('white %d / tumor %d / normal %d' % (len(white_list), len(tumor_list), len(fig_list)))
    print('Svs loaded')
    print('Data Loaded!!')
    # print(dataset)

    batch_size = 90
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )
    predict_list = []
    predict_cls_list = []
    for step, data in enumerate(tqdm(dataloader), 0):
        tup_img, position, _ = data
        img = tup_img.to(device)
        # print(img)
        # print(position)
        # torch.Size([90, 256, 256, 3])
        # print(img.shape)
        # img = img.permute(0, 3, 1, 2)
        # torch.Size([90, 3, 256, 256])
        # print(img.shape)
        w, h = position
        w = w.numpy()
        h = h.numpy()
        # print(w, h)
        position_list = []
        for i in range(len(w)):
            position_list.append((w[i], h[i]))
        # print(position_list)

        with torch.no_grad():
            logits = model(img)
            # print(logits)

            predict = logits.softmax(dim=1).cpu().numpy()  # 计算softmax，即该图片属于各类的概率
            predict_cls = logits.argmax(dim=1).cpu().numpy()
        # print(predict_normed.shape, k.shape)
        predict_temp = []
        # print(predict)
        for idx in range(len(predict_cls)):
            predict_temp.append([predict[idx].tolist(), position_list[idx]])
        # print(predict_temp)
        predict_list.extend(predict_temp)
        # print(predict_list)
        fig_predict = []
        for idx in range(len(predict_cls)):
            fig_predict.append((predict_cls[idx], position_list[idx]))
        predict_cls_list.extend(fig_predict)
    # print(len(predicet_list), predicet_list)
    # print(predict_normed_list[0:2])
    predict_to_csv(predict_list, target_head=target_head)
    total_list = white_list + tumor_list + predict_cls_list
    label_map = merge_picture(fig_shape, total_list, color_map, target_head=target_head, size=10)
    predict_to_WSI(Slice, label_map, Slice_k=0.1, target_head=target_head)

if __name__ == '__main__':


    root = '/home/hdd_SkyHawk/TCGAdata'

    target = '/home/hdd_SkyHawk/TCGAdata/240526_STAS_Pred/'
    if not os.path.exists(target):
        os.mkdir(target)

    # label_list_temp = glob.glob(os.path.join(root, 'TumorLabel', '*.json'))
    label_list_temp = glob.glob(os.path.join(root, 'StageI_json', '*.geojson'))
    # label_list_temp = glob.glob(os.path.join(root, '*.geojson'))

    label_list = [label for label in label_list_temp if len(read_Geojson(label)) > 0]
    title_list = [file_name.split(os.sep)[-1].split('.')[0] for file_name in label_list]

    SVS_list_temp = glob.glob(os.path.join(root, '*.svs'))
    SVS_list = [SVS for SVS in SVS_list_temp if SVS.split(os.sep)[-1].split('.')[0] in title_list]

    # print(label_list)
    # print(len(SVS_list))

    # sys.exit()

    for WSI_path in SVS_list:
        WSI_id = WSI_path.split(os.sep)[-1].split('.')[0]
        for label_path in label_list:
            label_id = label_path.split(os.sep)[-1].split('.')[0]
            if WSI_id == label_id:

                print('=' * 15, '!!%s Start!!' % WSI_id, '=' * 15)
                start_time = time.time()
                target_head = os.path.join(target, "%s_Pred" % WSI_id)
                test_WSI(WSI_path, label_path, target_head)
                end_time = time.time()
                print('=' * 15, '!!%s  Stop!! Time consume: %0.2f min' %  (WSI_id, (end_time - start_time)/60), '=' * 15)
                break
                # print(target_head)