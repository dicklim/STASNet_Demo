import torch
import os,glob,sys
import pandas as pd
import random,csv
import argparse
import torchvision
from torch import nn,optim,Tensor
from torch.nn import functional as F

import torch.backends.cudnn as cudnn
import numpy as np
np.seterr(divide='ignore',invalid='ignore')

from getdataset import Getdata
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, average_precision_score,precision_score,f1_score,recall_score,roc_auc_score
from pycm import ConfusionMatrix

import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


def plot_confusion_matrix(cm, labels_name, title, target):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print(cm)
    conf_matrix = pd.DataFrame(cm, index=labels_name,
                               columns=labels_name)
    plt.subplots(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=False, annot_kws={'font': 'Arial', "size": 20}, cmap="Blues")
    plt.title(title, fontsize=50)
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(target, bbox_inches='tight', dpi=600, transparent=True, pad_inches=0)
    # plt.show()

    return


def merge_dataset(STAS_set, Normal_set, Tumor_set=[], balance=False):
    name2label = {'normal': 0, 'STAS': 1}
    print(len(STAS_set), len(Normal_set))
    if balance == True:
        Normal_set = random.sample(Normal_set, int(len(STAS_set) * 1))
        STAS_set = random.sample(STAS_set, len(Normal_set))
    print(len(STAS_set), len(Normal_set))
    label_STAS = [name2label['STAS']] * (len(STAS_set) + len(Tumor_set))
    label_Normal = [name2label['normal']] * len(Normal_set)
    All_figure = []
    All_figure.extend(STAS_set)
    All_figure.extend(Tumor_set)
    All_figure.extend(Normal_set)
    All_figure = np.array(All_figure)
    All_label = []
    All_label.extend(label_STAS)
    All_label.extend(label_Normal)
    All_label = np.array(All_label)

    return All_figure, All_label


def load_dataloader(figure, label, args):
    dataset = Getdata(figure, label)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchsize,
        num_workers=4,
        pin_memory=True
    )

    return dataloader


def balance_dataset(train_figure_STAS, train_figure_Normal, name2label):
    train_figure, train_label = [], []

    for STAS_item in train_figure_STAS:
        train_figure.append(STAS_item)
        train_label.append(name2label['STAS'])

    for Normal_item in train_figure_Normal:
        train_figure.append(Normal_item)
        train_label.append(name2label['normal'])

    return train_figure, train_label


def process_result(test_loss_temp, total_count, STAS_count, normal_count, correct_count, STAS_correct_count, normal_correct_count, temp_predict_list, temp_label_list, args):
    # print(1)
    STAS_accuracy = STAS_correct_count / STAS_count
    normal_accuracy = normal_correct_count / normal_count
    all_accuracy = correct_count / total_count
    # print(2)
    predict_list = torch.zeros_like(temp_predict_list)
    # print(3)

    label_list = torch.zeros_like(temp_label_list)
    # print(4)

    return test_loss_temp, all_accuracy, STAS_accuracy, normal_accuracy, predict_list, label_list


def test(gpu, test_model, weight, device, test_loader, last_epoch, criteon, args):
    total = 0
    correct = 0
    STAS_number = 0
    STAS_correct = 0
    normal_number = 0
    normal_correct = 0
    test_model.load_state_dict(torch.load(last_epoch))


    test_model.eval()
    # criteon = nn.CrossEntropyLoss()
    test_loss = 0
    predict_list = torch.empty(1).to(device)
    label_list = torch.empty(1).to(device)
    # with torch.no_grad():
    for step, data in enumerate(tqdm(test_loader), 0):
        points, target, _ = data
        # target = target.float()
        points, target = points.to(device), target.to(device)

        with torch.inference_mode():
            logits = test_model(points)
            # loss = criteon(logits, target)
            one_hot_target = F.one_hot(target, num_classes=2).type(torch.float)
            loss = criteon(logits, one_hot_target)


        test_loss += loss.type(torch.float)


        predict = logits.argmax(dim=1)
        predict_list = torch.cat((predict_list, predict), 0)

        # target = target.to('cpu')
        label_list = torch.cat((label_list, target), 0)

        total += torch.eq(target, target).type(torch.float).sum()
        STAS_number += torch.eq(target, 1).type(torch.float).sum()
        normal_number += torch.eq(target, 0).type(torch.float).sum()

        correct += torch.eq(predict, target).type(torch.float).sum()

        correct_matrix = target * torch.eq(predict, target)
        STAS_correct += torch.eq(correct_matrix, 1).type(torch.float).sum()
        normal_correct += torch.eq(predict, target).type(torch.float).sum() - torch.eq(correct_matrix, 1).type(torch.float).sum()

    # print(3, len(new_label_list))
    test_loss = test_loss / len(test_loader)
    # test_loss = torch.tensor(test_loss / (step + 1)).to(device)


    test_loss_temp, \
    all_accuracy, STAS_accuracy, normal_accuracy, \
    predict_list, label_list = process_result(
        test_loss,
        total, STAS_number, normal_number,
        correct, STAS_correct, normal_correct,
        predict_list, label_list, args)


    return test_loss_temp, all_accuracy, STAS_accuracy, normal_accuracy, predict_list, label_list


def train(gpu, device, DL_model, learning_rate,
          train_fig_list, train_label_list,
          test_fig_list, test_label_list,
          Sry_figure, Sry_label,
          Tai_figure, Tai_label,
          Xu_figure, Xu_label,
          Szl_indepent_figure, Szl_indepent_label,
          save_dir_path, save_head, model_save, figure_save, args,
          train_transform=False):

    temp_model_path = os.path.join(save_dir_path, 'Temp_Model')
    if not os.path.exists(temp_model_path):
        os.mkdir(temp_model_path)
    # if rank == 0:
    #     print('Congfig:\nDevice:%s\nBatch_Size:%0.4f\nLearning_Rate:%0.4f\nTransform:%s\nBenchMark:%s' %
    #           (gpu, args.batchsize, learning_rate, train_transform, cudnn.benchmark))
    #     print('='*30, save_head, '='*30)

    train_dataset = Getdata(train_fig_list, train_label_list, transform=train_transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        # shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_dataloader = load_dataloader(test_fig_list, test_label_list, args)
    Sry_test_dataloader = load_dataloader(Sry_figure, Sry_label, args)
    Tai_test_dataloader = load_dataloader(Tai_figure, Tai_label, args)
    Xu_test_dataloader = load_dataloader(Xu_figure, Xu_label, args)
    Szl_indepent_test_dataloader = load_dataloader(Szl_indepent_figure, Szl_indepent_label, args)

    # print('All loaded!!')

    DL_model = DL_model.to(device)
    DL_model = nn.SyncBatchNorm.convert_sync_batchnorm(DL_model)
    # print(gpu)
    # model.load_state_dict(torch.load('last_epoch.mdl'))
    # scaler = GradScaler(enabled=False)

    optimizer = optim.Adam(DL_model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(DL_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


    weight = train_dataset.__weight__().to(device)
    # criteon = nn.CrossEntropyLoss(weight=weight)
    # criteon = nn.CrossEntropyLoss()
    criteon = nn.BCEWithLogitsLoss(weight=weight)
    # print(weight)
    # sys.exit()
    best_tumor_acc = 0
    best_val_acc = 0
    best_test_loss = 520
    best_predict_list = []
    best_label_list = []
    loss_value_list = []
    val_acc_value_list = []

    for epoch in range(args.epochs):

        epoch_loss = 0
        print('epoch:', epoch, end='==>')

        start_time = time.time()
        for step, data in enumerate(tqdm(train_dataloader), 0):
            # if step % 100 == 0:
            #     print(step, end=' ', file=log_save)
            points, target, _ = data
            points, target = points.to(device), target.to(device)

            DL_model.train()
            logits = DL_model(points)
            one_hot_target = F.one_hot(target, num_classes=2).type(torch.float)

            loss = criteon(logits, one_hot_target)
            # print('epoch:',epoch,'loss:',loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

            epoch_loss += loss.type(torch.float)

        # print(epoch_loss)
        end_time = time.time()

        loss_value = epoch_loss / len(train_dataloader)
        # print('lv', loss_value, epoch_loss, loss)
        model_param_path = os.path.join(temp_model_path, save_head + 'last_epoch.mdl')
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        print("Finished, loss:%0.8f Time consume: %0.2f min Learning rate: %0.4f" % (loss_value, (end_time - start_time)/60, optimizer.state_dict()['param_groups'][0]['lr']))
        loss_value_list.append(loss_value)
        torch.save(DL_model.state_dict(), model_param_path)

        time.sleep(5)

        test_start_time = time.time()

        test_single_loss, \
        all_accuracy, STAS_correct_rate, normal_correct_rate,\
        predict_list, label_list = test(
                gpu,
                DL_model,
                weight,
                device,
                test_dataloader,
                model_param_path,
                criteon,
                args
            )

        test_end_time = time.time()

        print('Test STAS_acc:%0.8f, normal_acc:%0.4f, all_acc:%0.4f Test Time consume %0.4f min' % (
            STAS_correct_rate,
            normal_correct_rate,
            all_accuracy,
            (test_end_time - test_start_time) / 60
        ))


        val_acc_value_list.append(all_accuracy)
        if all_accuracy > best_val_acc:
            best_val_acc = all_accuracy
            print('=' * 15, "best acc found:", best_val_acc, '=' * 15)
            torch.save(DL_model.state_dict(), os.path.join(temp_model_path, save_head + 'best_acc.mdl'))
        if STAS_correct_rate > best_tumor_acc:
            best_tumor_acc = STAS_correct_rate
            torch.save(DL_model.state_dict(), os.path.join(temp_model_path, save_head + 'best_tumordetect.mdl'))
            print('=' * 15, "best tumor dect acc found:", best_tumor_acc, '=' * 15)
        if test_single_loss < best_test_loss:
            best_test_loss = test_single_loss
            best_predict_list = predict_list
            best_label_list = label_list
            print('=' * 15, "best loss value found:", best_test_loss, '=' * 15)
        torch.save(DL_model.state_dict(), os.path.join(model_save, save_head + 'Epo%03d_acc%0.4f.mdl' % (epoch, all_accuracy)))

        if (epoch + 1) % 20 == 0:
            print('===============Four dataset test start!! ================')
            new_start_time = time.time()

            Sry_test_loss, \
            Sry_all_accuracy, Sry_STAS_correct_rate, Sry_normal_correct_rate, \
            Sry_predict_list, Sry_label_list = test(
                    gpu,
                    DL_model,
                    weight,
                    device,
                    Sry_test_dataloader,
                    model_param_path,
                    criteon,
                    args
                )

            Tai_test_loss, \
            Tai_all_accuracy, Tai_STAS_correct_rate, Tai_normal_correct_rate, \
            Tai_predict_list, Tai_label_list = test(
                    gpu,
                    DL_model,
                    weight,
                    device,
                    Tai_test_dataloader,
                    model_param_path,
                    criteon,
                    args
                )

            Xu_test_loss, \
            Xu_all_accuracy, Xu_STAS_correct_rate, Xu_normal_correct_rate, \
            Xu_predict_list, Xu_label_list = test(
                    gpu,
                    DL_model,
                    weight,
                    device,
                    Tai_test_dataloader,
                    model_param_path,
                    criteon,
                    args
                )

            Szl_indepent_test_loss, \
            Szl_indepent_all_accuracy, Szl_indepent_STAS_correct_rate, Szl_indepent_normal_correct_rate, \
            Szl_indepent_predict_list, Szl_indepent_label_list = test(
                    gpu,
                    DL_model,
                    weight,
                    device,
                    Szl_indepent_test_dataloader,
                    model_param_path,
                    criteon,
                    args
                )


            new_end_time = time.time()
            print('Jiangsu Province:Test Loss:%0.8f, Test STAS_acc:%0.4f, normal_acc:%0.4f, all_acc:%0.4f' % (
                Sry_test_loss,
                Sry_STAS_correct_rate,
                Sry_normal_correct_rate,
                Sry_all_accuracy
            ))
            print('Taizhou:Test Loss:%0.8f, Test STAS_acc:%0.4f, normal_acc:%0.4f, all_acc:%0.4f' % (
                Tai_test_loss,
                Tai_STAS_correct_rate,
                Tai_normal_correct_rate,
                Tai_all_accuracy
            ))
            print('Xuzhou Province:Test Loss:%0.8f, Test STAS_acc:%0.4f, normal_acc:%0.4f, all_acc:%0.4f' % (
                Xu_test_loss,
                Xu_STAS_correct_rate,
                Xu_normal_correct_rate,
                Xu_all_accuracy
            ))
            print('Jiangsu Cancer:Test Loss:%0.8f, Test STAS_acc:%0.4f, normal_acc:%0.4f, all_acc:%0.4f' % (
                Szl_indepent_test_loss,
                Szl_indepent_STAS_correct_rate,
                Szl_indepent_normal_correct_rate,
                Szl_indepent_all_accuracy
            ))
            print('Time consume %0.2f min' % ((new_end_time - new_start_time)/60))


    # print(loss_value_list)
    loss_value_list = [idx.item() for idx in loss_value_list]
    val_acc_value_list = [idx.item() for idx in val_acc_value_list]


    print("best tumor acc found:%0.4f  ///  best acc found:%0.4f" % (
        best_tumor_acc, best_val_acc))
    # print(loss_value_list, val_acc_value_list, best_predict_list, best_label_list,sep='\n')
    best_predict_list = best_predict_list.cpu().tolist()
    best_label_list = best_label_list.cpu().tolist()
    # print('===' * 15)
    # print(loss_value_list, val_acc_value_list, best_predict_list, best_label_list, sep='\n')
    num_epoch = range(1, len(loss_value_list) + 1)
    plt.figure()
    plt.plot(num_epoch, loss_value_list, "r", label="Training loss")
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss function ", fontsize=14)
    plt.title('Training loss', fontsize=10)
    plt.legend()
    plt.savefig(os.path.join(figure_save, save_head + 'Loss_fn.jpg'))  # 保存图
    plt.close()

    plt.figure()
    plt.plot(num_epoch, val_acc_value_list, color="b", label="Validation acc")
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy Value', fontsize=14)
    plt.title('Validation acc', fontsize=10)
    plt.legend()
    plt.savefig(os.path.join(figure_save, save_head + "Accuracy.jpg"))  # 保存图
    plt.close()

    # create confusion matrix
    # best_label_list, best_predict_list = best_label_list.to('cpu'), best_predict_list.to('cpu')
    # print(best_label_list, best_predict_list)
    y_true = np.array(best_label_list).astype(int)
    y_pred = np.array(best_predict_list).astype(int)
    # print(len(y_true), len(y_pred))
    # print(y_true, y_pred)
    y_true[y_true > 1] = 1
    y_true[y_true < 0] = 0
    y_pred[y_pred > 1] = 1
    y_pred[y_pred < 0] = 0
    fig_target = os.path.join(os.path.join(figure_save, save_head + "Accuracy.tiff"))
    cm = confusion_matrix(y_true, y_pred)
    # print(best_label_list, best_predict_list)
    # print(cm)
    plot_confusion_matrix(cm, ['normal', 'STAS'], save_head + 'ConfusionMatrix', fig_target)
    New_cm = ConfusionMatrix(y_true, y_pred)
    # print(New_cm)
    TPR = np.array([New_cm.TPR_Macro])
    FPR = np.array([New_cm.FPR_Macro])
    # FPR_TPR['ACHNJMU'] = [TPR, FPR]

    # sys.exit()
    txt_target = os.path.join(os.path.join(figure_save, save_head + "ConfusionMatrixValue.txt"))
    file = open(txt_target, 'w+')
    print('TPR', TPR, 'FPR', FPR, '\n',
          '------Weighted------', '\n',
          'Weighted precision', precision_score(y_true, y_pred, average='weighted'), '\n',
          'Weighted recall', recall_score(y_true, y_pred, average='weighted'), '\n',
          'Weighted f1-score', f1_score(y_true, y_pred, average='weighted'), '\n',
          '------Macro------', '\n',
          'Macro precision', precision_score(y_true, y_pred, average='macro'), '\n',
          'Macro recall', recall_score(y_true, y_pred, average='macro'), '\n',
          'Macro f1-score', f1_score(y_true, y_pred, average='macro'), '\n',
          '------Micro------', '\n',
          'Micro precision', precision_score(y_true, y_pred, average='micro'), '\n',
          'Micro recall', recall_score(y_true, y_pred, average='micro'), '\n',
          'Micro f1-score', f1_score(y_true, y_pred, average='micro'), '\n',
          file=file)


    file.close()


    return loss_value_list, val_acc_value_list, best_predict_list, best_label_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default=1, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batchsize', default=128, type=int,
                        metavar='N',
                        help='number of batchsize')
    args = parser.parse_args()

    # model_name = 'ResNet18'  # DenseNet121 // ResNet18
    # model_name = 'MobileNet_V3'  # ConvNeXt 128//MobileNet_V3 320//Swin_Tiny
    # model_name = 'Swin_Tiny'
    model_name = 'DenseNet121'
    date = time.strftime("%m%d", time.localtime())
    K_fold_value = 3
    cudnn.benchmark = True
    gpu = torch.cuda.get_device_name(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 0.001
    train_transform = True  # 是否扩充数据
    tile_size = 256

    # 保存输出
    save_path = '/home/ssd_1T/Demo4_Classify_history'
    save_dir = date + '_' + model_name + '_K_fold_%d' % tile_size
    save_dir_path = os.path.join(save_path, save_dir)  # 保存文件根目录
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)
    figure_save = os.path.join(save_dir_path, 'TrainFigures')
    if not os.path.exists(figure_save):
        os.mkdir(figure_save)
    model_save = os.path.join(save_dir_path, 'Models')  # 模型保存
    if not os.path.exists(model_save):
        os.mkdir(model_save)
    print('Model:%s; tile size:%d' % (model_name, tile_size),
      sep='\n')

    name2label = {'normal': 0, 'STAS': 1}

    root = '/home/ssd_1T/Demo4_Classify_history/230201Dataset/'
    target_path = os.path.join(root, 'Dataset_%d' % tile_size)

    figure_STAS = glob.glob(os.path.join(target_path, 'STAS/Fig', '*.jpg'))
    figure_Normal = glob.glob(os.path.join(target_path, 'Normal', '*.jpg'))
    figure_Tumor = glob.glob(os.path.join(target_path, 'Tumor', '*.jpg'))
    figure_Tumor = random.sample(figure_Tumor, int(len(figure_Tumor) * 0.15))

    patient_list = list(csv.DictReader(open(os.path.join(root, '221020FengSelect', 'ALL.csv'))))
    slide = list(csv.DictReader(open(os.path.join(root, '221020FengSelect', 'JSSZL_STAS_id.csv'))))

    wuzhe_list = []
    test_list = []

    for patient in patient_list:
        patient['slide'] = []
        for item in slide:
            if item['pathid'] == patient['pathid']:
                slide_id = '%s%s' % (item['data_num'].rjust(2, '0'), item['slide_num'].rjust(2, '0'))
                patient['slide'].append(slide_id)
        if patient['test\\train'] == 'wuzhe':
            wuzhe_list.append(patient)
        elif patient['test\\train'] == 'test':
            test_list.append(patient)
        else:
            pass

    test_indepent_slide = []
    for sample in test_list:
        test_indepent_slide += sample['slide']

    Sry_STAS, Sry_Normal, Sry_Tumor = [], [], []  # Shenrenyi 03
    Tai_STAS, Tai_Normal, Tai_Tumor = [], [], []  # Taizhou 05
    Xu_STAS, Xu_Normal, Xu_Tumor = [], [], []  # Xuzhou 04
    Szl_indepent_STAS, Szl_indepent_Normal, Szl_indepent_Tumor = [], [], []
    Szl_STAS, Szl_Normal, Szl_Tumor = [], [], []

    for figure in figure_STAS:
        if figure.split(os.sep)[-1][0:2] == '05':
            Tai_STAS.append(figure)
        elif figure.split(os.sep)[-1][0:2] == '03':
            Sry_STAS.append(figure)
        elif figure.split(os.sep)[-1][0] == '4':
            Xu_STAS.append(figure)
        elif figure.split(os.sep)[-1][0:4] in test_indepent_slide:
            Szl_indepent_STAS.append(figure)
        else:
            Szl_STAS.append(figure)

    for figure in figure_Normal:
        if figure.split(os.sep)[-1][0:2] == '05':
            Tai_Normal.append(figure)
        elif figure.split(os.sep)[-1][0:2] == '03':
            Sry_Normal.append(figure)
        elif figure.split(os.sep)[-1][0] == '4':
            Xu_Normal.append(figure)
        elif figure.split(os.sep)[-1][0:4] in test_indepent_slide:
            Szl_indepent_Normal.append(figure)
        else:
            Szl_Normal.append(figure)

    for figure in figure_Tumor:
        if figure.split(os.sep)[-1][0:2] == '05':
            Tai_Tumor.append(figure)
        elif figure.split(os.sep)[-1][0:2] == '03':
            Sry_Tumor.append(figure)
        elif figure.split(os.sep)[-1][0] == '4':
            Xu_Tumor.append(figure)
        elif figure.split(os.sep)[-1][0:4] in test_indepent_slide:
            Szl_indepent_Tumor.append(figure)
        else:
            Szl_Tumor.append(figure)


    Sry_figure, Sry_label = merge_dataset(Sry_STAS, Sry_Normal)
    Tai_figure, Tai_label = merge_dataset(Tai_STAS, Tai_Normal)
    Xu_figure, Xu_label = merge_dataset(Xu_STAS, Xu_Normal)
    Szl_indepent_figure, Szl_indepent_label = merge_dataset(Szl_indepent_STAS, Szl_indepent_Normal)

    def STAS_count(sample):
        return int(sample['STAS'])

    wuzhe_list.sort(key=STAS_count)
    # print(len(wuzhe_list))
    group_conut = int(len(wuzhe_list) / K_fold_value)
    group_idx = [(val + 1) for val in range(K_fold_value) for i in range(group_conut)]
    group_idx = group_idx + [K_fold_value] * (len(wuzhe_list) - len(group_idx))

    wuzhe_list = np.array(wuzhe_list)
    group_idx = np.array(group_idx)

    print('Tile Summary:', 'All', 'Normal', 'STAS', sep='\t')
    print('Jiangsu Province', len(Sry_figure), len(Sry_Normal), len(Sry_STAS), sep='\t')
    print('Jiangsu Cancer independent', len(Szl_indepent_figure), len(Szl_indepent_Normal), len(Szl_indepent_STAS), sep='\t')
    print('Xuzhou Pei', len(Xu_figure), len(Xu_Normal), len(Xu_STAS), sep='\t')
    print('Taizhou Chinese', len(Tai_figure), len(Tai_Normal), len(Tai_STAS), sep='\t')
    print('==' * 30)
    print('Wuzhe tile count:', len(wuzhe_list))


    skf = StratifiedKFold(n_splits=K_fold_value, random_state=114514, shuffle=True)
    Round = 1
    for train_index, test_index in skf.split(wuzhe_list, group_idx):
        if model_name == 'ResNet18':
            # model = ResNet.ResNet18(num_classes=2)
            model = torchvision.models.resnet18(pretrained=True)
            model.fc = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 2)
            )
        elif model_name == 'DenseNet121':
            # model = DenseNet.densenet121(num_classes=2)
            # model = torchvision.models.densenet121(pretrained=True)
            model = torchvision.models.densenet121(weights='IMAGENET1K_V1')
            model.classifier = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 2)
            )
        # model.load_state_dict(torch.load('/home/ssd/Demo4_Classify_history/DenseNet121_1031/Models/Epo200_acc0.9287.mdl'))
        # elif model_name == 'ConvNeXt':
        #     model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)
        #     model.classifier = nn.Sequential(
        #         # LayerNorm2d((768,), eps=1e-06, elementwise_affine=True),
        #         nn.Flatten(start_dim=1, end_dim=-1),
        #         nn.Linear(768, 256),
        #         nn.Hardswish(),
        #         nn.Dropout(0.2),
        #         nn.Linear(256, 2)
        #     )
        elif model_name == 'MobileNet_V3':
            model = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
            model.classifier = nn.Sequential(
                                nn.Linear(960, 1280),
                                nn.Hardswish(),
                                nn.Dropout(0.2),
                                nn.Linear(1280, 256),
                                nn.Hardswish(),
                                nn.Dropout(0.2),
                                nn.Linear(256, 2)
                            )
        elif model_name == 'Swin_Tiny':
            model = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.DEFAULT)
            model.head = nn.Sequential(
                nn.Linear(768, 256),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(256, 2)
            )
        else:
            sys.exit('Wrong Model Name!')

        save_head = 'Round-%d_' % Round
        Round += 1

        patient_train, patient_test = wuzhe_list[train_index], wuzhe_list[test_index]
        train_slide = []
        test_slide = []
        for sample in patient_train:
            train_slide += sample['slide']
        for sample in patient_test:
            test_slide += sample['slide']
        # print(train_slide)
        # print(test_slide)
        train_figure_STAS = []
        # train_label_STAS = []
        train_figure_Normal = []
        # train_label_Normal = []
        test_figure = []
        test_label = []
        trash = []
        for STAS_item in Szl_STAS:
            slide_item = STAS_item.split(os.sep)[-1][0:4]
            if slide_item in train_slide:
                train_figure_STAS.append(STAS_item)
                # train_label_STAS.append(name2label['STAS'])
            elif slide_item in test_slide:
                test_figure.append(STAS_item)
                test_label.append(name2label['STAS'])
            else:
                # print(STAS_item)
                trash.append(STAS_item)

        for Tumor_item in Szl_Tumor:
            slide_item = Tumor_item.split(os.sep)[-1][0:4]
            if slide_item in train_slide:
                train_figure_STAS.append(Tumor_item)

            else:
                # print(STAS_item)
                trash.append(Tumor_item)

        for Normal_item in Szl_Normal:
            slide_item = Normal_item.split(os.sep)[-1][0:4]
            if slide_item in train_slide:
                train_figure_Normal.append(Normal_item)
                # train_label_Normal.append(name2label['normal'])
            elif slide_item in test_slide:
                test_figure.append(Normal_item)
                test_label.append(name2label['normal'])
            else:
                # print(STAS_item)
                trash.append(Normal_item)


        train_figure, train_label = merge_dataset(train_figure_STAS, train_figure_Normal, balance=True)

        train_label = np.array(train_label)
        test_label = np.array(test_label)

        # sys.exit()
        #
        train(gpu, device, model, lr,
              train_figure, train_label,
              test_figure, test_label,
              Sry_figure, Sry_label,
              Tai_figure, Tai_label,
              Xu_figure, Xu_label,
              Szl_indepent_figure, Szl_indepent_label,
              save_dir_path, save_head,
              model_save, figure_save, args,
              train_transform)



            # log_save.close()

if __name__ == '__main__':
    main()