import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import os,glob
import random,csv
from torchvision import utils
from torch.utils.data import Dataset
# import Augmentation.myTransforms as myTransforms
# from torchvision.transforms import transforms as T
import Augmentation.dick_transforms as dick_T
from random import sample

# x_transform = T.Compose([
#     T.ToTensor(),
#     # 标准化至[-1,1],规定均值和标准差
#     T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])#torchvision.transforms.Normalize(mean, std, inplace=False)
# ])

class Getdata(Dataset):
    def __init__(self, fig_list, label_list, transform=False, gamma=False):
        # self.fig_list = self.load_figure(fig_list)
        self.fig_list = fig_list
        self.label_list = label_list
        self.gamma = gamma
        if transform == False:
            self.transform = None
        else:
            # 重写的transform，基本上原本的transform里都可以查到用法
            self.transform =dick_T.Compose([
                dick_T.Resize([256, 256]),  # 适应后续的多分辨率
                dick_T.RandomChoice([
                    dick_T.HEDJitter(theta=0.05),
                    dick_T.RandomGaussBlur(radius=[0.5, 1.5]),
                    dick_T.RandomAffineCV2(alpha=0.1)
                    # dick_T.RandomElastic(alpha=2, sigma=0.06)
                ]),
                dick_T.RandomChoice([
                    dick_T.AutoRandomRotation(),
                    dick_T.RandomHorizontalFlip(),
                    dick_T.RandomVerticalFlip()],)
            ])
        self.name2label = {'normal': 0, 'STAS': 1}
        self.label = ['normal', 'STAS']

    # def load_figure(self, path_list):
    #     image_list = []
    #     print('loading,figure')
    #     for fig_path in tqdm(path_list):
    #         image = Image.open(fig_path)
    #         image_list.append(image)
    #     print('Image Loaded!')
    #
    #     return image_list

    def gamma_arguement(self, image):
        down = 0.5
        up = 1.5
        gamma = down + (up - down) * np.random.random()
        invGamma = 1.0 / gamma
        table = []
        for i in range(256):
            table.append(((i / 255.0) ** invGamma) * 255)
        table = np.array(table).astype("uint8")
        return cv2.LUT(image, table)

    def __weight__(self):


        print(self.label_list, self.label_list)
        STAS_count = (self.label_list == self.name2label['STAS']).sum()
        # STAS_count = self.label_list.count(self.name2label['NucleusDivision'])
        Samples_Count = self.__len__()
        Class_Count = len(self.label)  # 2, 'Cell','NucleusDivision'

        # print(STAS_count, Samples_Count, Class_Count)

        STAS_Weight = Samples_Count / (Class_Count * STAS_count)
        Normal_Weight = Samples_Count / (Class_Count * (Samples_Count - STAS_count))

        weight = torch.tensor([Normal_Weight, STAS_Weight], dtype=torch.float32)

        return weight

    def __len__(self):
        return len(self.fig_list)

    def __getitem__(self, index):
        img_index = self.fig_list[index]
        image = Image.open(img_index)
        if self.transform == None:
            image = np.array(image)
        else:
            image = np.array(self.transform(image))
        # label_index = self.label_list[index]

        if self.gamma == True:
            image = self.gamma_arguement(image)

        if list(image.shape)[0] >= 256:
            image = cv2.resize(image, (256, 256))
        image = np.transpose(image)
        # label = self.name2label[label_index]
        label = self.label_list[index]
        image = torch.from_numpy(image.astype(np.float32))
        # label = torch.from_numpy(np.array([label]).astype(np.int64))

        return image, label, img_index

if __name__ == '__main__':

    name2label = {'Normal': 0, 'STAS': 1}
    traget_path = '/home/hdd/labelme/220903Dataset/Tiles/220905TempDataset'
    figure_Cell = glob.glob(os.path.join(traget_path, 'Normal', '*.jpg'))
    label_STAS = [name2label['Normal']] * len(figure_Cell)
    figure_NucleusDivision = glob.glob(os.path.join(traget_path, 'STAS/Fig', '*.jpg'))
    label_NucleusDivision = [name2label['STAS']] * len(figure_NucleusDivision)
    All_figure = []
    All_figure.extend(figure_NucleusDivision)
    All_figure.extend(figure_Cell)
    # print(len(All_figure))
    All_figure = np.array(All_figure)
    All_label = []
    All_label.extend(label_NucleusDivision)
    All_label.extend(label_STAS)
    All_label = np.array(All_label)
    # print(len(All_label))

    dataset = Getdata(All_figure, All_label, transform=True, gamma=True)
    print(len(dataset))
    print('Data Loaded!!')
    print(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=5,
        shuffle=False,
        num_workers=16
    )
    print(dataloader)
    print(len(dataloader))
    for epoch in range(20):
        for step, data in enumerate(dataloader, 0):
            points, target, mskname = data
            utils.save_image(points/255, './TEST/%s%s.jpg' % (step, epoch))
            # print(points.size, target)
            # points = points.squeeze()
            # print(points.shape)
            # points = points.numpy()
            # RGB转BRG
            # im = Image.fromarray(np.uint8(points))
            # im.save('./TEST/%s%s.jpg' % (step, epoch))
            # utils.save_image(points, './TEST/%s%s.jpg' % (step, epoch))

            if step >= 0:
                break

        # print(points, target, mskname)
        # print(points[0], target[0])
        # break

