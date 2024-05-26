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
from torchvision.transforms import transforms as T
from getWSI import cut_WSI

# x_transform = T.Compose([
#     T.ToTensor(),
#     # 标准化至[-1,1],规定均值和标准差
#     T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])#torchvision.transforms.Normalize(mean, std, inplace=False)
# ])

class Getdata(Dataset):
    def __init__(self, fig_list):
        self.fig_list = fig_list
        self.data, self.positions, self.labels = self.loaddata(self.fig_list)


    def loaddata(self, fig_list):
        data, positions, labels = [], [], []
        for item in fig_list:
            fig = np.array(item[0])
            data.append(fig)
            position = item[1]
            positions.append(position)
            labels.append(0)

        return data, positions, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        image = np.transpose(img)
        # label = self.name2label[label_index]  # 这儿是把字符串的STAS和Normal标签转成0和1
        image = torch.from_numpy(image.astype(np.float32))
        # image = torch.from_numpy(img.astype(np.float32))
        position = self.positions[index]
        label = self.labels[index]

        return image, position, label

if __name__ == '__main__':
    geojson_path = '/home/hdd/TCGAdata/TumorLabel/TCGA-64-5778-01Z-00-DX1.96C39819-8A65-4651-BE83-39959F6FAD05.json'
    svs_path = '/home/hdd/TCGAdata/TCGA_Stage_I_dataset/Raw/TCGA-64-5778-01Z-00-DX1.96C39819-8A65-4651-BE83-39959F6FAD05.svs'
    Slice, fig_list, white_list, tumor_list, (w_num, h_num) = cut_WSI(svs_path, geojson_path)

    dataset = Getdata(fig_list)

    print('Data Loaded!!')
    # print(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=40,
        shuffle=True,
        num_workers=16,
        pin_memory=True
    )
    print(dataloader)
    print(len(dataloader))
    for step, data in enumerate(dataloader,0):
        img, position, label = data
        print(position)
        # break
        if step >= 2:
            break
    #         print(points.shape, target)
        utils.save_image(img/255.0, '/home/hdd/Temp/%s.jpg' % (step), nrow = 10)
    #
    #         input_tensor = points.clone().detach()
    #         # 到cpu
    #         input_tensor = input_tensor.to(torch.device('cpu'))
    #         # 反归一化
    #         # input_tensor = unnormalize(input_tensor)
    #         # 去掉批次维度
    #         input_tensor = input_tensor.squeeze()
    #         print(input_tensor[1].shape)
    #         # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为numpy
    #         image1 = input_tensor[1].permute(1, 2, 0)
    #         image1 = np.uint8(image1.numpy())  # Image接受的图像格式必须为uint8，否则就会报错
    #
    #         print(input_tensor.shape)
    #         print(image1.shape)
    #         # image.show()
    #         # image1.save("gray.jpg")
    #         # 转成pillow
    #         im = Image.fromarray(image1)
    #         im.save('./TEST/%s%ssingle.jpg' % (step, epoch))
    #
    #     # 复制代码
    #         if step>=3:
    #             break


