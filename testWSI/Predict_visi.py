import numpy as np
import PIL.Image as Image
import gc
import csv
import cv2



def predict_to_csv(predict_list, target_head = './test'):
    '''

    :param predict_normed_list:a list with predict count of five type
    :return:
    '''
    csvFile = open(target_head+'.csv', "w")  # 创建csv文件
    writer = csv.writer(csvFile)  # 创建写的对象
    # 先写入columns_name

    writer.writerow(['0', '1', 'i', 'j'])  # 写入列的名称

    for probability in predict_list:
        predict_prob = probability[0]
        position = probability[1]
        # print(predict_prob, position)
        # predict_prob = np.round(probability[0], 3)
        writer.writerow([predict_prob[0], predict_prob[1], position[0], position[1]])
    csvFile.close()
    print('Predicet csv saved at ', csvFile)

    return

def merge_picture(shape, total_img_list, colormap, target_head = './test', size = 2):
    """

    :param shape: (w_num, h_num)
    :param total_img_list: [(idx, (w, h))
    :param target_path:
    :param size: width of each label
    :return:
    """
    w = shape[0] * size
    h = shape[1] * size
    channels = 3
    dst = np.zeros((h, w, channels), np.uint8)
    print('Start merge')
    for i in total_img_list:
        label_idx = i[0]
        # print(label_idx)
        color = colormap[label_idx]
        posi = i[1]

        tile = np.zeros([size, size, 3], np.uint8)
        tile[:, :, 0] = np.ones([size, size]) * color[0]
        tile[:, :, 1] = np.ones([size, size]) * color[1]
        tile[:, :, 2] = np.ones([size, size]) * color[2]
        # h, w
        # print(dst.shape, tile.shape)
        dst[posi[1] * size : (posi[1] + 1) * size, posi[0] * size: (posi[0] + 1) * size, :] = tile

    cv_label = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    # label_map = Image.fromarray(dst)
    # target_file = target_head+'.tiff'
    # label_map.save(target_file)
    # print('Result saved at ', target_file)

    return cv_label


def predict_to_WSI(Slice, label_map, Slice_k=0.1, target_head='./test'):

    weight, height = Slice.shape[0], Slice.shape[1]
    w, h = int(weight*Slice_k), int(height*Slice_k)
    # print(w,h)

    Slice_tmp = cv2.cvtColor(cv2.resize(Slice, (h, w)), cv2.COLOR_RGB2BGR)

    del Slice
    gc.collect()

    combine = cv2.addWeighted(Slice_tmp, 0.5, cv2.resize(label_map, (h, w), interpolation=cv2.INTER_NEAREST), 0.5, 0)

    cv2.imwrite(target_head + '_Origin.png', Slice_tmp)
    cv2.imwrite(target_head + '_Merge.png', combine)
    cv2.imwrite(target_head + '_Label.png', cv2.resize(label_map, (h, w), interpolation=cv2.INTER_NEAREST))

    del combine
    gc.collect()

    return