import os
import sys
import openslide
import matplotlib.pyplot as plt
import matplotlib.path as mplp
import shapely
from shapely.geometry import Polygon as Poly
from PIL import Image
import numpy as np
import json
import cv2
import torch
import time
import imghdr
from tqdm import tqdm
from apeer_ometiff_library import io as apeerIO


# from color_OME_WSI_StainTool import color_norm, White_Norm


def read_OMEtiff(WSIpath):
    OME = apeerIO.read_ometiff(WSIpath)
    # (1, 1, w, h, 3)
    tiff = OME[0].squeeze()
    # ycbcr = cv2.cvtColor(tiff, cv2.COLOR_YCR_CB2RGB)
    crcb_tiff = tiff[:, :, [0, 2, 1]]  # original is ycbcr ,change to ycrcb
    # print('OME loaded')
    ycrcb = cv2.cvtColor(crcb_tiff, cv2.COLOR_YCrCb2RGB)

    return ycrcb


def read_SVS(WSIpath):
    slide = openslide.open_slide(WSIpath)  # 读入图片（）
    w, h = slide.level_dimensions[0]  # 最高倍下的宽高
    region = slide.read_region((0, 0), 0, (w, h)).convert('RGB')
    # 查了一下这个代码出来的是PIL的RGBA格式图像，转化成RGB格式，先读取，再数组化
    region = np.asarray(region)
    # cv读出来也是array，这儿的array正好是RGB
    slide.close()
    # print('SVS loaded')

    return region


def get_WSI(WSIpath):
    if WSIpath.endswith('.svs'):
        print('Loading SVS ==> ', end='')
        Slice = read_SVS(WSIpath)
        print('SVS loaded!')
    elif WSIpath.endswith('.ome.tif'):
        print('Loading OME tiff ==> ', end='')
        Slice = read_OMEtiff(WSIpath)
        print('OME tiff loaded')

    return Slice


def read_Geojson(geojson_path):
    jsondata = json.load(open(geojson_path))
    polygon_list = jsondata['features']
    polygons = []
    for item in polygon_list:
        type = item['geometry']['type']
        if type == 'Polygon':
            polygon = np.array(item['geometry']['coordinates']).squeeze()
            polygons.append(polygon)
        else:
            print(type, 'Multi polygon found!', geojson_path.split(os.sep)[-1].split('.')[0])
            sys.exit()

    return polygons


def check_box(polygons, tile_box):
    is_tumor = False
    tile_box = Poly(np.array(tile_box))
    shape_polygon = [Poly(item.squeeze()) for item in polygons]

    for polygon in shape_polygon:
        # poly_path = mplp.Path(polygon.squeeze())
        # is_tumor = poly_path.contains_point(tile_center)
        is_tumor = tile_box.intersects(polygon)
        # print(is_tumor)
        if is_tumor == True:  # Tumor
            break

    return is_tumor


def White_Valid(tile, threshold=0.75):
    patch = np.array(tile)
    h, w, _ = patch.shape
    white_threshold = h * w * threshold
    # print(h * w)
    tempr = patch[:, :, 0] > 220
    tempg = patch[:, :, 1] > 220
    tempb = patch[:, :, 2] > 220
    temp = tempr * tempg * tempb
    # 三个都是True才可以是True
    if np.sum(temp == 1) > white_threshold:
        # 是白色，True，不是白色，False
        return True
    else:
        return False


def cut_WSI(WSIpath, geojson_path):
    polygons = read_Geojson(geojson_path)
    Slice = get_WSI(WSIpath)
    img = Image.fromarray(Slice)
    fig_list = []
    white_list = []
    tumor_list = []
    # normal_list = []
    weight = 256
    height = 256
    w_num = int(img.size[0] / weight)
    h_num = int(img.size[1] / height)
    for j in range(h_num):
        for i in range(w_num):
            box = (weight * i, height * j, weight * (i + 1), height * (j + 1))
            tile_box = [[weight * i, height * j],
                        [weight * i, height * (j + 1)],
                        [weight * (i + 1), height * (j + 1)],
                        [weight * (i + 1), height * j]]

            # center = (weight * (i + 0.5), height * (j + 0.5))
            if check_box(polygons, tile_box) == True:
                tumor_list.append((998, (i, j)))
            else:
                region = img.crop(box)
                if White_Valid(region) == True:
                    white_list.append((999, (i, j)))
                else:
                    fig_tmp = [region, (i, j)]
                    fig_list.append(fig_tmp)

    return Slice, fig_list, white_list, tumor_list, (w_num, h_num)


if __name__ == '__main__':
    geojson_path = '/home/hdd/TCGAdata/TumorLabel/TCGA-64-5778-01Z-00-DX1.96C39819-8A65-4651-BE83-39959F6FAD05.json'
    svs_path = '/home/hdd/TCGAdata/TCGA_Stage_I_dataset/Raw/TCGA-64-5778-01Z-00-DX1.96C39819-8A65-4651-BE83-39959F6FAD05.svs'
    fig_list, white_list, tumor_list, (w_num, h_num) = cut_WSI(svs_path, geojson_path)
    print(len(fig_list))
    print(len(white_list))
    print(len(tumor_list))
    print((w_num, h_num))

