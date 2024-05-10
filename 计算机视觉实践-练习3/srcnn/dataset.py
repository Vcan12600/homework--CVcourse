import os
import cv2
import random
import numpy as np


class MyData:
    def __init__(self, path):
        self.path = path
        self.img_list = os.listdir(path)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_item_path = os.path.join(self.path, img_name)
        img = cv2.imread(img_item_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img = img[:, :, 0]
        return img

    def __len__(self):
        return len(self.img_list)


class SRCNNDataset:
    def __init__(self, hr_list, lr_list):
        self.hr_list = hr_list
        self.lr_list = lr_list

    def __getitem__(self, index):
        input_low_resolution = self.lr_list[index]
        label_high_resolution = self.hr_list[index]
        return input_low_resolution, label_high_resolution

    def __len__(self):
        return len(self.hr_list)


# 数据增强
def data_augment(data_img, augment_kind):
    (w, h) = (data_img.shape[0], data_img.shape[1])
    center = (w / 2, h / 2)
    angle = 0
    if augment_kind == 1:
        angle = 90
    elif augment_kind == 2:
        angle = 180
    elif augment_kind == 3:
        angle = 270

    m = cv2.getRotationMatrix2D(center, angle, 1.0)  # 执行仿射变换（旋转）
    rotated = cv2.warpAffine(data_img, m, (h, w))
    return rotated


def data_augment2(data_img, augment_kind):
    scale = 1
    if augment_kind == 1:
        scale = 0.9
    elif augment_kind == 2:
        scale = 1.1
    elif augment_kind == 3:
        scale = 1.2

    return cv2.resize(data_img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


# size*size的patch
def get_patch(img_array, size):
    patch_list = []
    for n in range(len(img_array)):
        h = img_array[n].shape[0]
        step = int(h / size)
        for x in range(step):
            for y in range(step):
                temp_crop = img_array[n][size * x:size * (x + 1), size * y:size * (y + 1)]
                patch_list.append(temp_crop)
    return patch_list


img_data = MyData(r"E:\mindspore\srcnn\dataset\urban100")


# 将图像裁成正方形
def img_center_crop(image, crop_size):
    img_h = image.shape[0]
    img_w = image.shape[1]

    x_left = int(img_h / 2) - int(crop_size / 2)
    x_right = int(img_h / 2) + int(crop_size / 2)
    y_left = int(img_w/2) - int(crop_size / 2)
    y_right = int(img_w / 2) + int(crop_size / 2)
    img_crop = image[x_left:x_right, y_left:y_right]

    return img_crop


img_list = []
random.seed(22)
for data in img_data:
    kind = random.randint(1, 3)
    kind2 = random.randint(1, 3)
    if data.shape[0] > 640 and data.shape[1] > 640:
        result = img_center_crop(data, 640)
        img_list.append(result)

        result = data_augment(data, kind)
        result = img_center_crop(result, 640)
        img_list.append(result)

        result = data_augment2(data, kind2)
        if kind2 == 1:
            result = img_center_crop(result, 320)
            img_list.append(result)
        else:
            result = img_center_crop(result, 640)
            img_list.append(result)


# 生成低分辨率图
LR_list = []
for i in range(len(img_list)):
    LR = cv2.resize(img_list[i], dsize=None, fx=0.5, fy=0.5)
    LR_list.append(cv2.resize(LR, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))

size = 32
HR_patch_list = get_patch(img_list, size)
for i in range(len(HR_patch_list)):
    HR_patch_list[i] = HR_patch_list[i][6:size - 6, 6:size - 6]

dataset = SRCNNDataset(HR_patch_list, get_patch(LR_list, size))
