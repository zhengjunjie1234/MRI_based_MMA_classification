import os
from imgaug import augmenters as iaa
import numpy as np
import cv2
import random


def creat_X_Y_aug_img_data(mmd_path, hc_path, resize):

    seq_1_1 = iaa.Affine(scale={"x": 0.9, "y": 0.9})
    seq_1_2 = iaa.Affine(scale={"x": 1.1, "y": 1.1})
    seq_1_3 = iaa.Affine(rotate=-5)
    seq_1_4 = iaa.Affine(rotate=5)## 水平翻转，0-1， 1，完全水平反转

    seq_2 = iaa.Crop(percent=(0, 0.1))  # random crops

    ##  sigma在0~0.5间随机高斯模糊，且每张图纸生效的概率是0.5
    seq_3 = iaa.Sometimes(0.5,
                          iaa.GaussianBlur(sigma=(0, 0.5))
                          )

    seq_4 = iaa.ContrastNormalization((0.75, 1.5))

    seq_5 = iaa.Multiply((0.8, 1.2))

    seq_6 = iaa.Affine(
        translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
    )

    seq_list = []
    seq_list.append(seq_1_1)
    seq_list.append(seq_1_2)
    seq_list.append(seq_1_3)
    seq_list.append(seq_1_4)
    seq_list.append(seq_2)
    seq_list.append(seq_3)
    seq_list.append(seq_4)
    seq_list.append(seq_5)
    seq_list.append(seq_6)

    files_mmd = os.listdir(mmd_path)
    N_mmd = len(files_mmd)
    X1 = np.zeros((N_mmd*10, resize, resize, 3))
    for i in range(N_mmd):
        files_tmp = os.path.join(mmd_path, files_mmd[i])
        file_img = cv2.imread(files_tmp)
        file_img_reszie = cv2.resize(file_img, (resize, resize))
        X1[i*10] = file_img_reszie / 255
        for j in range(1,10):
            index_temp = random.randint(0,9)
            file_img_aug = seq_list[index_temp].augment_images(file_img_reszie)
            X1[i*10+j] = file_img_aug / 255

    files_hc = os.listdir(hc_path)
    N_hc = len(files_hc)
    X2 = np.zeros((N_hc*10, resize, resize, 3))
    for i in range(N_hc):
        files_tmp = os.path.join(hc_path, files_hc[i])
        file_img = cv2.imread(files_tmp)
        file_img_reszie = cv2.resize(file_img, (resize, resize))
        X2[i*10] = file_img_reszie / 255
        for j in range(1,10):
            index_temp = random.randint(0,9)
            file_img_aug = seq_list[index_temp].augment_images(file_img_reszie)
            X2[i*10+j] = file_img_aug / 255

    X = np.vstack((X1, X2))
    Y = np.zeros((N_mmd*10 + N_hc*10, 2))
    Y[0:N_mmd*10 - 1, 0] = 1
    Y[N_mmd*10 - 1:, 1] = 1

    return X, Y


if __name__ == '__main__':
    print("finished......")