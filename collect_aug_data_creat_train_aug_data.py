import numpy as np
import os
from imgaug import augmenters as iaa
import cv2
from sklearn import preprocessing


def creat_X_Y_aug_npy_data(X, Y):

    seq_1_0 = iaa.Affine(translate_percent={"x": 0.01, "y": 0.01})
    seq_1_1 = iaa.Affine(scale={"x": 0.9, "y": 0.9})
    seq_1_2 = iaa.Affine(scale={"x": 1.1, "y": 1.1})
    seq_1_3 = iaa.Affine(rotate=-5)
    seq_1_4 = iaa.Affine(rotate=5)

    seq_2 = iaa.Crop(percent=(0, 0.1))  # random crops

    ##  sigma在0~0.5间随机高斯模糊，且每张图纸生效的概率是0.5
    seq_3 = iaa.Sometimes(0.5,
                          iaa.GaussianBlur(sigma=(0, 0.5))
                          )

    seq_4 = iaa.ContrastNormalization((0.75, 1.5))

    seq_5 = iaa.Multiply((0.8, 1.2))

    seq_6 = iaa.Affine(
        translate_percent={"x": -0.01, "y": -0.01}
    )

    seq_list = []
    seq_list.append(seq_1_0)
    seq_list.append(seq_1_1)
    seq_list.append(seq_1_2)
    seq_list.append(seq_1_3)
    seq_list.append(seq_1_4)
    seq_list.append(seq_2)
    seq_list.append(seq_3)
    seq_list.append(seq_4)
    seq_list.append(seq_5)
    seq_list.append(seq_6)

    N = np.size(Y,0)
    X_aug = np.zeros((N*10, 128, 128, 3))
    Y_aug = np.zeros((N*10,1))
    for i in range(N):
        X_aug[i*10] = X[i]
        Y_aug[i*10] = Y[i]
        for j in range(1,10):
            #index_temp = random.randint(0,8)
            file_img_aug = seq_list[j].augment_images(np.uint8(X[i]))
            X_aug[i*10+j] = file_img_aug.astype('float32')
            Y_aug[i*10+j] = Y[i]

    return X_aug, Y_aug


def data_norm(X):
    X_norm = X
    for i in range(np.size(X,0)):
        X_norm[i] = X[i]/255
    return X_norm

def data_norm_zscore(X):
    X_norm = X
    for i in range(np.size(X,0)):
        #X_norm[i] = preprocessing.scale(X[i])
        X_norm[i,:,:,0] = preprocessing.scale(X[i,:,:,0])
        X_norm[i, :, :, 1] = preprocessing.scale(X[i, :, :, 1])
        X_norm[i, :, :, 2] = preprocessing.scale(X[i, :, :, 2])

    return X_norm



if __name__ == '__main__':

    ### load train data
    gulou_data = np.load('K:/2021fmri_work/moyamoya_mri_data/npy_data/gulou_data.npy.npz',allow_pickle=True)
    xuyi_data = np.load('K:/2021fmri_work/moyamoya_mri_data/npy_data/xuyi_data.npy.npz', allow_pickle=True)
    yifu_data = np.load('K:/2021fmri_work/moyamoya_mri_data/npy_data/yifu_data.npy.npz', allow_pickle=True)
    naoke_data = np.load('K:/2021fmri_work/moyamoya_mri_data/npy_data/naoke_data.npy.npz', allow_pickle=True)

    X_mix_gulou_train = gulou_data['arr_0']
    Y_mix_gulou_train = gulou_data['arr_1']
    X_mix_gulou_vali = gulou_data['arr_2']
    Y_mix_gulou_vali = gulou_data['arr_3']

    X_mix_xuyi_train = xuyi_data['arr_0']
    Y_mix_xuyi_train = xuyi_data['arr_1']
    X_mix_xuyi_vali = xuyi_data['arr_2']
    Y_mix_xuyi_vali = xuyi_data['arr_3']

    X_mix_yifu_train = yifu_data['arr_0']
    Y_mix_yifu_train = yifu_data['arr_1']
    X_mix_yifu_vali = yifu_data['arr_2']
    Y_mix_yifu_vali = yifu_data['arr_3']

    X_mix_naoke_train = naoke_data['arr_0']
    Y_mix_naoke_train = naoke_data['arr_1']
    X_mix_naoke_vali = naoke_data['arr_2']
    Y_mix_naoke_vali = naoke_data['arr_3']

    X_train = np.concatenate((X_mix_gulou_train, X_mix_xuyi_train), axis=0)
    X_train = np.concatenate((X_train, X_mix_yifu_train), axis=0)
    X_train = np.concatenate((X_train, X_mix_naoke_train), axis=0)

    X_vali = np.concatenate((X_mix_gulou_vali, X_mix_xuyi_vali), axis=0)
    X_vali = np.concatenate((X_vali, X_mix_yifu_vali), axis=0)
    X_vali = np.concatenate((X_vali, X_mix_naoke_vali), axis=0)

    Y_train = np.vstack((Y_mix_gulou_train, Y_mix_xuyi_train))
    Y_train = np.vstack((Y_train, Y_mix_yifu_train))
    Y_train = np.vstack((Y_train, Y_mix_naoke_train))

    Y_vali = np.vstack((Y_mix_gulou_vali, Y_mix_xuyi_vali))
    Y_vali = np.vstack((Y_vali, Y_mix_yifu_vali))
    Y_vali = np.vstack((Y_vali, Y_mix_naoke_vali))

    X_aug,Y_aug = creat_X_Y_aug_npy_data(X_train, Y_train)

    X_aug_norm = data_norm_zscore(X_aug)
    X_vali_norm = data_norm_zscore(X_vali)


    np.save(r'K:\2021fmri_work\moyamoya_mri_data\npy_data\x_train_aug.npy', X_aug_norm)
    np.save(r'K:\2021fmri_work\moyamoya_mri_data\npy_data\x_train_aug_label.npy', Y_aug)
    np.save(r'K:\2021fmri_work\moyamoya_mri_data\npy_data\x_vali.npy', X_vali_norm)
    np.save(r'K:\2021fmri_work\moyamoya_mri_data\npy_data\x_vali_label.npy', Y_vali)

    print("load data finished....")

