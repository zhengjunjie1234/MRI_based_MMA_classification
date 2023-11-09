import os.path
import cv2
import numpy as np
import pandas as pd

from CNN_mmd_cross_validation import creat_X_Y_aug_img_data, model_fit_result, multi_slice_data, model_simple_build, \
    creat_X_Y


def creat_mix_data(mmd_path,hc_path,resize):
    mmd_path1 = os.path.join(mmd_path,'pre')
    mmd_path2 = os.path.join(mmd_path, 'mid')
    mmd_path3 = os.path.join(mmd_path, 'post')

    hc_path1 = os.path.join(hc_path, 'pre')
    hc_path2 = os.path.join(hc_path, 'mid')
    hc_path3 = os.path.join(hc_path, 'post')

    files_mmd1 = os.listdir(mmd_path1)
    files_mmd2 = os.listdir(mmd_path2)
    files_mmd3 = os.listdir(mmd_path3)

    files_hc1 = os.listdir(hc_path1)
    files_hc2 = os.listdir(hc_path2)
    files_hc3 = os.listdir(hc_path3)


    N_mmd = len(files_mmd1)
    X1 = np.zeros((N_mmd, resize, resize, 3))
    Y1 = np.zeros((N_mmd,1))
    for i in range(N_mmd):
        files_tmp = os.path.join(mmd_path1, files_mmd1[i])
        file_img = np.load(files_tmp,allow_pickle=True)
        file_img_reszie = cv2.resize(file_img, (resize, resize))
        X1[i,:,:,0] = file_img_reszie

        files_tmp = os.path.join(mmd_path2, files_mmd2[i])
        file_img = np.load(files_tmp,allow_pickle=True)
        file_img_reszie = cv2.resize(file_img, (resize, resize))
        X1[i,:,:,1] = file_img_reszie

        files_tmp = os.path.join(mmd_path3, files_mmd3[i])
        file_img = np.load(files_tmp,allow_pickle=True)
        file_img_reszie = cv2.resize(file_img, (resize, resize))
        X1[i, :, :, 2] = file_img_reszie
        Y1[i] = 1



    N_hc = len(files_hc1)
    X2 = np.zeros((N_hc, resize, resize, 3))
    Y2 = np.zeros((N_hc, 1))
    for i in range(N_hc):
        files_tmp = os.path.join(hc_path1, files_hc1[i])
        file_img = np.load(files_tmp,allow_pickle=True)
        file_img_reszie = cv2.resize(file_img, (resize, resize))
        X2[i,:,:,0] = file_img_reszie

        files_tmp = os.path.join(hc_path2, files_hc2[i])
        file_img = np.load(files_tmp, allow_pickle=True)
        file_img_reszie = cv2.resize(file_img, (resize, resize))
        X2[i,:,:,1] = file_img_reszie

        files_tmp = os.path.join(hc_path3, files_hc3[i])
        file_img = np.load(files_tmp,allow_pickle=True)
        file_img_reszie = cv2.resize(file_img, (resize, resize))
        X2[i,:,:,2] = file_img_reszie
        Y2[i] = 0

    X = np.vstack((X1, X2))
    Y = np.vstack((Y1, Y2))
    #Y = np.zeros((N_mmd + N_hc, 2))
    #Y[0:N_mmd - 1, 0] = 1
    #Y[N_mmd - 1:, 1] = 1

    return X, Y


def split_data(X,Y):
    n = np.size(Y,0)
    index_all = np.arange(n)
    n_vali = 2*n//10
    np.random.shuffle(index_all)

    index = index_all[0:n_vali]
    index_out = index_all[n_vali:]
    X_train = X[index_out]
    Y_train = Y[index_out]
    X_vali = X[index]
    Y_vali = Y[index]

    return X_train,Y_train, X_vali, Y_vali,index


if __name__ == '__main__':

    ### test data
    ### gulou data:
    mmd_dir_gulou = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\gulou\mmd"
    hc_dir_gulou = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\gulou\hc"

    X_mix_gulou, Y_mix_gulou = creat_mix_data(mmd_dir_gulou, hc_dir_gulou, 128)
    X_mix_gulou_train, Y_mix_gulou_train,X_mix_gulou_vali, Y_mix_gulou_vali,vali_index_gulou = split_data(X_mix_gulou, Y_mix_gulou)

    np.savez(r'K:\2021fmri_work\moyamoya_mri_data\npy_data\gulou_data.npy', X_mix_gulou_train, Y_mix_gulou_train,X_mix_gulou_vali, Y_mix_gulou_vali,vali_index_gulou)

    print("finished gulou data prepare...")


    ### junzong data:
    mmd_dir_junzong = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\junzong\mmd"
    hc_dir_junzong = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\junzong\hc"

    X_mix_junzong, Y_mix_junzong = creat_mix_data(mmd_dir_junzong, hc_dir_junzong, 128)

    np.savez(r'K:\2021fmri_work\moyamoya_mri_data\npy_data\junzong_data.npy', X_mix_junzong, Y_mix_junzong)

    print("finished junzong data prepare...")

    ### luan data:
    mmd_dir_luan = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\luan\mmd"
    hc_dir_luan = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\luan\hc"

    X_mix_luan, Y_mix_luan = creat_mix_data(mmd_dir_luan, hc_dir_luan, 128)

    np.savez(r'K:\2021fmri_work\moyamoya_mri_data\npy_data\luan_data.npy', X_mix_luan, Y_mix_luan)
    print("finished luan data prepare...")

    ### xuyi data
    mmd_dir_xuyi = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\xuyi\mmd"
    hc_dir_xuyi = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\xuyi\hc"

    X_mix_xuyi, Y_mix_xuyi = creat_mix_data(mmd_dir_xuyi, hc_dir_xuyi, 128)
    X_mix_xuyi_train, Y_mix_xuyi_train, X_mix_xuyi_vali, Y_mix_xuyi_vali, vali_index_xuyi = split_data(X_mix_xuyi,
                                                                                                            Y_mix_xuyi)

    np.savez(r'K:\2021fmri_work\moyamoya_mri_data\npy_data\xuyi_data.npy', X_mix_xuyi_train, Y_mix_xuyi_train, X_mix_xuyi_vali, Y_mix_xuyi_vali, vali_index_xuyi)

    print("finished xuyi data prepare...")

    ### yifu data
    mmd_dir_yifu = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\yifu\mmd"
    hc_dir_yifu = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\yifu\hc"

    X_mix_yifu, Y_mix_yifu = creat_mix_data(mmd_dir_yifu, hc_dir_yifu, 128)
    X_mix_yifu_train, Y_mix_yifu_train, X_mix_yifu_vali, Y_mix_yifu_vali, vali_index_yifu = split_data(X_mix_yifu,
                                                                                                       Y_mix_yifu)

    np.savez(r'K:\2021fmri_work\moyamoya_mri_data\npy_data\yifu_data.npy', X_mix_yifu_train, Y_mix_yifu_train, X_mix_yifu_vali, Y_mix_yifu_vali, vali_index_yifu)
    print("finished yifu data prepare...")

    ### ertong
    mmd_dir_ertong = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\ertong\mmd"
    hc_dir_ertong = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\ertong\hc"

    X_mix_ertong, Y_mix_ertong = creat_mix_data(mmd_dir_ertong, hc_dir_ertong, 128)

    np.savez(r'K:\2021fmri_work\moyamoya_mri_data\npy_data\ertong_data.npy', X_mix_ertong, Y_mix_ertong)
    print("finished ertong data prepare...")

    ### naoke
    mmd_dir_naoke = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\naoke\mmd"
    hc_dir_naoke = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\naoke\hc"

    X_mix_naoke, Y_mix_naoke = creat_mix_data(mmd_dir_naoke, hc_dir_naoke, 128)
    X_mix_naoke_train, Y_mix_naoke_train, X_mix_naoke_vali, Y_mix_naoke_vali, vali_index_naoke = split_data(X_mix_naoke,
                                                                                                       Y_mix_naoke)

    np.savez(r'K:\2021fmri_work\moyamoya_mri_data\npy_data\naoke_data.npy', X_mix_naoke_train, Y_mix_naoke_train,
             X_mix_naoke_vali, Y_mix_naoke_vali, vali_index_naoke)


    print("finished naoke data prepare...")
    print("finished data save......")


















