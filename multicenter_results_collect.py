import os.path

import numpy as np
import pandas as pd

from CNN_mmd_cross_validation import creat_X_Y_aug_img_data, model_fit_result, multi_slice_data, model_simple_build, \
    creat_X_Y
from imgaug import augmenters as iaa
import cv2
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import f1_score


if __name__ == '__main__':
    ### gulou data:
    mmd_dir_post_gulou = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_gulou\post_slice_data\train\mmd"
    hc_dir_post_gulou = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_gulou\post_slice_data\train\hc"

    mmd_dir_pre_gulou = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_gulou\pre_slice_data\train\mmd"
    hc_dir_pre_gulou = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_gulou\pre_slice_data\train\hc"

    mmd_dir_mid_gulou = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_gulou\mid_slice_data\train\mmd"
    hc_dir_mid_gulou = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_gulou\mid_slice_data\train\hc"

    X_post, Y_post = creat_X_Y(mmd_dir_post_gulou, hc_dir_post_gulou, 128)
    X_pre, Y_pre = creat_X_Y(mmd_dir_pre_gulou, hc_dir_pre_gulou, 128)
    X_mid, Y_mid = creat_X_Y(mmd_dir_mid_gulou, hc_dir_mid_gulou, 128)
    # X_post, Y_post = creat_X_Y_aug_img_data_multicenter(mmd_dir_post_gulou, hc_dir_post_gulou, 128)
    # X_pre, Y_pre = creat_X_Y_aug_img_data_multicenter(mmd_dir_pre_gulou, hc_dir_pre_gulou, 128)
    # X_mid, Y_mid = creat_X_Y_aug_img_data_multicenter(mmd_dir_mid_gulou, hc_dir_mid_gulou, 128)
    X_mix_gulou = multi_slice_data(X_post, X_mid, X_pre)
    Y_mix_gulou = Y_post
    print("finished data prepare...")

    ### junzong data:
    mmd_dir_post_junzong = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_junzong\post_slice_data\train\mmd"
    hc_dir_post_junzong = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_junzong\post_slice_data\train\hc"

    mmd_dir_pre_junzong = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_junzong\pre_slice_data\train\mmd"
    hc_dir_pre_junzong = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_junzong\pre_slice_data\train\hc"

    mmd_dir_mid_junzong = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_junzong\mid_slice_data\train\mmd"
    hc_dir_mid_junzong = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_junzong\mid_slice_data\train\hc"

    X_post, Y_post = creat_X_Y(mmd_dir_post_junzong, hc_dir_post_junzong, 128)
    X_pre, Y_pre = creat_X_Y(mmd_dir_pre_junzong, hc_dir_pre_junzong, 128)
    X_mid, Y_mid = creat_X_Y(mmd_dir_mid_junzong, hc_dir_mid_junzong, 128)
    # X_post, Y_post = creat_X_Y_aug_img_data_multicenter(mmd_dir_post_junzong, hc_dir_post_junzong, 128)
    # X_pre, Y_pre = creat_X_Y_aug_img_data_multicenter(mmd_dir_pre_junzong, hc_dir_pre_junzong, 128)
    # X_mid, Y_mid = creat_X_Y_aug_img_data_multicenter(mmd_dir_mid_junzong, hc_dir_mid_junzong, 128)
    X_mix_junzong = multi_slice_data(X_post, X_mid, X_pre)
    Y_mix_junzong = Y_post
    print("finished data prepare...")

    ### luan data:
    mmd_dir_post_luan = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_luan\post_slice_data\train\mmd"
    hc_dir_post_luan = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_luan\post_slice_data\train\hc"

    mmd_dir_pre_luan = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_luan\pre_slice_data\train\mmd"
    hc_dir_pre_luan = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_luan\pre_slice_data\train\hc"

    mmd_dir_mid_luan = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_luan\mid_slice_data\train\mmd"
    hc_dir_mid_luan = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_luan\mid_slice_data\train\hc"

    X_post, Y_post = creat_X_Y(mmd_dir_post_luan, hc_dir_post_luan, 128)
    X_pre, Y_pre = creat_X_Y(mmd_dir_pre_luan, hc_dir_pre_luan, 128)
    X_mid, Y_mid = creat_X_Y(mmd_dir_mid_luan, hc_dir_mid_luan, 128)
    # X_post, Y_post = creat_X_Y_aug_img_data_multicenter(mmd_dir_post_luan, hc_dir_post_luan, 128)
    # X_pre, Y_pre = creat_X_Y_aug_img_data_multicenter(mmd_dir_pre_luan, hc_dir_pre_luan, 128)
    # X_mid, Y_mid = creat_X_Y_aug_img_data_multicenter(mmd_dir_mid_luan, hc_dir_mid_luan, 128)
    X_mix_luan = multi_slice_data(X_post, X_mid, X_pre)
    Y_mix_luan = Y_post
    print("finished data prepare...")

    ### xuyi data
    mmd_dir_post_xuyi = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_xuyi\post_slice_data\train\mmd"
    hc_dir_post_xuyi = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_xuyi\post_slice_data\train\hc"

    mmd_dir_pre_xuyi = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_xuyi\pre_slice_data\train\mmd"
    hc_dir_pre_xuyi = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_xuyi\pre_slice_data\train\hc"

    mmd_dir_mid_xuyi = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_xuyi\mid_slice_data\train\mmd"
    hc_dir_mid_xuyi = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_xuyi\mid_slice_data\train\hc"

    X_post, Y_post = creat_X_Y(mmd_dir_post_xuyi, hc_dir_post_xuyi, 128)
    X_pre, Y_pre = creat_X_Y(mmd_dir_pre_xuyi, hc_dir_pre_xuyi, 128)
    X_mid, Y_mid = creat_X_Y(mmd_dir_mid_xuyi, hc_dir_mid_xuyi, 128)
    # X_post, Y_post = creat_X_Y_aug_img_data_multicenter(mmd_dir_post_xuyi, hc_dir_post_xuyi, 128)
    # X_pre, Y_pre = creat_X_Y_aug_img_data_multicenter(mmd_dir_pre_xuyi, hc_dir_pre_xuyi, 128)
    # X_mid, Y_mid = creat_X_Y_aug_img_data_multicenter(mmd_dir_mid_xuyi, hc_dir_mid_xuyi, 128)
    X_mix_xuyi = multi_slice_data(X_post, X_mid, X_pre)
    Y_mix_xuyi = Y_post
    print("finished data prepare...")

    ### yifu data
    mmd_dir_post_yifu = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_yifu\post_slice_data\train\mmd"
    hc_dir_post_yifu = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_yifu\post_slice_data\train\hc"

    mmd_dir_pre_yifu = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_yifu\pre_slice_data\train\mmd"
    hc_dir_pre_yifu = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_yifu\pre_slice_data\train\hc"

    mmd_dir_mid_yifu = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_yifu\mid_slice_data\train\mmd"
    hc_dir_mid_yifu = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_yifu\mid_slice_data\train\hc"

    X_post, Y_post = creat_X_Y(mmd_dir_post_yifu, hc_dir_post_yifu, 128)
    X_pre, Y_pre = creat_X_Y(mmd_dir_pre_yifu, hc_dir_pre_yifu, 128)
    X_mid, Y_mid = creat_X_Y(mmd_dir_mid_yifu, hc_dir_mid_yifu, 128)
    # X_post, Y_post = creat_X_Y_aug_img_data_multicenter(mmd_dir_post_yifu, hc_dir_post_yifu, 128)
    # X_pre, Y_pre = creat_X_Y_aug_img_data_multicenter(mmd_dir_pre_yifu, hc_dir_pre_yifu, 128)
    # X_mid, Y_mid = creat_X_Y_aug_img_data_multicenter(mmd_dir_mid_yifu, hc_dir_mid_yifu, 128)
    X_mix_yifu = multi_slice_data(X_post, X_mid, X_pre)
    Y_mix_yifu = Y_post

    print("finished data prepare...")
    #####

    X_mix = np.concatenate((X_mix_gulou, X_mix_junzong), axis=0)
    X_mix = np.concatenate((X_mix, X_mix_xuyi), axis=0)
    X_mix = np.concatenate((X_mix, X_mix_luan), axis=0)
    X_mix = np.concatenate((X_mix, X_mix_yifu), axis=0)

    Y_mix = np.vstack((Y_mix_gulou, Y_mix_junzong))
    Y_mix = np.vstack((Y_mix, Y_mix_xuyi))
    Y_mix = np.vstack((Y_mix, Y_mix_luan))
    Y_mix = np.vstack((Y_mix, Y_mix_yifu))

    center_label_gulou = np.ones(np.size(Y_mix_gulou, axis=0))
    center_label_junzong = np.ones(np.size(Y_mix_junzong, axis=0)) + 1
    center_label_xuyi = np.ones(np.size(Y_mix_xuyi, axis=0)) + 2
    center_label_luan = np.ones(np.size(Y_mix_luan, axis=0)) + 3
    center_label_yifu = np.ones(np.size(Y_mix_yifu, axis=0)) + 4

    center_label_multicenter = np.hstack((center_label_gulou, center_label_junzong))
    center_label_multicenter = np.hstack((center_label_multicenter, center_label_xuyi))
    center_label_multicenter = np.hstack((center_label_multicenter, center_label_luan))
    center_label_multicenter = np.hstack((center_label_multicenter, center_label_yifu))

    collect_data = {}
    collect_data['y'] = Y_mix.tolist()
    collect_data['center'] = center_label_multicenter.tolist()

    model_result_mix = np.load("models/model_result_mix_aug_new_score_multicenter_cv.npy", allow_pickle=True)

    test_index = model_result_mix.item()['test_index_list']
    Y_test = model_result_mix.item()['test_lable_list']
    Y_pred = model_result_mix.item()['predict_lable_list']

    test_index_all = []
    test_index_all_acc =[]
    for i in range(len(Y_pred)):
        y_p = Y_pred[i].reshape(1, len(Y_pred[i]))
        y_t = Y_test[i].reshape(1, len(Y_test[i]))
        y_t = y_t.astype('int32')
        y_p = y_p[0].tolist()
        y_t = y_t[0].tolist()
        test_index_temp = test_index[i].tolist()
        for k in test_index_temp:

            if not k in test_index_all:
                test_index_all.append(k)
                test_index_all_acc.append(y_p[test_index_temp.index(k)]-y_t[test_index_temp.index(k)])

    test_all = np.ones(np.size(Y_mix, axis=0))
    test_all = test_all.tolist()
    test_all_acc = np.zeros(np.size(Y_mix, axis=0))
    for i in test_index_all:
        test_all[test_index_all.index(i)] = 2
        test_all_acc[test_index_all.index(i)] = test_index_all_acc[test_index_all.index(i)]

    collect_data['test_index2'] = test_all
    collect_data['test_index2_acc'] = test_all_acc


    ### data_file_name_list
    data_file_names = []
    data_class = []
    ### gulou
    mmd_dir_post_gulou = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_gulou\post_slice_data\train\mmd"
    data_path_filelist = os.listdir(mmd_dir_post_gulou)
    data_file_names = data_file_names + data_path_filelist
    for i in range(len(data_path_filelist)):
        data_class.append('mdd')
    ### gulou
    mmd_dir_post_gulou = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_gulou\post_slice_data\train\hc"
    data_path_filelist = os.listdir(mmd_dir_post_gulou)
    data_file_names = data_file_names + data_path_filelist
    for i in range(len(data_path_filelist)):
        data_class.append('hc')

    ### junzong
    mmd_dir_post_junzong = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_junzong\post_slice_data\train\mmd"
    data_path_filelist = os.listdir(mmd_dir_post_junzong)
    data_file_names = data_file_names + data_path_filelist
    for i in range(len(data_path_filelist)):
        data_class.append('mdd')
    ### junzong
    mmd_dir_post_junzong = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_junzong\post_slice_data\train\hc"
    data_path_filelist = os.listdir(mmd_dir_post_junzong)
    data_file_names = data_file_names + data_path_filelist
    for i in range(len(data_path_filelist)):
        data_class.append('hc')

    ### xuyi
    mmd_dir_post_xuyi = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_xuyi\post_slice_data\train\mmd"
    data_path_filelist = os.listdir(mmd_dir_post_xuyi)
    data_file_names = data_file_names + data_path_filelist
    for i in range(len(data_path_filelist)):
        data_class.append('mdd')
    ### xuyi
    mmd_dir_post_xuyi = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_xuyi\post_slice_data\train\hc"
    data_path_filelist = os.listdir(mmd_dir_post_xuyi)
    data_file_names = data_file_names + data_path_filelist
    for i in range(len(data_path_filelist)):
        data_class.append('hc')

    ### luan
    mmd_dir_post_luan = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_luan\post_slice_data\train\mmd"
    data_path_filelist = os.listdir(mmd_dir_post_luan)
    data_file_names = data_file_names + data_path_filelist
    for i in range(len(data_path_filelist)):
        data_class.append('mdd')

    ### luan
    mmd_dir_post_luan = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_luan\post_slice_data\train\hc"
    data_path_filelist = os.listdir(mmd_dir_post_luan)
    data_file_names = data_file_names + data_path_filelist
    for i in range(len(data_path_filelist)):
        data_class.append('hc')


    ### yifu
    mmd_dir_post_yifu = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_yifu\post_slice_data\train\mmd"
    data_path_filelist = os.listdir(mmd_dir_post_yifu)
    data_file_names = data_file_names + data_path_filelist
    for i in range(len(data_path_filelist)):
        data_class.append('mdd')

    ### yifu
    mmd_dir_post_yifu = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_yifu\post_slice_data\train\hc"
    data_path_filelist = os.listdir(mmd_dir_post_yifu)
    data_file_names = data_file_names + data_path_filelist
    for i in range(len(data_path_filelist)):
        data_class.append('hc')


    ## collect
    collect_data['file_name'] = data_file_names
    collect_data['group'] = data_class
    collect_data = pd.DataFrame(collect_data)
    ##collect_data.to_csv('collect_test_data.csv')


    print("finished data cnn...")

















