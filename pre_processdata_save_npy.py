#!/usr/bin/python
# -*- coding:utf-8 -*-
# author  : zhengjunjie
# time    :
# function:
# version : V1
import os

import cv2
import numpy as np
import pydicom

from read_dcm_data import savefig


def getFlist(path, index):
    # index ：文件列表的第几个文件，
    files_all = []
    # 获取样本名称
    files_name_all = []
    lsdir = os.listdir(path)
    dirs = [os.path.join(path, i) for i in lsdir if os.path.isdir(os.path.join(path, i))]


    for i in range(len(dirs)):
        files = os.listdir(dirs[i])
        #print(dirs[i])
        file = os.path.join(dirs[i], files[index])
        if file.endswith(".DCM") or file.endswith(".dcm"):
            files_all.append(file)
            files_name_all.append(lsdir[i])
    return files_all, files_name_all


def read_data(file_path):
    dcm = pydicom.read_file(file_path)
    image_array = dcm.pixel_array
    return image_array


def savefig(filepath, fig_name):
    # 读取图像信息，存储为图片；
    dcm = pydicom.read_file(filepath)
    image_array = dcm.pixel_array
    image_array_norm = np.zeros([np.size(image_array, 0), np.size(image_array, 1)])

    cv2.normalize(image_array, image_array_norm, 0, 255, cv2.NORM_MINMAX)

    img = np.uint8(image_array_norm)
    cv2.imwrite(fig_name+'.jpg', img)


def savefig2(filepath_input, file_name, file_path_output, num):
    # 读取图像信息，存储为图片；
    dcm = pydicom.read_file(filepath_input, force=True)
    #dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    image_array = dcm.pixel_array
    image_array_norm = np.zeros([np.size(image_array, 0), np.size(image_array, 1)])

    image_array_norm_1 = cv2.normalize(image_array, image_array_norm, 0, 255, cv2.NORM_MINMAX)

    img = np.uint8(image_array_norm_1)

    file_name_output = os.path.join(file_path_output, file_name + "_" + str(num)+'.jpg')
    cv2.imwrite(file_name_output, img)


def savefig3(filepath_input, file_name, file_path_output, num):
    # 读取图像信息，存储为图片；
    dcm = pydicom.read_file(filepath_input, force=True)
    dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    image_array = dcm.pixel_array
    image_array_norm = np.zeros([np.size(image_array, 0), np.size(image_array, 1)])

    image_array_norm_1 = cv2.normalize(image_array, image_array_norm, 0, 255, cv2.NORM_MINMAX)

    img = np.uint8(image_array_norm_1)

    file_name_output = os.path.join(file_path_output, file_name + "_" + str(num)+'.jpg')
    cv2.imwrite(file_name_output, img)


def getfile_savefig(getfile_path, savefig_path, slice):
    flist, f_name_list = getFlist(getfile_path, slice)
    for i in range(len(flist)):
        print(f_name_list[i])
        savefig2(flist[i], f_name_list[i], savefig_path, i)
    return 0


def save_npy(filepath_input, file_name, file_path_output, num):
    # 读取图像信息，存储为图片；
    dcm = pydicom.read_file(filepath_input, force=True)
    #dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    image_array = dcm.pixel_array
    image_array_norm = np.zeros([np.size(image_array, 0), np.size(image_array, 1)])

    image_array_norm_1 = cv2.normalize(image_array, image_array_norm, 0, 255, cv2.NORM_MINMAX)

    img = image_array_norm_1.astype('float32')

    file_name_output = os.path.join(file_path_output, file_name + "_" + str(num)+'.npy')
    #cv2.imwrite(file_name_output, img)
    np.save(file_name_output, img)


def save_npy1(filepath_input, file_name, file_path_output, num):
    # 读取图像信息，存储为图片；
    dcm = pydicom.read_file(filepath_input, force=True)
    dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    image_array = dcm.pixel_array
    image_array_norm = np.zeros([np.size(image_array, 0), np.size(image_array, 1)])

    image_array_norm_1 = cv2.normalize(image_array, image_array_norm, 0, 255, cv2.NORM_MINMAX)

    img = image_array_norm_1.astype('float32')

    file_name_output = os.path.join(file_path_output, file_name + "_" + str(num)+'.npy')
    #cv2.imwrite(file_name_output, img)
    np.save(file_name_output, img)


def getfile_savenpy(getfile_path, savefig_path, slice):
    flist, f_name_list = getFlist(getfile_path, slice)
    for i in range(len(flist)):
        print(f_name_list[i])
        save_npy(flist[i], f_name_list[i], savefig_path, i)
    return 0


def getfile_savenpy1(getfile_path, savefig_path, slice):
    flist, f_name_list = getFlist(getfile_path, slice)
    for i in range(len(flist)):
        print(f_name_list[i])
        save_npy1(flist[i], f_name_list[i], savefig_path, i)
    return 0


if __name__ == '__main__':

    ### 三层聚合，存储成 npy

    ### 鼓楼数据
    ### Dicom 数据文件路径

    hc_filepath1= r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data\hc_new"
    mmd_filepath1 = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data\mmd_new"

    ### 中间层图像数据：
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\gulou\hc\mid"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\gulou\mmd\mid"

    getfile_savenpy(hc_filepath1, hc_filepath2, 1)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 1)

    ### 1层图像数据：
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\gulou\hc\pre"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\gulou\mmd\pre"

    getfile_savenpy(hc_filepath1, hc_filepath2, 0)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 0)

    ### 3层图像数据：
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\gulou\hc\post"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\gulou\mmd\post"

    getfile_savenpy(hc_filepath1, hc_filepath2, 2)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 2)

    print("gulou datafinished......")

    ### 鼓楼数据 新增数据 加入非健康对照
    ### Dicom 数据文件路径

    hc_filepath1= r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\gulou_add\controls_1"
    mmd_filepath1 = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\gulou_add\2018MMD"
    mmd_filepath1_1 = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\gulou_add\20172021MMD"

    ### 中间层图像数据：
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\gulou\hc\mid"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\gulou\mmd\mid"

    getfile_savenpy(hc_filepath1, hc_filepath2, 1)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 1)
    getfile_savenpy(mmd_filepath1_1, mmd_filepath2, 1)

    ### 中间层 前一层图像数据
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\gulou\hc\pre"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\gulou\mmd\pre"

    getfile_savenpy(hc_filepath1, hc_filepath2, 0)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 0)
    getfile_savenpy(mmd_filepath1_1, mmd_filepath2, 0)

    ### 中间层 下一层的图像数据
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\gulou\hc\post"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\gulou\mmd\post"

    getfile_savenpy(hc_filepath1, hc_filepath2, 2)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 2)
    getfile_savenpy(mmd_filepath1_1, mmd_filepath2, 2)

    print("gulou add datafinished......")

    ### 军总数据

    hc_filepath1 = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_junzong\hc"
    mmd_filepath1 = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_junzong\mmd"

    ### 中间层图像数据：
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\junzong\hc\mid"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\junzong\mmd\mid"

    getfile_savenpy(hc_filepath1, hc_filepath2, 1)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 1)

    ### 中间层 前一层图像数据
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\junzong\hc\pre"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\junzong\mmd\pre"

    getfile_savenpy(hc_filepath1, hc_filepath2, 0)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 0)

    ### 中间层 下一层的图像数据
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\junzong\hc\post"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\junzong\mmd\post"

    getfile_savenpy(hc_filepath1, hc_filepath2, 2)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 2)

    print("junzong data finished......")

    ### 盱眙数据

    hc_filepath1 = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_xuyi\hc"
    mmd_filepath1 = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_xuyi\mmd"

    ### 中间层图像数据：
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\xuyi\hc\mid"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\xuyi\mmd\mid"

    getfile_savenpy(hc_filepath1, hc_filepath2, 1)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 1)

    ### 中间层 前一层图像数据
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\xuyi\hc\pre"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\xuyi\mmd\pre"

    getfile_savenpy(hc_filepath1, hc_filepath2, 0)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 0)

    ### 中间层 下一层的图像数据
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\xuyi\hc\post"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\xuyi\mmd\post"

    getfile_savenpy(hc_filepath1, hc_filepath2, 2)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 2)

    print("xuyi data finished......")


    ### 六安数据

    hc_filepath1 = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_luan\nc_luan"
    mmd_filepath1 = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_luan\mmd_luan"

    ### 中间层图像数据：
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\luan\hc\mid"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\luan\mmd\mid"

    getfile_savenpy1(hc_filepath1, hc_filepath2, 1)
    getfile_savenpy1(mmd_filepath1, mmd_filepath2, 1)

    ### 中间层 前一层图像数据
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\luan\hc\pre"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\luan\mmd\pre"

    getfile_savenpy1(hc_filepath1, hc_filepath2, 0)
    getfile_savenpy1(mmd_filepath1, mmd_filepath2, 0)

    ### 中间层 下一层的图像数据
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\luan\hc\post"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\luan\mmd\post"

    getfile_savenpy1(hc_filepath1, hc_filepath2, 2)
    getfile_savenpy1(mmd_filepath1, mmd_filepath2, 2)

    print("luan data finished......")

    ### 逸夫数据

    hc_filepath1 = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_yifu\hc"
    mmd_filepath1 = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_yifu\mmd"

    ### 中间层图像数据：
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\yifu\hc\mid"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\yifu\mmd\mid"

    getfile_savenpy(hc_filepath1, hc_filepath2, 1)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 1)

    ### 中间层 前一层图像数据
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\yifu\hc\pre"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\yifu\mmd\pre"

    getfile_savenpy(hc_filepath1, hc_filepath2, 0)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 0)

    ### 中间层 下一层的图像数据
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\yifu\hc\post"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\yifu\mmd\post"

    getfile_savenpy(hc_filepath1, hc_filepath2, 2)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 2)

    print("yifu data finished......")


    ### 儿童医院

    ### 儿童医院
    hc_filepath1 = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_ertong\hc"
    mmd_filepath1 = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_ertong\mmd"

    ### 中间层图像数据：
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\ertong\hc\mid"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\ertong\mmd\mid"

    getfile_savenpy(hc_filepath1, hc_filepath2, 1)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 1)

    ### 中间层 前一层图像数据
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\ertong\hc\pre"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\ertong\mmd\pre"

    getfile_savenpy(hc_filepath1, hc_filepath2, 0)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 0)

    ### 中间层 下一层的图像数据
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\ertong\hc\post"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\ertong\mmd\post"

    getfile_savenpy(hc_filepath1, hc_filepath2, 2)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 2)

    print("ertong data finished......")


    ### 脑科医院 数据

    #### 检查数据问题
    #data1 = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_naoke\bad\hc\10\1.dcm"
    #data2 = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_naoke\bad\hc\10\1.dcm"
    #data3 = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_naoke\bad\hc\10\1.dcm"
    #dcm = pydicom.read_file(data1, force=True)
    #dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    #image_array = dcm.pixel_array
    #image_array_norm = np.zeros([np.size(image_array, 0), np.size(image_array, 1)])
    #image_array_norm_1 = cv2.normalize(image_array, image_array_norm, 0, 255, cv2.NORM_MINMAX)
    #img = np.uint8(image_array_norm_1)


    hc_filepath1 = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_naoke_rename\hc"
    mmd_filepath1 = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_naoke_rename\mmd"

    ### 中间层图像数据：
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\naoke\hc\mid"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\naoke\mmd\mid"

    getfile_savenpy(hc_filepath1, hc_filepath2, 1)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 1)

    ### 中间层 前一层图像数据
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\naoke\hc\pre"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\naoke\mmd\pre"

    getfile_savenpy(hc_filepath1, hc_filepath2, 0)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 0)

    ### 中间层 下一层的图像数据
    hc_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\naoke\hc\post"
    mmd_filepath2 = r"K:\2021fmri_work\moyamoya_mri_data\npy_data\naoke\mmd\post"

    getfile_savenpy(hc_filepath1, hc_filepath2, 2)
    getfile_savenpy(mmd_filepath1, mmd_filepath2, 2)
    print("naoke data finished......")









