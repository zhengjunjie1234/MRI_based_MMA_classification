import os.path

import numpy as np
import cv2
from tensorflow.keras import models, layers, optimizers, losses, metrics, Sequential, regularizers
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import f1_score
from imgaug import augmenters as iaa
import random

from tensorflow.keras.applications import densenet
from tensorflow.keras.applications import xception

from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.layers import BatchNormalization


### resnet



def creat_X_Y_aug_img_data(mmd_path, hc_path, resize):

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

    files_mmd = os.listdir(mmd_path)
    N_mmd = len(files_mmd)
    X1 = np.zeros((N_mmd*10, resize, resize, 3))
    for i in range(N_mmd):
        files_tmp = os.path.join(mmd_path, files_mmd[i])
        file_img = cv2.imread(files_tmp)
        file_img_reszie = cv2.resize(file_img, (resize, resize))
        X1[i*10] = file_img_reszie / 255
        for j in range(1,10):
            #index_temp = random.randint(0,8)
            file_img_aug = seq_list[j].augment_images(file_img_reszie)
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
            #index_temp = random.randint(0,8)
            file_img_aug = seq_list[j].augment_images(file_img_reszie)
            X2[i*10+j] = file_img_aug / 255

    X = np.vstack((X1, X2))
    Y = np.zeros((N_mmd*10 + N_hc*10, 2))
    Y[0:N_mmd*10 - 1, 0] = 1
    Y[N_mmd*10 - 1:, 1] = 1
    return X, Y


def creat_X_Y(mmd_path, hc_path, resize):
    files_mmd = os.listdir(mmd_path)
    N_mmd = len(files_mmd)
    X1 = np.zeros((N_mmd, resize, resize, 3))
    for i in range(N_mmd):
        files_tmp = os.path.join(mmd_path, files_mmd[i])
        file_img = cv2.imread(files_tmp)
        file_img_reszie = cv2.resize(file_img, (resize, resize))
        X1[i] = file_img_reszie/255


    files_hc = os.listdir(hc_path)
    N_hc = len(files_hc)
    X2 = np.zeros((N_hc, resize, resize, 3))
    for i in range(N_hc):
        files_tmp = os.path.join(hc_path, files_hc[i])
        file_img = cv2.imread(files_tmp)
        file_img_reszie = cv2.resize(file_img, (resize, resize))
        X2[i] = file_img_reszie/255

    X= np.vstack((X1,X2))
    Y= np.zeros((N_mmd+N_hc,2))
    Y[0:N_mmd-1,0] = 1
    Y[N_mmd-1:,1] = 1

    return X,Y


def creat_X_Y_savefig(mmd_path, hc_path, resize, mmd_savefig_path, hc_savefig_path):
    files_mmd = os.listdir(mmd_path)
    N_mmd = len(files_mmd)
    X1 = np.zeros((N_mmd, resize, resize, 3))
    for i in range(N_mmd):
        files_tmp = os.path.join(mmd_path, files_mmd[i])
        file_img = cv2.imread(files_tmp)
        file_img_reszie = cv2.resize(file_img, (resize, resize))
        X1[i] = file_img_reszie
        file_name_output = os.path.join(mmd_savefig_path, str(i) + '.jpg')
        cv2.imwrite(file_name_output, file_img_reszie)

    files_hc = os.listdir(hc_path)
    N_hc = len(files_hc)
    X2 = np.zeros((N_hc, resize, resize, 3))
    for i in range(N_hc):
        files_tmp = os.path.join(hc_path, files_hc[i])
        file_img = cv2.imread(files_tmp)
        file_img_reszie = cv2.resize(file_img, (resize, resize))
        X2[i] = file_img_reszie
        file_name_output = os.path.join(hc_savefig_path, str(i) + '.jpg')
        cv2.imwrite(file_name_output, file_img_reszie)

    X = np.vstack((X1, X2))
    Y = np.zeros((N_mmd + N_hc, 2), dtype= int)
    Y[0:N_mmd - 1, 0] = 1
    Y[N_mmd - 1:, 1] = 1

    return X, Y


def model_simple_build(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape,kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())

    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1, activation='sigmoid'))
    #model.summary()
    # 二分类问题，你面对的是一个二分类问题，所以网络最后一层是使用 sigmoid 激活的单一单元（大小为 1 的 Dense 层）。输出为某一类的概率
    print(model.summary())
    return model


def model_simple_build_refine(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())

    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1, activation='sigmoid'))
    #model.summary()
    # 二分类问题，你面对的是一个二分类问题，所以网络最后一层是使用 sigmoid 激活的单一单元（大小为 1 的 Dense 层）。输出为某一类的概率
    print(model.summary())
    return model


def vgg_16_net(input_shape):
    model = Sequential()
    model.add(layers.Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu', name='conv1_block'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_block'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_block'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv4_block'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5_block'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6_block'))
    model.add(layers.Conv2D(256, (1, 1), activation='relu', padding='same', name='conv7_block'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv8_block'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv9_block'))
    model.add(layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='conv10_block'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv11_block'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv12_block'))
    model.add(layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='conv13_block'))
    model.add(layers.BatchNormalization())
    #model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu',name='dense1'))
    #model.add(layers.Dropout(0.5))
    #model.add(layers.Dense(4096, activation='relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid',name='dense2'))
    #model.add(Dense(1, activation='sigmoid'))
    return model


def lenet(input_shape):
    model = Sequential()
    model.add(layers.Conv2D(6, (5, 5), input_shape=input_shape, padding='valid', activation='relu', name='conv1_block'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation='relu', padding='valid', name='conv3_block'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    # model.add(layers.Dense(4096, activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))
    return model


def resnet50():
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    model = Sequential(resnet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def resnet50_new():
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    x = resnet.output
    x = layers.GlobalAveragePooling2D()(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=resnet.input, outputs=predictions)
    return model


def Densenet_mmd():

    densenet_mmd = densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    model = Sequential(densenet_mmd)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def Densenet_mmd_new():
    densenet_mmd = densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    x = densenet_mmd.output
    x = layers.GlobalAveragePooling2D()(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=densenet_mmd.input, outputs=predictions)
    return model


def model_fit_result(X,Y):
    data = X
    label = Y
    label_predict = Y[:, 0]
    acc_train = []
    loss_train = []
    acc_val = []
    loss_val = []
    acc_test = []
    sensitivity = []
    specificity = []
    f1_scores = []
    test_lable_list = []
    test_index_list = []
    predict_lable_list =[]
    predict_score_list = []

    kfold = KFold(n_splits=5, shuffle=True, random_state=7)

    for train, test in kfold.split(data, label.argmax(1)):
        # model = vgg_16_net()
        model_new = model_simple_build(input_shape=(128, 128, 3))
        # model_new = vgg_16_net(input_shape=(256,256,3))
        # sgd = SGD(lr=0.001, decay=1e-8, momentum=0.9, nesterov=True)
        # model_new.compile(optimizer="adam", loss='binary_crossentropy', metrics=[metrics.categorical_accuracy, metrics.AUC, metrics.SpecificityAtSensitivity])
        model_new.compile(optimizer="SGD", loss='binary_crossentropy',
                          metrics=['accuracy'])

        # model_new.fit(data[train], label_predict[train], validation_data=(data[train], label_predict[train]), epochs=40, batch_size=30,
        #          shuffle=True)
        model_new.fit(data[train], label_predict[train], validation_split=0.3, epochs=30,
                      batch_size=50, shuffle=True)
        hist = model_new.history
        acc_train.append(hist.history["accuracy"])
        acc_val.append(hist.history["val_accuracy"])
        loss_train.append(hist.history["loss"])
        loss_val.append(hist.history["val_loss"])

        predicts = model_new.predict_classes(data[test])
        predicts_score = model_new.predict(data[test])
        test_label = label_predict[test]

        TP = np.sum((test_label))
        TN = len(test_label) - TP
        TTP = 0
        TTN = 0
        for i in range(len(test_label)):
            if predicts[i] == test_label[i] and predicts[i] ==0:
                TTN+=1
            elif predicts[i] == test_label[i] and predicts[i] ==1:
                TTP+=1
        sen_temp = TTP/TP
        sep_temp = TTN/TN
        acc_temp = (TTP+TTN)/(TP+TN)
        f1_temp = f1_score(predicts,test_label,average='weighted')
        acc_test.append(acc_temp)
        sensitivity.append(sen_temp)
        specificity.append(sep_temp)
        f1_scores.append(f1_temp)
        test_index_list.append(test)
        test_lable_list.append(test_label)
        predict_lable_list.append(predicts)
        predict_score_list.append(predicts_score)

    model_result = {}
    model_result["acc_train"] = acc_train
    model_result["acc_val"] = acc_val
    model_result["loss_train"] = loss_train
    model_result["loss_val"] = loss_val
    model_result["acc_test"] = acc_test
    model_result["sensitivity"] = sensitivity
    model_result["specificity"] = specificity
    model_result["fi_score"] = f1_scores
    model_result["test_index_list"] = test_index_list
    model_result["test_lable_list"] = test_lable_list
    model_result["predict_lable_list"] = predict_lable_list
    model_result["predict_score_list"] = predict_score_list
    return model_result


def model_fit_result_rf(X,Y):
    data = X
    label = Y
    label_predict = Y[:, 0]
    acc_train = []
    loss_train = []
    acc_val = []
    loss_val = []
    acc_test = []
    sensitivity = []
    specificity = []
    f1_scores = []
    test_lable_list = []
    test_index_list = []
    predict_lable_list =[]
    predict_score_list = []

    kfold = KFold(n_splits=5, shuffle=True, random_state=7)

    for train, test in kfold.split(data, label.argmax(1)):
        # model = vgg_16_net()
        #model_new = model_simple_build(input_shape=(128, 128, 3))
        model_new = vgg_16_net(input_shape=(128,128,3))
        # sgd = SGD(lr=0.001, decay=1e-8, momentum=0.9, nesterov=True)
        # model_new.compile(optimizer="adam", loss='binary_crossentropy', metrics=[metrics.categorical_accuracy, metrics.AUC, metrics.SpecificityAtSensitivity])
        model_new.compile(optimizer="adam", loss='binary_crossentropy',
                          metrics=['accuracy'])

        # model_new.fit(data[train], label_predict[train], validation_data=(data[train], label_predict[train]), epochs=40, batch_size=30,
        #          shuffle=True)
        model_new.fit(data[train], label_predict[train], validation_split=0.2, epochs=30,
                      batch_size=30, shuffle=True)
        hist = model_new.history
        acc_train.append(hist.history["accuracy"])
        acc_val.append(hist.history["val_accuracy"])
        loss_train.append(hist.history["loss"])
        loss_val.append(hist.history["val_loss"])

        predicts = model_new.predict_classes(data[test])
        predicts_score = model_new.predict(data[test])
        test_label = label_predict[test]

        TP = np.sum((test_label))
        TN = len(test_label) - TP
        TTP = 0
        TTN = 0
        for i in range(len(test_label)):
            if predicts[i] == test_label[i] and predicts[i] ==0:
                TTN+=1
            elif predicts[i] == test_label[i] and predicts[i] ==1:
                TTP+=1
        sen_temp = TTP/TP
        sep_temp = TTN/TN
        acc_temp = (TTP+TTN)/(TP+TN)
        f1_temp = f1_score(predicts,test_label,average='weighted')
        acc_test.append(acc_temp)
        sensitivity.append(sen_temp)
        specificity.append(sep_temp)
        f1_scores.append(f1_temp)
        test_index_list.append(test)
        test_lable_list.append(test_label)
        predict_lable_list.append(predicts)
        predict_score_list.append(predicts_score)

    model_result = {}
    model_result["acc_train"] = acc_train
    model_result["acc_val"] = acc_val
    model_result["loss_train"] = loss_train
    model_result["loss_val"] = loss_val
    model_result["acc_test"] = acc_test
    model_result["sensitivity"] = sensitivity
    model_result["specificity"] = specificity
    model_result["fi_score"] = f1_scores
    model_result["test_index_list"] = test_index_list
    model_result["test_lable_list"] = test_lable_list
    model_result["predict_lable_list"] = predict_lable_list
    model_result["predict_score_list"] = predict_score_list
    return model_result


def multi_slice_data(X_post, X_mid, X_pre):

    X_mix = X_mid
    X_mix[:,:,:,0] = X_pre[:,:,:,1]
    X_mix[:,:,:,2] = X_post[:,:,:,1]

    return X_mix


if __name__ == '__main__':

    #print(keras.__version__)

    mmd_dir_post = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_gulou\post_slice_data\train\mmd"
    hc_dir_post = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_gulou\post_slice_data\train\hc"

    mmd_dir_pre = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_gulou\pre_slice_data\train\mmd"
    hc_dir_pre = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_gulou\pre_slice_data\train\hc"

    mmd_dir_mid = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_gulou\mid_slice_data\train\mmd"
    hc_dir_mid = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\all_data_fig_gulou\mid_slice_data\train\hc"

    ## 存储 reshape 后的
    ##savefig_path_mmd = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\reshape_data1\mmd"
    ##savefig_path_hc = r"K:\2021fmri_work\moyamoya_mri_data\moyamoya_data\reshape_data1\hc"

    ##X,Y = creat_X_Y_savefig(mmd_dir_post, hc_dir_post, 128, savefig_path_mmd, savefig_path_hc)

    #X_post,Y_post = creat_X_Y(mmd_dir_post, hc_dir_post, 256)
    #X_pre, Y_pre = creat_X_Y(mmd_dir_pre, hc_dir_pre, 256)
    #X_mid, Y_mid = creat_X_Y(mmd_dir_mid, hc_dir_mid, 256)
    X_post, Y_post = creat_X_Y_aug_img_data(mmd_dir_post, hc_dir_post, 128)
    X_pre, Y_pre = creat_X_Y_aug_img_data(mmd_dir_pre, hc_dir_pre, 128)
    X_mid, Y_mid = creat_X_Y_aug_img_data(mmd_dir_mid, hc_dir_mid, 128)

    model_result_pre = model_fit_result(X_pre, Y_pre)

    #np.save('model_result_pre.npy', model_result_pre)
    #np.save('models/model_result_pre_aug_new.npy', model_result_pre)
    np.save('models/model_result_pre_aug_new_gulou.npy', model_result_pre)

    model_result_mid = model_fit_result(X_mid, Y_mid)

    #np.save("model_result_mid.npy", model_result_mid)
    #np.save("models/model_result_mid_aug_new.npy", model_result_mid)
    np.save("models/model_result_mid_aug_new_gulou.npy", model_result_mid)

    model_result_post = model_fit_result(X_post, Y_post)

    #np.save("model_result_post.npy", model_result_post)
    #np.save("models/model_result_post_aug_new.npy", model_result_post)
    np.save("models/model_result_post_aug_new_gulou.npy", model_result_post)

    #X_mix = multi_slice_data(X_post, X_mid, X_pre)
    #Y_mix = Y_post
    #model_result_mix = model_fit_result(X_mix, Y_mix)

    #np.save("model_result_mix.npy", model_result_mix)
    #np.save("models/model_result_mix_aug_new_score_gulou.npy", model_result_mix)

    print("finished...")

