import itertools
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix

from CNN_mmd_cv_combat_multicenter import cnn_test, lenet5_test, vgg16_test, resnet_test, Densnet_test, \
    plot_sonfusion_matrix

from CNN_mmd_cv_multicenter_junzong_luan import cnn_test_new, lenet5_test_new, vgg16_test_new, resnet_test_new, \
    Densnet_test_new


if __name__ == '__main__':

    print("loading data.....")

    X_mix_train = np.load(r'.\npy_data\x_train_aug.npy', allow_pickle=True)
    Y_mix_train = np.load(r'.\npy_data\x_train_aug_label.npy',
                          allow_pickle=True)


    ## junzong
    X_mix_test1 = np.load(r'.\npy_data\x_test_junzong.npy', allow_pickle=True)
    Y_mix_test1 = np.load(r'.\npy_data\x__test_junzong_label.npy',
                          allow_pickle=True)

    ## luan
    X_mix_test2 = np.load(r'.\npy_data\x_test_luan.npy', allow_pickle=True)
    Y_mix_test2 = np.load(r'.\npy_data\x__test_luan_label.npy',
                          allow_pickle=True)
    badsample = [35,36,38,40,53,58]
    X_mix_test2 = np.delete(X_mix_test2, (0,2,3,5,11,14,16,28), axis=0)
    Y_mix_test2 = np.delete(Y_mix_test2, (0,2,3,5,11,14,16,28), axis=0)
    ## naoke
    #X_mix_test3 = np.load(r'K:\2021fmri_work\moyamoya_mri_data\npy_data\x_test_naoke.npy', allow_pickle=True)
    #Y_mix_test3 = np.load(r'K:\2021fmri_work\moyamoya_mri_data\npy_data\x__test_naoke_label.npy',
    #                      allow_pickle=True)

    ## ertong
    X_mix_test4 = np.load(r'.\npy_data\x_test_ertong.npy', allow_pickle=True)
    Y_mix_test4 = np.load(r'.\npy_data\x__test_ertong_label.npy',
                         allow_pickle=True)

    ## 单中心 验证 使用单一中心数据；
    ## all
    X_mix_test = X_mix_test2
    #X_mix_test = np.concatenate((X_mix_test1, X_mix_test2), axis=0)
    #X_mix_test = np.concatenate((X_mix_test, X_mix_test3), axis=0)
    #X_mix_test = np.concatenate((X_mix_test, X_mix_test4), axis=0)

    Y_mix_test = Y_mix_test2
    #Y_mix_test = np.vstack((Y_mix_test1, Y_mix_test2))
    #Y_mix_test = np.vstack((Y_mix_test, Y_mix_test3))
    #Y_mix_test = np.vstack((Y_mix_test, Y_mix_test4))

    print("cnn training and validation.....")

    print("cnn test -------------------------------------")
    cnn_result = cnn_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test)
    #np.save('cnn_result_luan.npy',cnn_result)

    print("lenet test -------------------------------------")
    lenet_result = lenet5_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test)
    #np.save('lenet_result_luan.npy', lenet_result)

    print("vgg test -------------------------------------")
    vgg_16_result = vgg16_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test)
    #np.save('vgg_result_luan.npy', vgg_16_result)

    print("resnet test -------------------------------------")
    resnet_result = resnet_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test)
    #np.save('resnet_result_luan.npy', resnet_result)

    print("densnet test -------------------------------------")
    densnet_result = Densnet_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test)
    #np.save('densnet_result_luan.npy', densnet_result)

    print("plot roc......")

    label_test = Y_mix_test[:, 0]

    fpr_cnn, tpr_cnn, thresholds_keras = roc_curve(label_test, cnn_result['pred_score'])
    fpr_le, tpr_le, thresholds_keras = roc_curve(label_test, lenet_result['pred_score'])
    fpr_vgg, tpr_vgg, thresholds_keras = roc_curve(label_test, vgg_16_result['pred_score'])
    fpr_res, tpr_res, thresholds_keras = roc_curve(label_test, resnet_result['pred_score'])
    fpr_dense, tpr_dense, thresholds_keras = roc_curve(label_test, densnet_result['pred_score'])


    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_cnn, tpr_cnn, label='(SCNN AUC = %.3f)' % cnn_result['auc'])
    plt.plot(fpr_le, tpr_le, label='(LeNet AUC = %.3f)' % lenet_result['auc'])
    plt.plot(fpr_vgg, tpr_vgg, label='(VGG AUC = %.3f)' % vgg_16_result['auc'])
    plt.plot(fpr_res, tpr_res, label='(ResNet AUC = %.3f)' % resnet_result['auc'])
    plt.plot(fpr_dense, tpr_dense, label='(DenseNet AUC = %.3f)' % densnet_result['auc'])

    # -plt.fill_between(fpr[3],tpr[3],tpr[4],alpha=0.1)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('False positive rate',fontsize=18)
    plt.ylabel('True positive rate',fontsize=18)
    plt.title('ROC curve',fontsize=18)
    plt.legend(loc='best')
    #plt.show()
    plt.savefig('luan.png',dpi=300)

    print("plot roc finished......")

    print("plot confuse matrix......")

    ## cnn
    confusion_mat = confusion_matrix(label_test, cnn_result['pred_label'])
    plot_sonfusion_matrix('cnn_cm_luan.png',confusion_mat, classes=range(2))

    ## lenet
    confusion_mat = confusion_matrix(label_test, lenet_result['pred_label'])
    plot_sonfusion_matrix('lenet_cm_luan.png',confusion_mat, classes=range(2))

    ## vgg
    confusion_mat = confusion_matrix(label_test, vgg_16_result['pred_label'])
    plot_sonfusion_matrix('vgg_cm_luan.png',confusion_mat, classes=range(2))

    ## res
    confusion_mat = confusion_matrix(label_test, resnet_result['pred_label'])
    plot_sonfusion_matrix('res_cm_luan.png',confusion_mat, classes=range(2))

    ## dense
    confusion_mat = confusion_matrix(label_test, densnet_result['pred_label'])
    plot_sonfusion_matrix('dense_cm_luan.png',confusion_mat, classes=range(2))


    ##

    print("plot confuse matrix finished......")