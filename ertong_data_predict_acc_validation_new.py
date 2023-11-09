import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix


from CNN_mmd_cv_combat_multicenter import cnn_test, lenet5_test, vgg16_test, resnet_test, Densnet_test, \
    plot_sonfusion_matrix
from CNN_mmd_cv_multicenter_junzong_luan import cnn_test_new, lenet5_test_new, vgg16_test_new, resnet_test_new, \
    Densnet_test_new

if __name__ == '__main__':

    #X_mix_new = np.load("K:/2021fmri_work/moyamoya_mri_data/img_data/img_combat_data.npy", allow_pickle=True)
    #Y_mix = np.load("K:/2021fmri_work/moyamoya_mri_data/img_data/img_ori_data_label.npy", allow_pickle=True)

    #X_mix = np.load("K:/2021fmri_work/moyamoya_mri_data/img_data/img_ori_data_aug.npy", allow_pickle=True)
    #Y_mix = np.load("K:/2021fmri_work/moyamoya_mri_data/img_data/img_ori_data_aug_label.npy", allow_pickle=True)

    ## train data
    '''
    X_mix_train = np.load("K:/2021fmri_work/moyamoya_mri_data/img_data/img_ori_data_aug_train.npy", allow_pickle=True)
    Y_mix_train = np.load("K:/2021fmri_work/moyamoya_mri_data/img_data/img_ori_data_aug_label_train.npy", allow_pickle=True)

    model_result_mix = model_fit_result(X_mix_train, Y_mix_train)

    np.save("models/model_result_mix_aug_train.npy", model_result_mix)
    '''

    print("loading data.....")

    X_mix_train = np.load("K:/2021fmri_work/moyamoya_mri_data/img_data/img_ori_data_aug_train_junzong_luan.npy", allow_pickle=True)
    Y_mix_train = np.load("K:/2021fmri_work/moyamoya_mri_data/img_data/img_ori_data_aug_label_train_junzong_luan.npy",
                          allow_pickle=True)

    X_mix_test = np.load("K:/2021fmri_work/moyamoya_mri_data/img_data/img_ori_data_aug_ertong.npy", allow_pickle=True)
    Y_mix_test = np.load("K:/2021fmri_work/moyamoya_mri_data/img_data/img_ori_data_aug_ertong_label.npy",
                          allow_pickle=True)

    print("cnn training and validation.....")

    cnn_result = cnn_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test)

    lenet_result = lenet5_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test)

    acc_16_result = vgg16_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test)

    resnet_result = resnet_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test)

    densnet_result = Densnet_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test)

    print("plot roc......")

    label_test = Y_mix_test[:, 0]

    fpr_cnn, tpr_cnn, thresholds_keras = roc_curve(label_test, cnn_result['pred_score'])
    fpr_le, tpr_le, thresholds_keras = roc_curve(label_test, lenet_result['pred_score'])
    fpr_vgg, tpr_vgg, thresholds_keras = roc_curve(label_test, acc_16_result['pred_score'])
    fpr_res, tpr_res, thresholds_keras = roc_curve(label_test, resnet_result['pred_score'])
    fpr_dense, tpr_dense, thresholds_keras = roc_curve(label_test, densnet_result['pred_score'])


    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_cnn, tpr_cnn, label='(CNN AUC = %.3f)' % cnn_result['auc'])
    plt.plot(fpr_le, tpr_le, label='(LeNet AUC = %.3f)' % lenet_result['auc'])
    plt.plot(fpr_vgg, tpr_vgg, label='(VGG AUC = %.3f)' % acc_16_result['auc'])
    plt.plot(fpr_res, tpr_res, label='(ResNet AUC = %.3f)' % resnet_result['auc'])
    plt.plot(fpr_dense, tpr_dense, label='(DenseNet AUC = %.3f)' % densnet_result['auc'])

    # -plt.fill_between(fpr[3],tpr[3],tpr[4],alpha=0.1)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    print("plot roc finished......")


    print("plot confuse matrix......")

    ## cnn
    confusion_mat = confusion_matrix(label_test, cnn_result['pred_label'])
    plot_sonfusion_matrix(confusion_mat, classes=range(2))

    ## lenet
    confusion_mat = confusion_matrix(label_test, lenet_result['pred_label'])
    plot_sonfusion_matrix(confusion_mat, classes=range(2))

    ## vgg
    confusion_mat = confusion_matrix(label_test, acc_16_result['pred_label'])
    plot_sonfusion_matrix(confusion_mat, classes=range(2))

    ## res
    confusion_mat = confusion_matrix(label_test, resnet_result['pred_label'])
    plot_sonfusion_matrix(confusion_mat, classes=range(2))

    ## dense
    confusion_mat = confusion_matrix(label_test, densnet_result['pred_label'])
    plot_sonfusion_matrix(confusion_mat, classes=range(2))


    print("plot confuse matrix finished......")