import itertools
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix
from statsmodels.stats.proportion import proportion_confint
from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.python.keras.models import Model

from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

from tensorflow.python.keras.saving.save import load_model

from CNN_mmd_cross_validation import model_fit_result, model_fit_result_rf, model_simple_build, vgg_16_net, lenet, \
    resnet50, Densenet_mmd, Densenet_mmd_new, resnet50_new
from CNN_mmd_cv_combat_multicenter import cnn_test, lenet5_test, vgg16_test, resnet_test, Densnet_test, \
    plot_sonfusion_matrix


def Densnet_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test):

    label_train = Y_mix_train[:, 0]
    label_test = Y_mix_test[:, 0]

    model_path = "./junzong_luan_test_models/model_densnet.h5"

    acc_train = []
    loss_train = []
    acc_val = []
    loss_val = []

    if os.path.exists(model_path):
        print("model training finished... loading...")
        model_new = load_model(model_path)

    else:
        print("model doesnot exist, training start... ")
        ## train and test

        model_new = Densenet_mmd_new()

        # model_new.compile(optimizer="adam", loss='binary_crossentropy', metrics=[metrics.categorical_accuracy,
        # metrics.AUC, metrics.SpecificityAtSensitivity])
        model_new.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=['accuracy'])

        # model_new.fit(data[train], label_predict[train], validation_data=(data[train], label_predict[train]), epochs=40, batch_size=30,
        #          shuffle=True)
        #model_new.fit(X_mix_train, label_train, validation_split=0.3, epochs=20,
        #              batch_size=64, shuffle=True)
        model_new.fit(X_mix_train, label_train, validation_data=(X_mix_test, label_test), epochs=20, batch_size=32 , shuffle=True)
        model_new.save(model_path)
        hist = model_new.history
        acc_train.append(hist.history["accuracy"])
        acc_val.append(hist.history["val_accuracy"])
        loss_train.append(hist.history["loss"])
        loss_val.append(hist.history["val_loss"])

    #predicts = model_new.predict_classes(X_mix_test)
    predicts_score = model_new.predict(X_mix_test)
    predicts = (predicts_score > 0.5).astype('int32')
    test_label = label_test
    TP = np.sum((test_label))
    TN = len(test_label) - TP
    TTP = 0
    TTN = 0
    for i in range(len(test_label)):
        if predicts[i][0] == test_label[i] and predicts[i][0] == 0:
            TTN += 1
        elif predicts[i][0] == test_label[i] and predicts[i][0] == 1:
            TTP += 1
    sen_temp = TTP / TP
    sep_temp = TTN / TN
    acc_temp = (TTP + TTN) / (TP + TN)
    f1_temp = f1_score(predicts, test_label, average='weighted')

    sen_temp_ci1, sen_temp_ci2 = ci95value(TTP, TP)
    sep_temp_ci1, sep_temp_ci2 = ci95value(TTN, TN)
    acc_temp_ci1, acc_temp_ci2 = ci95value(TTP + TTN, TP + TN)
    f1_temp_ci1, f1_temp_ci2 = ci95value(TTP + TTN, TP + TN)

    print("sensitivity == %.4f ......" % sen_temp)
    print("sensitivity_ci1 == %.4f ......" % sen_temp_ci1)
    print("sensitivity_ci2 == %.4f ......" % sen_temp_ci2)

    print("specificity == %.4f ......" % sep_temp)
    print("specificity_ci1 == %.4f ......" % sep_temp_ci1)
    print("specificity_ci2 == %.4f ......" % sep_temp_ci2)

    print("accuracy == %.4f ......" % acc_temp)
    print("accuracy_ci1 == %.4f ......" % acc_temp_ci1)
    print("accuracy_ci2 == %.4f ......" % acc_temp_ci2)

    print("f1 score == %.4f ......" % f1_temp)
    print("f1 score_ci1 == %.4f ......" % f1_temp_ci1)
    print("f1 score_ci2 == %.4f ......" % f1_temp_ci2)

    Y_train0 = test_label
    Y_pred_0 = predicts_score

    Y_train0_temp = Y_train0
    Y_pred_0_temp = Y_pred_0
    fpr_temp, tpr_temp, thresholds_keras = roc_curve(Y_train0_temp, Y_pred_0_temp)
    auc_1_temp = auc(fpr_temp, tpr_temp)

    print("AUC : ", auc_1_temp)

    acc_results = {}
    acc_results['sensitivity'] = sen_temp
    acc_results['specificity'] = sep_temp
    acc_results['accuracy'] = acc_temp
    acc_results['f1 score'] = f1_temp
    acc_results['auc'] = auc_1_temp
    acc_results['pred_label'] = predicts
    acc_results['pred_score'] = predicts_score
    acc_results['ttp'] = TTP
    acc_results['ttn'] = TTN
    acc_results["acc_train"] = acc_train
    acc_results["acc_val"] = acc_val
    acc_results["loss_train"] = loss_train
    acc_results["loss_val"] = loss_val

    return acc_results


def resnet_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test):

    label_train = Y_mix_train[:, 0]
    label_test = Y_mix_test[:, 0]

    model_path = "./junzong_luan_test_models/model_resnet.h5"

    acc_train = []
    loss_train = []
    acc_val = []
    loss_val = []

    if os.path.exists(model_path):
        print("model training finished... loading...")
        model_new = load_model(model_path)

    else:
        print("model doesnot exist, training start... ")

        model_new = resnet50_new ()

        # model_new.compile(optimizer="adam", loss='binary_crossentropy', metrics=[metrics.categorical_accuracy,
        # metrics.AUC, metrics.SpecificityAtSensitivity])
        model_new.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=['accuracy'])

        # model_new.fit(data[train], label_predict[train], validation_data=(data[train], label_predict[train]), epochs=40, batch_size=30,
        #          shuffle=True)
        #model_new.fit(X_mix_train, label_train, validation_split=0.3, epochs=20,
        #              batch_size=64, shuffle=True)
        model_new.fit(X_mix_train, label_train, validation_data=(X_mix_test, label_test), epochs=20, batch_size=32, shuffle=True)
        model_new.save(model_path)
        hist = model_new.history
        acc_train.append(hist.history["accuracy"])
        acc_val.append(hist.history["val_accuracy"])
        loss_train.append(hist.history["loss"])
        loss_val.append(hist.history["val_loss"])

    #predicts = model_new.predict_classes(X_mix_test)
    predicts_score = model_new.predict(X_mix_test)
    predicts = (predicts_score > 0.5).astype('int32')
    test_label = label_test
    TP = np.sum((test_label))
    TN = len(test_label) - TP
    TTP = 0
    TTN = 0
    for i in range(len(test_label)):
        if predicts[i][0] == test_label[i] and predicts[i][0] == 0:
            TTN += 1
        elif predicts[i][0] == test_label[i] and predicts[i][0] == 1:
            TTP += 1
    sen_temp = TTP / TP
    sep_temp = TTN / TN
    acc_temp = (TTP + TTN) / (TP + TN)
    f1_temp = f1_score(predicts, test_label, average='weighted')

    sen_temp_ci1, sen_temp_ci2 = ci95value(TTP, TP)
    sep_temp_ci1, sep_temp_ci2 = ci95value(TTN, TN)
    acc_temp_ci1, acc_temp_ci2 = ci95value(TTP + TTN, TP + TN)
    f1_temp_ci1, f1_temp_ci2 = ci95value(TTP + TTN, TP + TN)

    print("sensitivity == %.4f ......" % sen_temp)
    print("sensitivity_ci1 == %.4f ......" % sen_temp_ci1)
    print("sensitivity_ci2 == %.4f ......" % sen_temp_ci2)

    print("specificity == %.4f ......" % sep_temp)
    print("specificity_ci1 == %.4f ......" % sep_temp_ci1)
    print("specificity_ci2 == %.4f ......" % sep_temp_ci2)

    print("accuracy == %.4f ......" % acc_temp)
    print("accuracy_ci1 == %.4f ......" % acc_temp_ci1)
    print("accuracy_ci2 == %.4f ......" % acc_temp_ci2)

    print("f1 score == %.4f ......" % f1_temp)
    print("f1 score_ci1 == %.4f ......" % f1_temp_ci1)
    print("f1 score_ci2 == %.4f ......" % f1_temp_ci2)

    Y_train0 = test_label
    Y_pred_0 = predicts_score

    Y_train0_temp = Y_train0
    Y_pred_0_temp = Y_pred_0
    fpr_temp, tpr_temp, thresholds_keras = roc_curve(Y_train0_temp, Y_pred_0_temp)
    auc_1_temp = auc(fpr_temp, tpr_temp)

    print("AUC : ", auc_1_temp)

    acc_results = {}
    acc_results['sensitivity'] = sen_temp
    acc_results['specificity'] = sep_temp
    acc_results['accuracy'] = acc_temp
    acc_results['f1 score'] = f1_temp
    acc_results['auc'] = auc_1_temp
    acc_results['pred_label'] = predicts
    acc_results['pred_score'] = predicts_score
    acc_results['ttp'] = TTP
    acc_results['ttn'] = TTN
    acc_results["acc_train"] = acc_train
    acc_results["acc_val"] = acc_val
    acc_results["loss_train"] = loss_train
    acc_results["loss_val"] = loss_val

    return acc_results


def vgg16_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test):

    label_train = Y_mix_train[:, 0]
    label_test = Y_mix_test[:, 0]

    model_path = "./junzong_luan_test_models/model_vgg16.h5"

    acc_train = []
    loss_train = []
    acc_val = []
    loss_val = []

    if os.path.exists(model_path):
        print("model training finished... loading...")
        model_new = load_model(model_path)

    else:
        print("model doesnot exist, training start... ")
        ## train and test
        # model_new = model_simple_build(input_shape=(128, 128, 3))
        model_new = vgg_16_net(input_shape=(128, 128, 3))
        # model_new.compile(optimizer="adam", loss='binary_crossentropy', metrics=[metrics.categorical_accuracy, metrics.AUC, metrics.SpecificityAtSensitivity])
        model_new.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=['accuracy'])

        # model_new.fit(data[train], label_predict[train], validation_data=(data[train], label_predict[train]), epochs=40, batch_size=30,
        #          shuffle=True)
        #model_new.fit(X_mix_train, label_train, validation_split=0.3, epochs=30,
        #              batch_size=64, shuffle=True)
        model_new.fit(X_mix_train, label_train, validation_data=(X_mix_test, label_test), epochs=20,batch_size=32, shuffle=True)
        model_new.save(model_path)
        hist = model_new.history
        acc_train.append(hist.history["accuracy"])
        acc_val.append(hist.history["val_accuracy"])
        loss_train.append(hist.history["loss"])
        loss_val.append(hist.history["val_loss"])

    predicts = model_new.predict_classes(X_mix_test)
    predicts_score = model_new.predict(X_mix_test)
    test_label = label_test
    TP = np.sum((test_label))
    TN = len(test_label) - TP
    TTP = 0
    TTN = 0
    for i in range(len(test_label)):
        if predicts[i][0] == test_label[i] and predicts[i][0] == 0:
            TTN += 1
        elif predicts[i][0] == test_label[i] and predicts[i][0] == 1:
            TTP += 1
    sen_temp = TTP / TP
    sep_temp = TTN / TN
    acc_temp = (TTP + TTN) / (TP + TN)
    f1_temp = f1_score(predicts, test_label, average='weighted')

    sen_temp_ci1, sen_temp_ci2 = ci95value(TTP, TP)
    sep_temp_ci1, sep_temp_ci2 = ci95value(TTN, TN)
    acc_temp_ci1, acc_temp_ci2 = ci95value(TTP + TTN, TP + TN)
    f1_temp_ci1, f1_temp_ci2 = ci95value(TTP + TTN, TP + TN)

    print("sensitivity == %.4f ......" % sen_temp)
    print("sensitivity_ci1 == %.4f ......" % sen_temp_ci1)
    print("sensitivity_ci2 == %.4f ......" % sen_temp_ci2)

    print("specificity == %.4f ......" % sep_temp)
    print("specificity_ci1 == %.4f ......" % sep_temp_ci1)
    print("specificity_ci2 == %.4f ......" % sep_temp_ci2)

    print("accuracy == %.4f ......" % acc_temp)
    print("accuracy_ci1 == %.4f ......" % acc_temp_ci1)
    print("accuracy_ci2 == %.4f ......" % acc_temp_ci2)

    print("f1 score == %.4f ......" % f1_temp)
    print("f1 score_ci1 == %.4f ......" % f1_temp_ci1)
    print("f1 score_ci2 == %.4f ......" % f1_temp_ci2)

    Y_train0 = test_label
    Y_pred_0 = predicts_score

    Y_train0_temp = Y_train0
    Y_pred_0_temp = Y_pred_0
    fpr_temp, tpr_temp, thresholds_keras = roc_curve(Y_train0_temp, Y_pred_0_temp)
    auc_1_temp = auc(fpr_temp, tpr_temp)

    print("AUC : ", auc_1_temp)

    acc_results = {}
    acc_results['sensitivity'] = sen_temp
    acc_results['specificity'] = sep_temp
    acc_results['accuracy'] = acc_temp
    acc_results['f1 score'] = f1_temp
    acc_results['auc'] = auc_1_temp
    acc_results['pred_label'] = predicts
    acc_results['pred_score'] = predicts_score
    acc_results['ttp'] = TTP
    acc_results['ttn'] = TTN
    acc_results["acc_train"] = acc_train
    acc_results["acc_val"] = acc_val
    acc_results["loss_train"] = loss_train
    acc_results["loss_val"] = loss_val

    return acc_results


def cnn_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test):
    label_train = Y_mix_train[:, 0]
    label_test = Y_mix_test[:, 0]

    model_path = "./junzong_luan_test_models/model_cnn.h5"

    acc_train = []
    loss_train = []
    acc_val = []
    loss_val = []

    if os.path.exists(model_path):
        print("model training finished... loading...")
        model_new = load_model(model_path)
    else:
        print("model doesnot exist, training start... ")
        ## train and test
        model_new = model_simple_build(input_shape=(128, 128, 3))
        #model_new = vgg_16_net(input_shape=(128, 128, 3))

        # model_new.compile(optimizer="adam", loss='binary_crossentropy', metrics=[metrics.categorical_accuracy, metrics.AUC, metrics.SpecificityAtSensitivity])
        model_new.compile(optimizer="adam", loss='binary_crossentropy',
                          metrics=['accuracy'])

        # model_new.fit(data[train], label_predict[train], validation_data=(data[train], label_predict[train]), epochs=40, batch_size=30,
        #          shuffle=True)
        #model_new.fit(X_mix_train, label_train, validation_split=0.3, epochs=30,
        #              batch_size=64, shuffle=True)
        model_new.fit(X_mix_train, label_train, validation_data=(X_mix_test,label_test), epochs=20,batch_size=32, shuffle=True)
        model_new.save(model_path)
        hist = model_new.history
        acc_train.append(hist.history["accuracy"])
        acc_val.append(hist.history["val_accuracy"])
        loss_train.append(hist.history["loss"])
        loss_val.append(hist.history["val_loss"])

    predicts = model_new.predict_classes(X_mix_test)
    predicts_score = model_new.predict(X_mix_test)
    test_label = label_test
    TP = np.sum((test_label))
    TN = len(test_label) - TP
    TTP = 0
    TTN = 0
    for i in range(len(test_label)):
        if predicts[i][0] == test_label[i] and predicts[i][0] == 0:
            TTN += 1
        elif predicts[i][0] == test_label[i] and predicts[i][0] == 1:
            TTP += 1
    sen_temp = TTP / TP
    sep_temp = TTN / TN
    acc_temp = (TTP + TTN) / (TP + TN)
    f1_temp = f1_score(predicts, test_label, average='weighted')

    sen_temp_ci1,sen_temp_ci2 = ci95value(TTP,TP)
    sep_temp_ci1,sep_temp_ci2 = ci95value(TTN, TN)
    acc_temp_ci1, acc_temp_ci2= ci95value(TTP + TTN, TP + TN)
    f1_temp_ci1,f1_temp_ci2  = ci95value(TTP + TTN, TP + TN)

    print("sensitivity == %.4f ......" % sen_temp)
    print("sensitivity_ci1 == %.4f ......" % sen_temp_ci1)
    print("sensitivity_ci2 == %.4f ......" % sen_temp_ci2)

    print("specificity == %.4f ......" % sep_temp)
    print("specificity_ci1 == %.4f ......" % sep_temp_ci1)
    print("specificity_ci2 == %.4f ......" % sep_temp_ci2)

    print("accuracy == %.4f ......" % acc_temp)
    print("accuracy_ci1 == %.4f ......" % acc_temp_ci1)
    print("accuracy_ci2 == %.4f ......" % acc_temp_ci2)

    print("f1 score == %.4f ......" % f1_temp)
    print("f1 score_ci1 == %.4f ......" % f1_temp_ci1)
    print("f1 score_ci2 == %.4f ......" % f1_temp_ci2)

    Y_train0 = test_label
    Y_pred_0 = predicts_score

    Y_train0_temp = Y_train0
    Y_pred_0_temp = Y_pred_0
    fpr_temp, tpr_temp, thresholds_keras = roc_curve(Y_train0_temp, Y_pred_0_temp)
    auc_1_temp = auc(fpr_temp, tpr_temp)

    print("AUC : ", auc_1_temp)

    acc_results = {}
    acc_results['sensitivity'] = sen_temp
    acc_results['specificity'] = sep_temp
    acc_results['accuracy'] = acc_temp
    acc_results['f1 score'] = f1_temp
    acc_results['auc'] = auc_1_temp
    acc_results['pred_label'] = predicts
    acc_results['pred_score'] = predicts_score
    acc_results['ttp'] = TTP
    acc_results['ttn'] = TTN
    acc_results["acc_train"] = acc_train
    acc_results["acc_val"] = acc_val
    acc_results["loss_train"] = loss_train
    acc_results["loss_val"] = loss_val

    return acc_results


def lenet5_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test):
    label_train = Y_mix_train[:, 0]
    label_test = Y_mix_test[:, 0]

    model_path = "./junzong_luan_test_models/model_lenet.h5"

    acc_train = []
    loss_train = []
    acc_val = []
    loss_val = []

    if os.path.exists(model_path):
        print("model training finished... loading...")
        model_new = load_model(model_path)

    else:
        print("model doesnot exist, training start... ")
        ## train and test
        # model_new = model_simple_build(input_shape=(128, 128, 3))
        model_new = lenet(input_shape=(128, 128, 3))

        # model_new.compile(optimizer="adam", loss='binary_crossentropy', metrics=[metrics.categorical_accuracy, metrics.AUC, metrics.SpecificityAtSensitivity])
        model_new.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=['accuracy'])

        # model_new.fit(data[train], label_predict[train], validation_data=(data[train], label_predict[train]), epochs=40, batch_size=30,
        #          shuffle=True)
        #model_new.fit(X_mix_train, label_train, validation_split=0.3, epochs=30,
        #              batch_size=64, shuffle=True)
        model_new.fit(X_mix_train, label_train, validation_data=(X_mix_test, label_test), epochs=20,batch_size=32,shuffle=True)
        model_new.save(model_path)

        hist = model_new.history
        acc_train.append(hist.history["accuracy"])
        acc_val.append(hist.history["val_accuracy"])
        loss_train.append(hist.history["loss"])
        loss_val.append(hist.history["val_loss"])

    predicts = model_new.predict_classes(X_mix_test)
    predicts_score = model_new.predict(X_mix_test)
    test_label = label_test
    TP = np.sum((test_label))
    TN = len(test_label) - TP
    TTP = 0
    TTN = 0
    for i in range(len(test_label)):
        if predicts[i][0] == test_label[i] and predicts[i][0] == 0:
            TTN += 1
        elif predicts[i][0] == test_label[i] and predicts[i][0] == 1:
            TTP += 1
    sen_temp = TTP / TP
    sep_temp = TTN / TN
    acc_temp = (TTP + TTN) / (TP + TN)
    f1_temp = f1_score(predicts, test_label, average='weighted')

    sen_temp_ci1, sen_temp_ci2 = ci95value(TTP, TP)
    sep_temp_ci1, sep_temp_ci2 = ci95value(TTN, TN)
    acc_temp_ci1, acc_temp_ci2 = ci95value(TTP + TTN, TP + TN)
    f1_temp_ci1, f1_temp_ci2 = ci95value(TTP + TTN, TP + TN)

    print("sensitivity == %.4f ......" % sen_temp)
    print("sensitivity_ci1 == %.4f ......" % sen_temp_ci1)
    print("sensitivity_ci2 == %.4f ......" % sen_temp_ci2)

    print("specificity == %.4f ......" % sep_temp)
    print("specificity_ci1 == %.4f ......" % sep_temp_ci1)
    print("specificity_ci2 == %.4f ......" % sep_temp_ci2)

    print("accuracy == %.4f ......" % acc_temp)
    print("accuracy_ci1 == %.4f ......" % acc_temp_ci1)
    print("accuracy_ci2 == %.4f ......" % acc_temp_ci2)

    print("f1 score == %.4f ......" % f1_temp)
    print("f1 score_ci1 == %.4f ......" % f1_temp_ci1)
    print("f1 score_ci2 == %.4f ......" % f1_temp_ci2)

    Y_train0 = test_label
    Y_pred_0 = predicts_score

    Y_train0_temp = Y_train0
    Y_pred_0_temp = Y_pred_0
    fpr_temp, tpr_temp, thresholds_keras = roc_curve(Y_train0_temp, Y_pred_0_temp)
    auc_1_temp = auc(fpr_temp, tpr_temp)

    print("AUC : ", auc_1_temp)

    acc_results = {}
    acc_results['sensitivity'] = sen_temp
    acc_results['specificity'] = sep_temp
    acc_results['accuracy'] = acc_temp
    acc_results['f1 score'] = f1_temp
    acc_results['auc'] = auc_1_temp
    acc_results['pred_label'] = predicts
    acc_results['pred_score'] = predicts_score
    acc_results['ttp'] = TTP
    acc_results['ttn'] = TTN
    acc_results["acc_train"] = acc_train
    acc_results["acc_val"] = acc_val
    acc_results["loss_train"] = loss_train
    acc_results["loss_val"] = loss_val

    return acc_results


def shuffle_data(X_mix_train,Y_mix_train):
    index = np.arange(np.size(Y_mix_train,0))
    np.random.shuffle(index)

    X_train = X_mix_train[index, :, :, :]  # X_train是训练集，y_train是训练标签
    y_train = Y_mix_train[index]
    return X_train,y_train


def ci95value(n_acc, n_all):
    ### 95%ci  z = 1.96
    #civalue = 1.96*np.sqrt( (acc * (1 - acc)) / n)
    lower, upper = proportion_confint(n_acc, n_all, 0.05)

    return lower,upper


if __name__ == '__main__':

    ## test_ load model
    #model_new = load_model("./junzong_luan_test_models/model_cnn.h5")

    #plot_acc_loss(model_new)


    print("loading data.....")

    X_mix_train = np.load(r'.\npy_data\x_train_aug.npy', allow_pickle=True)
    Y_mix_train = np.load(r'.\npy_data\x_train_aug_label.npy',
                          allow_pickle=True)


    X_mix_test = np.load(r'.\npy_data\x_vali.npy', allow_pickle=True)
    Y_mix_test = np.load(r'.\npy_data\x_vali_label.npy',
                          allow_pickle=True)

    print("cnn training and validation.....")

    print("cnn test -------------------------------------")
    cnn_result = cnn_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test)
    #np.save('cnn_result_train.npy',cnn_result)

    print("lenet test -------------------------------------")
    lenet_result = lenet5_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test)
    #np.save('lenet_result_train.npy', lenet_result)

    print("vgg test -------------------------------------")
    vgg_16_result = vgg16_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test)
    #np.save('vgg_result_train.npy', vgg_16_result)

    print("resnet test -------------------------------------")
    resnet_result = resnet_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test)
    #np.save('resnet_result_train.npy', resnet_result)

    print("densnet test -------------------------------------")
    densnet_result = Densnet_test_new(X_mix_train, Y_mix_train, X_mix_test, Y_mix_test)
    #np.save('densnet_result_train .npy', densnet_result)

    print("plot roc......")

    label_test = Y_mix_test[:, 0]

    fpr_cnn, tpr_cnn, thresholds_keras = roc_curve(label_test, cnn_result['pred_score'])
    fpr_le, tpr_le, thresholds_keras = roc_curve(label_test, lenet_result['pred_score'])
    fpr_vgg, tpr_vgg, thresholds_keras = roc_curve(label_test, vgg_16_result['pred_score'])
    fpr_res, tpr_res, thresholds_keras = roc_curve(label_test, resnet_result['pred_score'])
    fpr_dense, tpr_dense, thresholds_keras = roc_curve(label_test, densnet_result['pred_score'])


    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_cnn, tpr_cnn, label='(CNN AUC = %.3f)' % cnn_result['auc'])
    plt.plot(fpr_le, tpr_le, label='(LeNet AUC = %.3f)' % lenet_result['auc'])
    plt.plot(fpr_vgg, tpr_vgg, label='(VGG AUC = %.3f)' % vgg_16_result['auc'])
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
    confusion_mat = confusion_matrix(label_test, vgg_16_result['pred_label'])
    plot_sonfusion_matrix(confusion_mat, classes=range(2))

    ## res
    confusion_mat = confusion_matrix(label_test, resnet_result['pred_label'])
    plot_sonfusion_matrix(confusion_mat, classes=range(2))

    ## dense
    confusion_mat = confusion_matrix(label_test, densnet_result['pred_label'])
    plot_sonfusion_matrix(confusion_mat, classes=range(2))


    ##
    print("plot confuse matrix finished......")