import os

import numpy as np
import tensorflow.keras as tk
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.applications.efficientnet import EfficientNet, EfficientNetB0
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.applications.nasnet import NASNet, NASNetLarge, NASNetMobile
from tensorflow.python.keras.callbacks import ModelCheckpoint


def to_one_hot(data_labels, dimension=3):
    results = np.zeros((len(data_labels), dimension))
    for i, data_labels in enumerate(data_labels):
        results[i, data_labels] = 1
    return results
c

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_1 = true_positives / (possible_positives + K.epsilon())
    return recall_1


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_1 = true_positives / (predicted_positives + K.epsilon())
    return precision_1


def f1(y_true, y_pred):
    pre = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    score = 2 * (pre * rec) / (pre + rec)
    return score


def shuffle_list(files, labels):
    import random
    data = list(zip(files, labels))
    random.shuffle(data)
    files[:], labels[:] = zip(*data)


def load_data(data_dir):
    data = []
    label = []
    total_num = 0
    for label_type in ['Readable', 'Neutral', 'Unreadable']:
        dir_name = os.path.join(data_dir, label_type)
        file_list = os.listdir(dir_name)
        if dir_name == 'Neutral':
            file_list.sort()
            file_list = file_list[0:len(file_list) // 2]
        for f_name in file_list:
            f = open(os.path.join(dir_name, f_name), errors='ignore')
            lines = []
            if not f_name.startswith('.'):
                total_num += 1
                for line in f:
                    line = line.strip(',\n')
                    info = line.split(',')
                    info_int = []
                    count = 0
                    for item in info:
                        if count < 305:
                            info_int.append(int(item))
                            count += 1
                    while count < 305:
                        info_int.append(int(-1))
                        count += 1
                    info_int = np.asarray(info_int)
                    lines.append(info_int)

                while len(lines) < 50:
                    info_int = []
                    count = 0
                    for i in range(305):
                        if count < 305:
                            info_int.append(int(-1))
                            count += 1
                    info_int = np.asarray(info_int)
                    lines.append(info_int)
                f.close()
                lines = np.asarray(lines)
                if label_type == 'Readable':
                    label.append(2)
                elif label_type == 'Unreadable':
                    label.append(0)
                else:
                    label.append(1)
                data.append(lines)

    data = np.asarray(data)
    data = data.reshape((total_num, 50, 305, 1)) / 127
    label = np.asarray(label)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', label.shape)
    label = to_one_hot(label, 3)
    return data, label


if __name__ == '__main__':
    test_set = '../../Dataset/original_dataset/validate'
    x_test, y_test = load_data(test_set)
    shuffle_list(x_test, y_test)
    data_set = '../../Dataset/RQ2/2gan'
    X, Y = load_data(data_set)
    shuffle_list(X, Y)
    print('Shape of train data tensor:', X.shape)
    print('Shape of validate data tensor:', x_test.shape)
    X = np.append(X, np.append(X, X, axis=3), axis=3)
    x_test = np.append(x_test, np.append(x_test, x_test, axis=3), axis=3)
    print('Shape of train data tensor:', X.shape)
    print('Shape of validate data tensor:', x_test.shape)

    resNetModel = Sequential()
    resNetModel.add(VGG19(weights='imagenet', include_top=False, input_shape=(50, 305, 3)))
    resNetModel.add(Flatten())
    resNetModel.add(Dense(units=128, activation='relu', kernel_regularizer=tk.regularizers.l2(0.001)))
    resNetModel.add(Dropout(0.5))
    resNetModel.add(Dense(units=16, activation='relu'))
    resNetModel.add(Dense(3, activation='sigmoid'))
    rms = RMSprop(learning_rate=0.0025)
    # resNetModel.summary()
    resNetModel.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy', f1])

    filepath = "../best_result1.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                 model='max')
    callbacks_list = [checkpoint]
    resNetHistory = resNetModel.fit(X, Y, epochs=15, batch_size=32, verbose=0, callbacks=callbacks_list,
                                    validation_data=(x_test, y_test))

    resNetModel.load_weights(filepath)
    y_pre = resNetModel.predict(x_test)

    print('MobileNetV2')

    print(roc_auc_score(np.asarray(y_test), np.asarray(y_pre), multi_class='ovo'))
    print(roc_auc_score(np.asarray(y_test), np.asarray(y_pre), average='micro', multi_class='ovo'))
    #
    print(roc_auc_score(np.asarray(y_test), np.asarray(y_pre), multi_class='ovr'))
    print(roc_auc_score(np.asarray(y_test), np.asarray(y_pre), average='micro', multi_class='ovr'))

    y_result = []
    y_test_array = []
    for item in y_pre:
        max_value = -1
        i = 0
        max_index = 0
        while i < 3:
            if item[i] > max_value:
                max_value = item[i]
                max_index = i
            i += 1
        y_result.append(max_index)

    for item in y_test:
        max_value = -1
        i = 0
        max_index = 0
        while i < 3:
            if item[i] > max_value:
                max_value = item[i]
                max_index = i
            i += 1
        y_test_array.append(max_index)

    print('micro f1 score:', f1_score(y_test_array, y_result, average='micro'))
    print('macro f1 score:', f1_score(y_test_array, y_result, average='macro'))

    true = 0
    false = 0
    for i in range(len(y_result)):
        if y_result[i] == y_test_array[i]:
            true += 1
        else:
            false += 1
    print(true / (true + false))
    print('acc:', accuracy_score(y_test_array, y_result))