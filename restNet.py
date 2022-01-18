import os

import numpy as np
import tensorflow.keras as tk
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import RMSprop


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


def to_one_hot(data_labels, dimension=3):
    results = np.zeros((len(data_labels), dimension))
    for i, data_labels in enumerate(data_labels):
        results[i, data_labels] = 1
    return results


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


def classifier():
    model = Sequential()
    model.add(Reshape((50, 305, 1), input_shape=(50, 305)))

    model.add(Convolution2D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Convolution2D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Convolution2D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=3))

    model.add(Flatten())
    model.add(Dense(units=128, activation='relu', kernel_regularizer=tk.regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))

    rms = RMSprop(lr=0.0015)

    # 定义优化器，代价函数，训练过程中的准确率
    model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy', f1])
    return model


if __name__ == '__main__':
    test_set = 'validate'
    x_test, y_test = load_data(test_set)
    shuffle_list(x_test, y_test)
    data_set = 'train'
    X, Y = load_data(data_set)
    shuffle_list(X, Y)
    print('Shape of train data tensor:', X.shape)
    print('Shape of validate data tensor:', x_test.shape)
    X = np.append(X, np.append(X, X, axis=3), axis=3)
    x_test = np.append(x_test, np.append(x_test, x_test, axis=3), axis=3)
    print('Shape of train data tensor:', X.shape)
    print('Shape of validate data tensor:', x_test.shape)

    restNetModel = Sequential()
    restNetModel.add(ResNet50(weights='imagenet', include_top=False, input_shape=(50, 305, 3)))
    restNetModel.add(Flatten())
    restNetModel.add(Dense(units=128, activation='relu', kernel_regularizer=tk.regularizers.l2(0.001)))
    restNetModel.add(Dropout(0.5))
    restNetModel.add(Dense(units=16, activation='relu'))
    restNetModel.add(Dense(3, activation='sigmoid'))
    rms = RMSprop(lr=0.0015)
    restNetModel.summary()
    restNetModel.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy', f1])
    restNetHistory = restNetModel.fit(X, Y, epochs=40, batch_size=32, verbose=0,
                                      validation_data=(x_test, y_test))
    print('train')
    print('最大精度:' + str(max(restNetHistory.history['val_accuracy'])))
    print('最大f1:' + str(max(restNetHistory.history['val_f1'])))

    VGG16Model = Sequential()
    VGG16Model.add(VGG16(weights='imagenet', include_top=False, input_shape=(50, 305, 3)))
    VGG16Model.add(Flatten())
    VGG16Model.add(Dense(units=128, activation='relu', kernel_regularizer=tk.regularizers.l2(0.001)))
    VGG16Model.add(Dropout(0.5))
    VGG16Model.add(Dense(units=16, activation='relu'))
    VGG16Model.add(Dense(3, activation='sigmoid'))
    rms = RMSprop(lr=0.0015)
    VGG16Model.summary()
    VGG16Model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy', f1])
    VGG16History = VGG16Model.fit(X, Y, epochs=40, batch_size=32, verbose=0,
                                  validation_data=(x_test, y_test))
    print('train')
    print('最大精度:' + str(max(VGG16History.history['val_accuracy'])))
    print('最大f1:' + str(max(VGG16History.history['val_f1'])))


    VGG19Model = Sequential()
    VGG19Model.add(VGG19(weights='imagenet', include_top=False, input_shape=(50, 305, 3)))
    VGG19Model.add(Flatten())
    VGG19Model.add(Dense(units=128, activation='relu', kernel_regularizer=tk.regularizers.l2(0.001)))
    VGG19Model.add(Dropout(0.5))
    VGG19Model.add(Dense(units=16, activation='relu'))
    VGG19Model.add(Dense(3, activation='sigmoid'))
    rms = RMSprop(lr=0.0015)
    VGG19Model.summary()
    VGG19Model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy', f1])
    VGG19History = VGG19Model.fit(X, Y, epochs=40, batch_size=32, verbose=0,
                                  validation_data=(x_test, y_test))
    print('train')
    print('最大精度:' + str(max(VGG19History.history['val_accuracy'])))
    print('最大f1:' + str(max(VGG19History.history['val_f1'])))
