import os

import numpy as np
import tensorflow.keras as tk
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint


def delta(list1, list2):
    len1 = len(list1)
    len2 = len(list2)
    large_num = 0
    small_num = 0
    for item1 in list1:
        for item2 in list2:
            if item1 > item2:
                large_num += 1
            elif item1 < item2:
                small_num += 1
    return (large_num - small_num) / (len1 * len2)


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

    rms = RMSprop(learning_rate=0.0025)

    # 定义优化器，代价函数，训练过程中的准确率
    model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy', f1])
    return model


if __name__ == '__main__':
    data_set = '../Dataset/RQ1/domain'
    X, Y = load_data(data_set)
    shuffle_list(X, Y)
    test_set = '../Dataset/original_dataset/validate'
    x_test, y_test = load_data(test_set)
    shuffle_list(x_test, y_test)
    print('Shape of train data tensor:', X.shape)
    print('Shape of validate data tensor:', x_test.shape)

    model1 = classifier()
    filepath = "../best_result.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                 model='max')
    callbacks_list = [checkpoint]
    history = model1.fit(X, Y, epochs=50, batch_size=32, callbacks=callbacks_list, verbose=0,
                         validation_data=(x_test, y_test))

    print('domain result:')
    print('最大精度:' + str(max(history.history['val_accuracy'])))
    print('最大f1:' + str(max(history.history['val_f1'])))
