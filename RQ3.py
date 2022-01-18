import os

import numpy as np
import scipy.stats
import tensorflow.keras as tk
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import f1_score


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
            total_num += 1
            f = open(os.path.join(dir_name, f_name), errors='ignore')
            lines = []
            if not f_name.startswith('.'):
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
    model.add(Dense(units=32, activation='relu'))
    rms = RMSprop(lr=0.0015)

    # 定义优化器，代价函数，训练过程中的准确率
    model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    original_set = 'train'
    X, Y = load_data(original_set)
    shuffle_list(X, Y)
    augment_set = 'RQ1/'
    X1, Y1 = load_data(augment_set)
    shuffle_list(X1, Y1)

    test_set = 'I:/Code Dataset/判别网络beta/判别网络beta/验证集'
    x_test, y_test = load_data(test_set)
    shuffle_list(x_test, y_test)
    print('Shape of train data tensor:', X.shape)
    print('Shape of validate data tensor:', x_test.shape)

    forest_acc_results = []
    forest_f1_results = []

    forest_acc_results1 = []
    forest_f1_results1 = []

    knn_acc_result = []
    knn_f1_result = []

    knn_acc_result1 = []
    knn_f1_result1 = []

    for i in range(10):
        model = classifier()
        train_feature = []
        val_feature = []
        train_feature1 = []
        for feature in model.predict(X):
            train_feature.append(feature)

        for feature in model.predict(X1):
            train_feature1.append(feature)

        for feature in model.predict(x_test):
            val_feature.append(feature)

        train_feature = np.asarray(train_feature)
        train_feature1 = np.asarray(train_feature1)

        for feature_num in range(2, 80, 2):
            forest1 = RandomForestClassifier(n_estimators=feature_num, random_state=20, bootstrap=True,
                                             max_features='sqrt', warm_start=True)
            forest1.fit(train_feature, Y)
            result1 = forest1.predict(val_feature)
            forest_acc_results.append(forest1.score(val_feature, y_test))
            f1 = f1_score(y_test, result1, average='weighted')
            forest_f1_results.append(f1)

            forest2 = RandomForestClassifier(n_estimators=feature_num, random_state=20, bootstrap=True,
                                             max_features='sqrt', warm_start=True)
            forest2.fit(train_feature1, Y1)
            result2 = forest2.predict(val_feature)
            forest_acc_results1.append(forest2.score(val_feature, y_test))
            f1 = f1_score(y_test, result2, average='weighted')
            forest_f1_results1.append(f1)

        for n_neighbors in range(1, 20, 1):
            knn1 = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn1.fit(train_feature, Y)
            result1 = knn1.predict(val_feature)
            knn_acc_result.append(knn1.score(val_feature, y_test))
            f1 = f1_score(y_test, result1, average='weighted')
            knn_f1_result.append(f1)

            knn2 = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn2.fit(train_feature1, Y1)
            result2 = knn2.predict(val_feature)
            knn_acc_result1.append(knn2.score(val_feature, y_test))
            f1 = f1_score(y_test, result2, average='weighted')
            knn_f1_result1.append(f1)

    print(dir_name)
    print('random forest acc result:', np.max(forest_acc_results))
    print('random forest f1 result:', np.max(forest_f1_results))

    print('knn acc result:', np.max(knn_acc_result))
    print('knn f1 result:', np.max(knn_f1_result))

    print(dir_name1)
    print('random forest result:', np.max(forest_acc_results1))
    print('random forest f1 result:', np.max(forest_f1_results1))

    print('knn result:', np.max(knn_acc_result1))
    print('knn f1 result:', np.max(knn_f1_result1))

    print('数据统计:')

    print('RF acc improve:', np.max(forest_acc_results1) - np.max(forest_acc_results))
    print('RF f1 improve:', np.max(forest_f1_results1) - np.max(forest_f1_results))
    print('KNN acc improve:', np.max(knn_acc_result1) - np.max(knn_acc_result))
    print('KNN f1 improve:', np.max(knn_f1_result1) - np.max(knn_f1_result))

