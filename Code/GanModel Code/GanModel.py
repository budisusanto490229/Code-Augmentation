import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Conv2D, Reshape, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp

train_epoch = 200


def to_one_hot(data_labels, dimension=3):
    results = np.zeros((len(data_labels), dimension))
    for i, data_labels in enumerate(data_labels):
        results[i, data_labels] = 1
    return results


def load_data(data_dir):
    data = []
    label = []
    data_neutral = []
    data_readable = []
    data_unreadable = []
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
                    data_readable.append(lines)
                elif label_type == 'Unreadable':
                    label.append(0)
                    data_unreadable.append(lines)
                else:
                    label.append(1)
                    data_neutral.append(lines)
                data.append(lines)

    data = np.asarray(data)
    data = data.reshape((total_num, 50, 305, 1)) / 127
    data_neutral = np.asarray(data_neutral)
    data_neutral = data_neutral.reshape((total_num // 2, 50, 305, 1)) / 127
    data_readable = np.asarray(data_readable)
    data_readable = data_readable.reshape((total_num // 4, 50, 305, 1)) / 127
    data_unreadable = np.asarray(data_unreadable)
    data_unreadable = data_unreadable.reshape((total_num // 4, 50, 305, 1)) / 127
    label = np.asarray(label)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', label.shape)
    return data, label, data_neutral, data_readable, data_unreadable


def classifier():
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=3, activation='relu', input_shape=(50, 305, 1)))
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

    rms = RMSprop(learning_rate=0.0015)

    # 定义优化器，代价函数，训练过程中的准确率
    model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def shuffle_list(files, labels):
    import random
    data = list(zip(files, labels))
    random.shuffle(data)
    files[:], labels[:] = zip(*data)


tf = tf.compat.v1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


class WGAN():

    def __init__(self):
        self.img_rows = 50
        self.img_cols = 305
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 50

        # Following parameter and optimizer set as recommended in paper
        # 优化参数
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSProp(learning_rate=0.0005)  # 0.00005

        # Build and compile the critic 建造编译判别器
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])

        # 建造生成器
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        # 输入噪声得到图像
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        # 到此仅训练判别器
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        # 得到判别结果
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        # 编译模型（生成器和判别器的堆叠）
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    # EM损失 就是 Wasserstein loss
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)  #

    def build_generator(self):

        model = Sequential()

        model.add(Dense(50 * 305, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((50, 305, 1)))
        #        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        #        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, x, category, epochs, batch_size=128, sample_interval=1):

        # Rescale -1 to 1
        X_train = (np.array(x).astype(np.float32) - 63) / 128
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths 真实值
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights 权重消减
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, category)

    def sample_images(self, epoch, category):
        save_dir = '../../Dataset/generated_dataset/details'
        if os.path.exists(save_dir) == 0:
            os.makedirs(save_dir)
        filename = save_dir + '/' + str(epoch) + category + '.java.matrix'
        r, c = 1, 1
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = np.array((gen_imgs + 1) * 64 - 1)
        gen_imgs = gen_imgs.reshape((1, 50, 305))
        gen_imgs = np.array(gen_imgs).astype('int32')
        for i in range(len(gen_imgs[0])):
            result_data = ''
            for a in range(len(gen_imgs[0][i])):
                result_data += str(gen_imgs[0][i][a]) + ','
            if i == 0:
                open(filename, 'w').write(result_data)
            else:
                result_data = '\n' + result_data
                open(filename, 'a').write(result_data)


if __name__ == '__main__':
    data_dir = '../../Dataset/original_dataset/train'
    data_set, label_set, data_neutral, data_readable, data_unreadabale = load_data(data_dir)
    label_neutral = [1 for i in range(len(data_neutral))]
    label_readable = [2 for i in range(len(data_readable))]
    label_unreadable = [0 for i in range(len(data_unreadabale))]
    label_set = to_one_hot(label_set, 3)
    label_neutral = to_one_hot(label_neutral, 3)
    label_readable = to_one_hot(label_readable, 3)
    label_unreadable = to_one_hot(label_unreadable, 3)
    x_train = []
    y_train = []

    neutral_gan = WGAN()
    neutral_gan.train(x=data_neutral, category='neutral', epochs=train_epoch, batch_size=32, sample_interval=1)
    for num in range(0, 1000, 1):
        noise = np.random.normal(0, 1, (1, 50))
        gen_data = neutral_gan.generator.predict(noise)
        gen_data = np.array((gen_data + 1) * 64 - 1)
        gen_data = gen_data.reshape((1, 50, 305))
        gen_data = np.array(gen_data).astype('int32')
        x_train.append(gen_data)
        y_train.append([0, 1, 0])
        for i in range(len(gen_data[0])):
            store_data = ''
            for a in range(len(gen_data[0][i])):
                store_data += str(gen_data[0][i][a]) + ','
            if i == 0:
                open('../../Dataset/generated_dataset/Neutral/' + str(num) + 'neutral.txt', 'w').write(store_data)
            else:
                store_data = '\n' + store_data
                open('../../Dataset/generated_dataset/Neutral/' + str(num) + 'neutral.txt', 'a').write(store_data)

    readable_gan = WGAN()
    readable_gan.train(x=data_readable, category='readable', epochs=train_epoch, batch_size=32, sample_interval=1)
    for num in range(0, 500, 1):
        noise = np.random.normal(0, 1, (1, 50))
        gen_data = readable_gan.generator.predict(noise)
        gen_data = np.array((gen_data + 1) * 64 - 1)
        gen_data = gen_data.reshape((1, 50, 305))
        gen_data = np.array(gen_data).astype('int32')
        x_train.append(gen_data)
        y_train.append([0, 0, 1])
        for i in range(len(gen_data[0])):
            store_data = ''
            for a in range(len(gen_data[0][i])):
                store_data += str(gen_data[0][i][a]) + ','
            if i == 0:
                open('../../Dataset/generated_dataset/Readable/' + str(num) + 'readable.txt', 'w').write(store_data)
            else:
                store_data = '\n' + store_data
                open('../../Dataset/generated_dataset/Readable/' + str(num) + 'readable.txt', 'a').write(store_data)

    unreadable_gan = WGAN()
    unreadable_gan.train(x=data_unreadabale, category='unreadable', epochs=train_epoch, batch_size=32,
                         sample_interval=1)
    for num in range(0, 500, 1):
        noise = np.random.normal(0, 1, (1, 50))
        gen_data = unreadable_gan.generator.predict(noise)
        gen_data = np.array((gen_data + 1) * 64 - 1)
        gen_data = gen_data.reshape((1, 50, 305))
        gen_data = np.array(gen_data).astype('int32')
        x_train.append(gen_data)
        y_train.append([1, 0, 0])
        for i in range(len(gen_data[0])):
            store_data = ''
            for a in range(len(gen_data[0][i])):
                store_data += str(gen_data[0][i][a]) + ','
            if i == 0:
                open('../../Dataset/generated_dataset/Unreadable/' + str(num) + 'unreadable.txt', 'w').write(store_data)
            else:
                store_data = '\n' + store_data
                open('../../Dataset/generated_dataset/Unreadable/' + str(num) + 'unreadable.txt', 'a').write(store_data)

