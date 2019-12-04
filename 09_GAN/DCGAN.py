from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np
import os
from tensorflow import keras
import cv2


class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.001)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU())
        model.add(Dropout(0.1))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        # model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.1))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.1))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        X_train = []
        # y_train = []
        image_path = "../../datasets/animefacedataset/images/"
        image_path_list = os.listdir(image_path)
        for i, p in enumerate(image_path_list):
            path = os.path.join(image_path, p)
            img = keras.preprocessing.image.load_img(path, target_size=(64, 64))
            img = keras.preprocessing.image.img_to_array(img)
            X_train.append(img.astype(np.float32) / 255 * 2 - 1)
            # y_train.append(0)
            print(i + 1, "/", len(image_path_list))
            if i == 50000:
                break
        X_train = np.array(X_train)
        # y_train = np.array(y_train)
        # y_train = np.zeros((X_train.shape[0], 1))

        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        flag = True
        frame = np.ones((480, 640, 3))
        # cv2.imshow("frame", frame)
        # cv2.waitKey(0)

        for epoch in range(1, epochs):

            idx = np.arange(0, X_train.shape[0])
            np.random.shuffle(idx)
            cur_batch = 0
            max_batch = X_train.shape[0] // batch_size
            d_loss = [0, 0]
            while cur_batch < max_batch:
                idx_begin = cur_batch * batch_size
                idx_end = min(idx_begin + batch_size, idx.shape[0])
                imgs = X_train[idx[idx_begin:idx_end]]

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                if flag:
                    d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                    d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                g_loss = self.combined.train_on_batch(noise, valid)

                if d_loss[1] > 0.9 > g_loss[1]:
                    flag = False
                else:
                    flag = True

                if g_loss[1] > 0.9:
                    self.save_imgs(epoch)
                    # frame = cv2.imread("images/%d.png" % epoch, cv2.IMREAD_UNCHANGED)
                # cv2.imshow("frame", frame)
                # cv2.waitKey(1)

                # Plot the progress
                cur_batch += 1
                print ("%d (%d / %d) [D loss: %f, acc.: %.2f%%] [G loss: %f, acc.: %.2f%%]" % (epoch, cur_batch, max_batch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1] * 100))

            # If at save interval => save generated images samples
            # if epoch % save_interval == 0:
            #     self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 4, 8
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=128, save_interval=1)
