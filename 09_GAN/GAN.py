import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# https://arxiv.org/pdf/1610.09585.pdf


class ACGan(object):
    def __init__(self, input_shape=(28, 28, 1), num_of_classes=10, num_of_noises=100, batch_size=32):
        self.input_shape = input_shape
        self.num_of_classes = num_of_classes
        self.num_of_noises = num_of_noises
        self.batch_size = batch_size

        self.noise_input = keras.Input(shape=(self.num_of_noises,), name="Noise")
        self.label_input = keras.Input(shape=(1,), dtype="int32", name="Label")
        self.losses = ["binary_crossentropy", "sparse_categorical_crossentropy"]
        self.optimizer = keras.optimizers.Adam(0.0005)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=self.losses,
            optimizer=self.optimizer,
            metrics=["accuracy"]
        )

        self.generator = self.build_generator()

        fake_img_tensor = self.generator([self.noise_input, self.label_input])
        self.discriminator.trainable = False
        validity, label = self.discriminator(fake_img_tensor)
        self.model = keras.Model(
            [self.noise_input, self.label_input],
            [validity, label],
            name="GAN_Model"
        )
        self.model.compile(
            loss=self.losses,
            optimizer=self.optimizer,
            metrics=["accuracy"]
        )
        self.model.summary()
        # keras.utils.plot_model(self.model, "images/GAN_Model.png", show_shapes=True, show_layer_names=True)

    def build_discriminator(self)->keras.Model:
        input_1 = keras.Input(self.input_shape, name="images")
        x = keras.layers.Conv2D(32, (5, 5), (2, 2), "same")(input_1)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Dropout(0.1)(x)

        x = keras.layers.Conv2D(64, (3, 3), (2, 2), "same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Dropout(0.1)(x)

        x = keras.layers.Flatten()(x)

        output_1 = keras.layers.Dense(1, activation="sigmoid", name="validity")(x)
        output_2 = keras.layers.Dense(self.num_of_classes, activation="softmax", name="classes")(x)

        model = keras.Model(input_1, [output_1, output_2], name="Discriminator")
        model.summary()
        # keras.utils.plot_model(model, "images/discriminator.png", show_shapes=True, show_layer_names=True)
        return model

    def build_generator(self)->keras.Model:
        w, h = self.input_shape[0] // 8, self.input_shape[1] // 8
        c = self.input_shape[2]

        x = keras.layers.Embedding(self.num_of_classes, self.num_of_noises)(self.label_input)
        x = keras.layers.Flatten()(x)
        x = keras.layers.multiply([self.noise_input, x])

        x = keras.layers.Dense(256 * w * h, use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

        x = keras.layers.Reshape((w, h, 256))(x)

        x = keras.layers.Conv2DTranspose(128, (5, 5), (2, 2), "same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

        x = keras.layers.Conv2DTranspose(64, (5, 5), (2, 2), "same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

        x = keras.layers.Conv2DTranspose(c, (5, 5), (2, 2), "same", use_bias=False, activation="tanh", name="Fake_Image")(x)

        model = keras.Model([self.noise_input, self.label_input], x, name="Generator")
        model.summary()
        # keras.utils.plot_model(model, "images/generator.png", show_shapes=True, show_layer_names=True)
        return model

    def save_model(self):
        self.discriminator.save("models/discriminator.h5")
        self.generator.save("models/generator.h5")

    def save_samples(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, self.num_of_noises))
        sampled_labels = np.array([0 for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                if self.input_shape[2] == 1:
                    axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap="gray")
                else:
                    axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch, quality=95)
        plt.close()
        return gen_imgs


if __name__ == "__main__":
    import os
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
    X_train = np.array(X_train)
    # y_train = np.array(y_train)
    y_train = np.zeros((X_train.shape[0], 1))

    # (X_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
    # X_train = X_train.astype(np.float32) / 255 * 2 - 1
    # X_train = X_train / 255 * 2 - 1
    if len(X_train.shape) == 3:
        X_train = np.expand_dims(X_train, axis=3)
    print(X_train.shape, y_train.shape)

    gan = ACGan(input_shape=X_train.shape[1:], num_of_classes=1, batch_size=128)
    validity_real = np.ones((gan.batch_size, gan.num_of_classes))
    validity_fake = np.zeros((gan.batch_size, gan.num_of_classes))

    epoch = 0
    log_path = "logs"
    t_board = keras.callbacks.TensorBoard(
        log_dir='{}'.format(log_path),
        histogram_freq=0,
        batch_size=gan.batch_size,
        write_graph=True,
        write_grads=True,
        write_images=True
    )
    t_board.set_model(gan.model)
    while True:
        idx = np.arange(0, X_train.shape[0])
        np.random.shuffle(idx)
        cur_batch = 0
        max_batch = X_train.shape[0] // gan.batch_size
        while cur_batch < max_batch:
            idx_begin = cur_batch * gan.batch_size
            idx_end = min(idx_begin + gan.batch_size, idx.shape[0])
            X_batch = X_train[idx[idx_begin:idx_end]]
            y_batch = y_train[idx[idx_begin:idx_end]]

            noise = np.random.normal(0, 1, (gan.batch_size, gan.num_of_noises))
            # y_generated = np.random.randint(0, gan.num_of_classes, (gan.batch_size, 1))
            y_generated = np.zeros((gan.batch_size, 1))
            # print(noise.shape, y_generated.shape, noise.dtype)
            X_generated = gan.generator.predict([noise, y_generated])

            d_loss_real = gan.discriminator.train_on_batch(
                X_batch,
                [validity_real, y_batch]
            )
            d_loss_fake = gan.discriminator.train_on_batch(
                X_generated,
                [validity_fake, y_generated]
            )
            g_loss = gan.model.train_on_batch(
                [noise, y_generated],
                [validity_real, y_generated]
            )

            data = {
                "loss_DR": d_loss_real[0],
                "DR_acc1": d_loss_real[3],
                "DR_acc2": d_loss_real[4],
                "loss_DF": d_loss_fake[0],
                "DF_acc1": d_loss_fake[3],
                "DF_acc2": d_loss_fake[4],
                "loss_G": g_loss[0],
                "G_acc1": g_loss[3],
                "G_acc2": g_loss[4]
            }
            t_board.on_epoch_end(epoch * max_batch + cur_batch, data)

            print("Epoch: %d (%d/%d) -- DR_acc1: %.2f, DR_acc2: %.2f -- DF_acc1: %.2f, DF_acc2: %.2f -- G_acc1: %.2f, G_acc2: %.2f" % (
                epoch, cur_batch + 1, max_batch,
                data["DR_acc1"], data["DR_acc2"],
                data["DF_acc1"], data["DF_acc2"],
                data["G_acc1"], data["G_acc2"]
            ))

            cur_batch += 1

        if epoch % 1 == 0:
            gan.save_samples(epoch)
        if epoch % 10 == 0:
            gan.save_model()

        epoch += 1
