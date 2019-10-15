import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import time


class ACGan(object):
    def __init__(self, input_shape=(28, 28, 1), num_of_classes=10, num_of_noises=100, batch_size=32):
        self.input_shape = input_shape
        self.num_of_classes = num_of_classes
        self.num_of_noises = num_of_noises
        self.batch_size = batch_size

        self.noise_input = keras.Input(shape=(self.num_of_noises,), name="Noise")
        self.label_input = keras.Input(shape=(1,), dtype="int32", name="Label")
        self.losses = ["binary_crossentropy", "sparse_categorical_crossentropy"]

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=self.losses,
            optimizer="adam",
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
            optimizer="adam",
            metrics=["accuracy"]
        )
        self.model.summary()
        keras.utils.plot_model(self.model, "models/GAN_Model.png", show_shapes=True, show_layer_names=True, expand_nested=False)

    def build_discriminator(self)->keras.Model:
        input_1 = keras.Input(self.input_shape, name="images")
        x = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        )(input_1)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        )(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        )(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(512, activation="relu")(x)
        output_1 = keras.layers.Dense(1, activation="sigmoid", name="validity")(x)
        output_2 = keras.layers.Dense(self.num_of_classes, activation="softmax", name="classes")(x)

        model = keras.Model(input_1, [output_1, output_2], name="Discriminator")
        model.summary()
        keras.utils.plot_model(model, "models/discriminator.png", show_shapes=True, show_layer_names=True, expand_nested=False)
        return model

    def build_generator(self)->keras.Model:
        w, h = self.input_shape[0] // 4, self.input_shape[1] // 4
        x = keras.layers.Embedding(self.num_of_classes, self.num_of_noises)(self.label_input)
        x = keras.layers.Flatten()(x)
        x = keras.layers.multiply([self.noise_input, x])
        # x = keras.layers.Dense(1024, activation="relu")(x)
        x = keras.layers.Dense(128 * w * h, activation="relu")(x)
        x = keras.layers.Reshape((w, h, 128))(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        )(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(
            filters=self.input_shape[2],
            kernel_size=(3, 3),
            padding="same",
            activation="tanh",
            name="Fake_Image"
        )(x)

        model = keras.Model([self.noise_input, self.label_input], x, name="Generator")
        model.summary()
        keras.utils.plot_model(model, "models/generator.png", show_shapes=True, show_layer_names=True, expand_nested=False)
        return model

    def save_model(self):
        self.discriminator.save("models/discriminator.h5", save_format="tf")
        self.generator.save("models/generator.h5", save_format="tf")

    def write_log(self, callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()

    def save_samples(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, self.num_of_noises))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
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
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == "__main__":
    (X_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
    X_train = X_train.astype(np.float32) / 255 * 2 - 1
    # X_train = np.expand_dims(X_train, axis=3)
    print(X_train.shape, y_train.shape)

    gan = ACGan(input_shape=(32, 32, 3), num_of_classes=10)
    validity_real = np.ones((gan.batch_size, 1))
    validity_fake = np.zeros((gan.batch_size, 1))

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
        idx = np.random.randint(0, X_train.shape[0], gan.batch_size)
        X_batch = X_train[idx]
        y_batch = y_train[idx]

        noise = np.random.normal(0, 1, (gan.batch_size, gan.num_of_noises))
        y_generated = np.random.randint(0, gan.num_of_classes, (gan.batch_size, 1))
        X_generated = gan.generator([noise, y_generated])

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

        # print(d_loss_real, d_loss_fake)
        # print(g_loss)
        print("Epoch: %d -- D_loss: %.4f, D_acc1: %.2f, D_acc2: %.2f -- G_loss: %.4f, G_acc1: %.2f, G_acc2: %.2f" % (
            epoch,
            (d_loss_real[0] + d_loss_fake[0]) / 2,
            (d_loss_real[3] + d_loss_fake[3]) / 2,
            (d_loss_real[4] + d_loss_fake[4]) / 2,
            g_loss[0],
            g_loss[3],
            g_loss[4]
        ))
        data = {
            "D_loss": (d_loss_real[0] + d_loss_fake[0]) / 2,
            "D_acc1": (d_loss_real[3] + d_loss_fake[3]) / 2,
            "D_acc2": (d_loss_real[4] + d_loss_fake[4]) / 2,
            "G_loss": g_loss[0],
            "G_acc1": g_loss[3],
            "G_acc2": g_loss[4]
        }
        t_board.on_epoch_end(epoch, data)

        if epoch % 100 == 0:
            gan.save_samples(epoch)

        epoch += 1
