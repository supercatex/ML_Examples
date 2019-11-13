from tensorflow import keras
from tensorflow.keras import layers


class DCGAN(object):
    def __init__(self, input_shape=(28, 28, 1)):
        self.input_shape = input_shape
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        self.generator = self.build_generator()
        self.generator.summary()

    def build_discriminator(self)->keras.Model:
        input_1 = keras.Input(self.input_shape)

        x = layers.Conv2D(32, 3, 1, "same")(input_1)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(64, 3, 1, "same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Flatten()(x)

        output_1 = layers.Dense(1, keras.activations.sigmoid)(x)
        return keras.Model(input_1, output_1)

    def build_generator(self)->keras.Model:
        w, h = self.input_shape[0] // 4, self.input_shape[1] // 4
        c = self.input_shape[2]

        input_1 = keras.Input((1,))
        input_2 = keras.Input((100,))
        x = keras.layers.Embedding(10, 100)(input_1)
        x = keras.layers.Flatten()(x)
        x = keras.layers.multiply([input_2, x])
        return keras.Model([input_1, input_2], x)

if __name__ == "__main__":
    gan = DCGAN()
