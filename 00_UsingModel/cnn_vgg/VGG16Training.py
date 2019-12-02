from keras.applications import VGG16
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt


class VGG16Training(object):
    def __init__(self, image_shape, train_path, valid_path=None, batch_size=128, trainable_layers=4):
        self.image_shape = image_shape
        self.trainable_layers = trainable_layers
        self.batch_size = batch_size
        self.train_path = train_path
        self.valid_path = valid_path

        vgg = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.image_shape
        )
        for layer in vgg.layers[:-self.trainable_layers]:
            layer.trainable = False

        self.train_data_gen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        self.train_generator = self.train_data_gen.flow_from_directory(
            self.train_path,
            target_size=self.image_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        if self.valid_path is not None:
            self.valid_data_gen = ImageDataGenerator(rescale=1.0 / 255)
            self.validation_generator = self.valid_data_gen.flow_from_directory(
                self.valid_path,
                target_size=self.image_shape[:2],
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )

        input_shape = layers.Input(shape=self.image_shape, name="image_shape")
        x = vgg(input_shape)
        x = layers.Flatten()(x)
        output = layers.Dense(units=self.train_generator.num_classes, activation="softmax", name="softmax")(x)
        self.model = models.Model(input_shape, output)
        self.model.summary()
        plot_model(self.model, "VGG16_model.png", show_shapes=True, show_layer_names=True, expand_nested=True)

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizers.RMSprop(lr=1e-4),
            metrics=['accuracy']
        )

    def train(self, epochs=10, save_path=None):
        if self.valid_path is not None:
            self.history = self.model.fit_generator(
                self.train_generator,
                steps_per_epoch=self.train_generator.samples // self.train_generator.batch_size,
                epochs=epochs,
                validation_data=self.validation_generator,
                validation_steps=self.validation_generator.samples // self.validation_generator.batch_size,
                verbose=1
            )
        else:
            self.history = self.model.fit_generator(
                self.train_generator,
                steps_per_epoch=self.train_generator.samples // self.train_generator.batch_size,
                epochs=epochs,
                verbose=1
            )
        if save_path is not None:
            self.model.save(save_path)

    def show_history(self, accuracy_path=None, loss_path=None):
        history = self.history.history

        acc = history['accuracy']
        loss = history['loss']
        epochs = range(len(acc))

        plt.figure()
        plt.plot(epochs, acc, 'b', label='Training acc')
        if "val_accuracy" in history:
            plt.plot(epochs, history["val_accuracy"], 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        if accuracy_path is not None:
            plt.savefig(accuracy_path)

        plt.figure()
        plt.plot(epochs, loss, 'b', label='Training loss')
        if "val_loss" in history:
            plt.plot(epochs, history["val_loss"], 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        if loss_path is not None:
            plt.savefig(loss_path)
        plt.show()



if __name__ == "__main__":
    model = VGG16Training((100, 100, 3), "../../../datasets/pcms/features/", "../../../datasets/pcms/features/")
    model.train(epochs=100, save_path="model.h5")
    model.show_history("history_acc.jpg", "history_loss.jpg")
    print(model.train_generator.class_indices)
