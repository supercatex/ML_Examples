from keras.applications import VGG16
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf


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
            name = ".".join(save_path.split(".")[:-1])

            self.model.save(save_path)

            f = open(name + ".txt", "w")
            for label in self.train_generator.class_indices:
                f.write(label + "\n")
            f.close()

            plot_model(self.model, name + ".png", show_shapes=True, show_layer_names=True, expand_nested=True)

            # frozen_graph = self.freeze_session(
            #     K.get_session(),
            #     output_names=[out.op.name for out in self.model.outputs]
            # )
            # import tensorflow.tools.graph_transforms as graph_transforms
            #
            # frozen_graph = graph_transforms.TransformGraph(frozen_graph,
            #     ["input_1"],  # inputs nodes
            #     ["dense_2/Softmax"],  # outputs nodes
            #     ['fold_constants()',
            #      'strip_unused_nodes(type=float, shape="None,32,32,1")',
            #      'remove_nodes(op=Identity, op=CheckNumerics)',
            #      'fold_batch_norms',
            #      'fold_old_batch_norms'
            #      ]
            # )
            # tf.train.write_graph(frozen_graph, "", "my_model.pb", as_text=False)
            # print(frozen_graph)

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

    def freeze_session(self, session, keep_var_names=None, output_names=None, clear_devices=True):
        """
        Freezes the state of a session into a pruned computation graph.

        Creates a new computation graph where variable nodes are replaced by
        constants taking their current value in the session. The new graph will be
        pruned so subgraphs that are not necessary to compute the requested
        outputs are removed.
        @param session The TensorFlow session to be frozen.
        @param keep_var_names A list of variable names that should not be frozen,
                              or None to freeze all the variables in the graph.
        @param output_names Names of the relevant graph outputs.
        @param clear_devices Remove the device directives from the graph for better portability.
        @return The frozen graph definition.
        """
        from tensorflow.python.framework.graph_util import convert_variables_to_constants
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            # Graph -> GraphDef ProtoBuf
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                          output_names, freeze_var_names)
            return frozen_graph


if __name__ == "__main__":
    model = VGG16Training((100, 100, 3), "../../../datasets/pcms/features/", batch_size=128)
    model.train(epochs=1, save_path="model.h5")
    model.show_history("history_acc.jpg", "history_loss.jpg")
    print(model.train_generator.class_indices)
