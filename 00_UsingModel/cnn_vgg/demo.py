from keras.applications import VGG16
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


image_shape = (100, 100, 3)

vgg = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=image_shape
)
for layer in vgg.layers[:-4]:
    layer.trainable = False
for layer in vgg.layers:
    print(layer, layer.trainable)

model = models.Sequential()
model.add(vgg)
model.add(layers.Flatten())
model.add(layers.Dense(units=4, activation="softmax"))
model.summary()

train_data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
valid_data_gen = ImageDataGenerator(rescale=1.0 / 255)

train_batch_size = 100
valid_batch_size = 10

train_generator = train_data_gen.flow_from_directory(
    "../../../datasets/pcms/features/",
    target_size=image_shape[:2],
    batch_size=train_batch_size,
    class_mode='categorical',
    shuffle=True
)

# validation_generator = valid_data_gen.flow_from_directory(
#     "../../../datasets/pcms/photos/",
#     target_size=image_shape[:2],
#     batch_size=valid_batch_size,
#     class_mode='categorical',
#     shuffle=False
# )

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=30,
    # validation_data=validation_generator,
    # validation_steps=validation_generator.samples / validation_generator.batch_size,
    verbose=1)

model.save('model.h5')
print(history.history)

acc = history.history['acc']
# val_acc = history.history['val_acc']
loss = history.history['loss']
# val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
