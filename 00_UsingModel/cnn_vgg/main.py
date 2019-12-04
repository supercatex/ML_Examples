from VGG16Training import VGG16Training


model = VGG16Training((100, 100, 3), "../../../datasets/pcms/features/", "../../../datasets/pcms/features/")
model.train(epochs=100, save_path="VGG16_model.h5")
model.show_history("VGG16_history_acc.jpg", "VGG16_history_loss.jpg")
print(model.train_generator.class_indices)
