import os
import cv2
import numpy as np
from keras.models import load_model


f = open("VGG16_labels.txt", "r")
labels = [x.strip() for x in f.readlines()]
f.close()
print(labels)

model = load_model("VGG16_model.h5")

test_dir = "test"
for f in os.listdir(test_dir):
    path_img = os.path.join(test_dir, f)
    image = cv2.imread(path_img, cv2.IMREAD_COLOR)
    img = cv2.resize(image, (100, 100))
    predict = model.predict(np.array([img]))
    idx = np.argmax(predict)
    label = labels[idx]
    print(idx, label)
    cv2.imshow("image", image)
    key_code = cv2.waitKey(0)
    if key_code in [27, ord('q')]:
        break
