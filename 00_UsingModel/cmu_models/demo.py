from OpenPose import OpenPose
from PoseNorimalization import *
import cv2
import keras
import numpy as np


pose = OpenPose()
model = keras.models.load_model("model.h5")
# model2 = cv2.dnn.readNet("../cnn_vgg/my_model.pb")


camera = cv2.VideoCapture(0)
while camera.isOpened():
    success, frame = camera.read()
    if not success:
        continue

    # points = pose.detect(frame)
    output = f(frame, pose)
    for img in output:
        img = cv2.resize(img, (100, 100))
        predict = model.predict(np.array([img]))
        print(img.shape)
        # blob = cv2.dnn.blobFromImage(np.float32(img), 1.0 / 255, (100, 100), (0, 0, 0), False, False)
        # model2.setInput(blob)
        # predict = model2.forward()

        idx = np.argmax(predict)
        labels = ["stand", "sit", "squat", "fall"]
        print(idx, labels[idx])
    # for p in points:
    #     pose.draw(frame, p)
    #
    #     labels = ['fall', 'sit', 'squat', 'stand']


    cv2.imshow("frame", frame)

    key_code = cv2.waitKey(1)
    if key_code in [27, ord('q')]:
        break
camera.release()
cv2.destroyAllWindows()
