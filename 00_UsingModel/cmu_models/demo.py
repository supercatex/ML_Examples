from OpenPose import OpenPose
import cv2
import keras
import numpy as np


pose = OpenPose()
model = keras.models.load_model("model.h5")

camera = cv2.VideoCapture(0)
while camera.isOpened():
    success, frame = camera.read()
    if not success:
        continue

    # frame = cv2.imread("sample1.jpg", cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, (720, 480))

    points = pose.detect(frame)
    for p in points:
        pose.draw(frame, p)

        data = []
        for i in range(len(p)):
            if i in [1, 8, 9, 10, 11, 12, 13]:
                data.append(p[i][0])
                data.append(p[i][1])
        data = np.array([data])
        predict = model.predict(data)
        idx = np.argmax(predict)
        labels = ['fall', 'sit', 'squat', 'stand']
        print(idx, labels[idx])
        # cv2.imshow("frame", frame)
        # cv2.waitKey(0)
    print()

    cv2.imshow("frame", frame)

    key_code = cv2.waitKey(500)
    if key_code in [27, ord('q')]:
        break
camera.release()
cv2.destroyAllWindows()
