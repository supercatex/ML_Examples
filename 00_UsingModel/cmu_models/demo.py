from OpenPose import OpenPose
import cv2
import numpy as np


pose = OpenPose()

camera = cv2.VideoCapture(0)
while camera.isOpened():
    success, frame = camera.read()
    if not success:
        continue

    points = pose.detect(frame, in_height=168)
    for p in points:
        pose.draw(frame, p)

    cv2.imshow("frame", frame)

    key_code = cv2.waitKey(1)
    if key_code in [27, ord('q')]:
        break
camera.release()
cv2.destroyAllWindows()
