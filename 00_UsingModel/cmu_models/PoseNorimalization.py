import os
import cv2
import time
import numpy as np
from OpenPose import OpenPose


def f(image, pose):
    output = []
    h, w, _ = image.shape
    detected = pose.detect(image)
    for points in detected:
        for i, point in enumerate(points):
            if i not in [1, 8, 9, 10, 11, 12, 13]:
                points[i][0] = -1
                points[i][1] = -1
        black_image = np.zeros(image.shape)
        pose.draw(black_image, points, 10)

        x1 = np.min(points[:, 0][np.where(points[:, 0] > 0)])
        y1 = np.min(points[:, 1][np.where(points[:, 1] > 0)])
        x2 = np.max(points[:, 0][np.where(points[:, 0] > 0)])
        y2 = np.max(points[:, 1][np.where(points[:, 1] > 0)])

        w = x2 - x1 + 10
        h = y2 - y1 + 10
        s = max(w, h)
        o = s // 20
        ox = (s - w) // 2 + o
        oy = (s - h) // 2 + o
        s = s + o + o

        square_image = np.zeros((s, s, 3))
        square_image[oy:oy+h, ox:ox+w, :] = black_image[y1-5:y2+5, x1-5:x2+5, :]

        output.append(square_image)
    return output


if __name__ == "__main__":
    _src = "../../../datasets/pcms/1126/"
    if not os.path.exists(_src):
        raise Exception("source directory not found.")
    _dst = "../../../datasets/pcms/features/"
    if not os.path.exists(_dst):
        os.mkdir(_dst)

    _pose = OpenPose()

    for _f1 in os.listdir(_src):
        _src_f1 = os.path.join(_src, _f1)
        _dst_f1 = os.path.join(_dst, _f1)
        if not os.path.exists(_dst_f1):
            os.mkdir(_dst_f1)
        for _f2 in os.listdir(_src_f1):
            _image = cv2.imread(os.path.join(_src_f1, _f2), cv2.IMREAD_COLOR)
            _output = f(_image, _pose)
            for _img in _output:
                cv2.imwrite(os.path.join(_dst_f1, str(int(time.time() * 1000000)) + ".jpg"), _img)
                cv2.imshow("image", _image)
                cv2.imshow("img", _img)
                cv2.waitKey(500)

    # _pose = OpenPose()
    # _image = cv2.imread("sample1.jpg", cv2.IMREAD_COLOR)
    # _output = f(_image, _pose)
    # cv2.imshow("image", _image)
    # for i in range(len(_output)):
    #     cv2.imshow("%d" % i, _output[i])
    # cv2.waitKey(0)
