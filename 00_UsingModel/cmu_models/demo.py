import cv2
import numpy as np


coco_pose_net = cv2.dnn.readNet("./pose/coco/pose_iter_440000.caffemodel", "./pose/coco/pose_deploy_linevec.prototxt")
coco_pose_points = ('Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear')
coco_pose_colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

camera = cv2.VideoCapture(0)
while camera.isOpened():
    success, frame = camera.read()
    if not success:
        break

    frame = cv2.imread("sample1.jpg", cv2.IMREAD_COLOR)

    h, w, _ = frame.shape
    ih = 368
    iw = int((ih / h) * w)

    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (iw, ih), (0, 0, 0), False, False)
    coco_pose_net.setInput(blob)
    output = coco_pose_net.forward()

    for i, p in enumerate(coco_pose_points):
        prob_map = output[0, i, :, :]
        prob_map = cv2.resize(prob_map, (w, h))
        mask_map = cv2.inRange(prob_map, 0.5, 1.0)

        if i == 0:
            contours, _ = cv2.findContours(mask_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                mask_cnt = np.zeros(mask_map.shape)
                mask_cnt = cv2.fillConvexPoly(mask_cnt, cnt, 1)
                prob_cnt = prob_map * mask_cnt
                _, _, _, max_loc = cv2.minMaxLoc(prob_cnt)
                cv2.circle(frame, max_loc, 15, coco_pose_colors[i], -1)

    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    cv2.imshow("frame", frame)

    key_code = cv2.waitKey(0)
    if key_code in [27, ord('q')]:
        break
camera.release()
cv2.destroyAllWindows()
