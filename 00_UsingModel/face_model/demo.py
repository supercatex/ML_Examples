import cv2

'''
Reference:
    https://talhassner.github.io/home/publication/2015_CVPR
    https://github.com/GilLevi/AgeGenderDeepLearning
    https://gist.github.com/GilLevi/54aee1b8b0397721aa4b
'''

face_net = cv2.dnn.readNet("./face/opencv_face_detector_uint8.pb", "./face/opencv_face_detector.pbtxt")
face_net_mean = (104, 117, 123)

gender_net = cv2.dnn.readNet("./gender/gender_net.caffemodel", "./gender/gender_deploy.prototxt")
gender_net_mean = (78.4263377603, 87.7689143744, 114.895847746)
gender_net_label = ('Male', 'Female')

age_net = cv2.dnn.readNet("./age/age_net.caffemodel", "./age/age_deploy.prototxt")
age_net_mean = (78.4263377603, 87.7689143744, 114.895847746)
age_net_label = ("(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)")

emotion_net = cv2.dnn.readNet("./emotion/EmotiW_VGG_S.caffemodel", "./emotion/deploy.prototxt")
emotion_net_label = ('Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise')

import os
import caffe
mean_filename=os.path.join("./emotion/",'mean.binaryproto')
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]


camera = cv2.VideoCapture(0)
while camera.isOpened():
    success, frame = camera.read()
    if not success:
        break

    h, w, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), face_net_mean, True, False)
    face_net.setInput(blob)
    face_net_output = face_net.forward()
    for i in range(face_net_output.shape[2]):
        confidence = face_net_output[0, 0, i, 2]
        if confidence > 0.5:
            x1 = max(int(face_net_output[0, 0, i, 3] * w), 0)
            y1 = max(int(face_net_output[0, 0, i, 4] * h), 0)
            x2 = min(int(face_net_output[0, 0, i, 5] * w), w)
            y2 = min(int(face_net_output[0, 0, i, 6] * h), h)

            face = frame[y1:y2, x1:x2, :]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), gender_net_mean, False, False)
            gender_net.setInput(blob)
            gender_net_output = gender_net.forward()
            gender_index = gender_net_output[0].argmax()

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), age_net_mean, False, False)
            age_net.setInput(blob)
            age_net_output = age_net.forward()
            age_index = age_net_output[0].argmax()

            blob = cv2.dnn.blobFromImage(face, 1.0, (256, 256), (0, 0, 0), False, False)
            emotion_net.setInput(blob)
            emotion_net_output = emotion_net.forward()
            emotion_index = emotion_net_output[0].argmax()
            print(emotion_net_label[emotion_index])

            label = gender_net_label[gender_index] + ": " + age_net_label[age_index]
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.imshow("frame", frame)

    key_code = cv2.waitKey(1)
    if key_code in [27, ord('q')]:
        break
camera.release()
cv2.destroyAllWindows()
