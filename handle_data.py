import os
import cv2
import dlib
import pickle
import numpy as np
from mtcnn import MTCNN
from imutils import face_utils


raw_folder = "face_data"

detector = MTCNN()
landmark_detector = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

landmark_list = []
label_list = []

for folder in os.listdir(raw_folder):
    if folder[0] != ".":
        print("--------------------------------")
        print("processing folder: " + folder)
        for file in os.listdir(os.path.join(raw_folder, folder)):
            print("processing file: " + file)
            file_path = os.path.join(raw_folder, folder, file)
            # doc anh
            image = cv2.imread(file_path)
            # phat hien khuon mat
            results = detector.detect_faces(image)
            if results:
                # lay khuon mat dau tien
                result = results[0]
                x1, y1, width, height = result['box']
                x1, y1 = abs(x1), abs(y1)
                x2 = x1 + width
                y2 = y1 + height
                face = image[y1:y2, x1:x2]
                landmark = landmark_detector(image, dlib.rectangle(x1, y1, x2, y2))
                landmark = face_utils.shape_to_np(landmark)
                landmark = landmark.reshape(68 * 2)
                # them cac landmark vao list
                landmark_list.append(landmark)
                label_list.append(folder)

print(len(landmark_list))
landmark_list = np.array(landmark_list)
label_list = np.array(label_list)
# write to landmarks.pkl
with open("landmarks.pkl", "wb") as landmarks_file:
    pickle.dump(landmark_list, landmarks_file)
    landmarks_file.close()
# write to labels.pkl
with open("labels.pkl", "wb") as labels_file:
    pickle.dump(label_list, labels_file)
    labels_file.close()

