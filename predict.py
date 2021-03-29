import cv2
import base64
from urllib.parse import quote
from mtcnn import MTCNN
import dlib
import pickle
from imutils import face_utils


detector = MTCNN()
landmark_detector = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")


def predict_faceshape(image_path):
    image = cv2.imread(image_path)
    # xac dinh khuon mat
    results = detector.detect_faces(image)
    if not results:
        return {"error": "Không tìm thấy khuôn mặt trong ảnh"}
    # trich xuat khuon mat dau tien
    result = results[0]
    x1, y1, width, height = result['box']
    x1, y1 = abs(x1), abs(y1)
    x2 = x1 + width
    y2 = y1 + height
    face = image[y1:y2, x1:x2]
    # convert image to text
    _, buffer = cv2.imencode(".jpg", face)
    image_to_text = base64.b64encode(buffer)
    # extract dlib
    landmark = landmark_detector(image, dlib.rectangle(x1, y1, x2, y2))
    landmark = face_utils.shape_to_np(landmark)
    landmark = landmark.reshape(68 * 2)
    # load model
    model = None
    with open("model/model.sav", "rb") as model_file:
        model = pickle.load(model_file)
        model_file.close()
    if not model:
        return {"error": "Không load được model"}
    # predict faceshape
    face_shape = model.predict([landmark])
    return {"image": "data:image/png;base64,{}".format(quote(image_to_text)), "label": face_shape[0]}
