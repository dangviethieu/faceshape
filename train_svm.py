import pickle
from sklearn import svm


# load du lieu tu file pkl
landmark_list = []
label_list = []

with open("model/landmarks.pkl", "rb") as landmarks_file:
    landmark_list = pickle.load(landmarks_file)
    landmarks_file.close()
with open("model/labels.pkl", "rb") as labels_file:
    label_list = pickle.load(labels_file)
    labels_file.close()

# train model
svc = svm.SVC(kernel="linear")
svc.fit(landmark_list, label_list)

# test
result = svc.predict([landmark_list[0]])
print("ket qua predict: ", result, ", gia tri thuc te: ", label_list[0])

# save model
model_file_name = "model/model.sav"
with open(model_file_name, "wb") as model_file:
    pickle.dump(svc, model_file)
    model_file.close()
