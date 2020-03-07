import numpy as np
import cv2
import sys, os
import pickle
from tensorflow.keras.models import model_from_json, load_model

width = 640
height = 480

model = load_model("DigitClassifierModel.h5")
cap = cv2.VideoCapture(0)

# pickle_in = open('savedmodel.p', 'rb')
# model = pickle.load(pickle_in)



# json_file = open("model-bw.json", "r")
# model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(model_json)
# # load weights into new model
# loaded_model.load_weights("model-bw.h5")
# print("Loaded model from disk")

while True:
    success, imgOriginal = cap.read()
    img = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    pred = model.predict(np.array([img]))
    print("********************** => ", pred)

    # img = np.array(imgOriginal)
    # img = cv2.resize(img, (28, 28))
    # print(img.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = img/255
    # cv2.imshow("", img)
    # img = img.reshape(1, 28, 28)
    # print(img.shape)
    # classIndex = int(loaded_model.predict(img))
    # print(classIndex)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()