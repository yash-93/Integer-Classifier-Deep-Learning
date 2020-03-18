from tensorflow.keras.models import load_model
from resizeimage import resizeimage
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = load_model('mnistCNN.h5')
cap = cv2.VideoCapture(0)
count = 0
while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower = np.array([93, 123, 176])
    # upper = np.array([128, 255, 255])
    lower = np.array([71, 112, 112])
    upper = np.array([128, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    upper_left = (38, 200)
    bottom_right = (100, 262)
    cv2.rectangle(mask, upper_left, bottom_right, (255, 0, 0), 3)
    cv2.rectangle(frame, upper_left, bottom_right, (255, 0, 0), 3)
    img = np.array(mask)
    rect_img = img[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
    rect_img = cv2.resize(rect_img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    im2arr = rect_img.reshape(1, 28, 28, 1).astype('float32')
    im2arr /= 255
    pred = model.predict_classes(im2arr)
    cv2.putText(mask, str(pred[0]), (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("rect_img", rect_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

