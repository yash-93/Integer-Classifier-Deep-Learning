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
    success, imgOriginal = cap.read()
    cv2.imwrite('test_frame' + str(count) + '.jpg', imgOriginal)
    count += 1

    # with open('test_frame.jpg', 'r+b') as f:
    #     with Image.open(f) as image:
    #             cover = resizeimage.resize_cover(image, [28, 28])
    #             cover.save('resized.jpg', image.format)
    #
    # img = Image.open('resized.jpg').convert("L")
    # im2arr = np.array(img)
    # im2arr_new = 255 - im2arr
    # plt.imsave('check.jpg', im2arr_new, cmap='gray')
    #
    # img = Image.open('check.jpg').convert("L")
    # im2arr = np.array(img)
    # im2arr = im2arr.reshape(1, 28, 28, 1).astype('float32')
    # im2arr /= 255
    #
    # pred = model.predict_classes(im2arr)
    # print(pred)

    cv2.imshow('frame', imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

