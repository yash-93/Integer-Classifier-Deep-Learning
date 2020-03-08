from tensorflow.keras.models import load_model
from resizeimage import resizeimage
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import cv2
import matplotlib.pyplot as plt


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1).astype('float32')
X_test = X_test.reshape(10000, 28, 28, 1).astype('float32')

X_train /= 255
X_test /= 255

y_train_one_hot = to_categorical(y_train, 10)
y_test_one_hot = to_categorical(y_test, 10)

model = load_model('mnistCNN.h5')

with open('images.jpg', 'r+b') as f:
    with Image.open(f) as image:
            cover = resizeimage.resize_cover(image, [28, 28])
            cover.save('resized.jpg', image.format)

img = Image.open('resized.jpg')
im2arr = np.array(img)
im2arr_new = 255 - im2arr
plt.imsave('check.jpg', im2arr_new, cmap='gray')

img = Image.open('check.jpg').convert("L")
im2arr = np.array(img)
im2arr = im2arr.reshape(1, 28, 28, 1).astype('float32')
im2arr /= 255

pred = model.predict_classes(im2arr)
print(pred)

# predictions = model.predict_classes(X_test[10].reshape(1,28,28,1))
# print(predictions)
#
# print(y_test[10])