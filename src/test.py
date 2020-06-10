import cv2
from .. import config
import sys, os
import numpy as np

from keras.models import load_model

# label
diseases_label = os.listdir(config.dataset_path())

# get image name
img_name = str(sys.argv[0])

# get image dir
img_file = os.path.join(config.test_path(), img_name)

IMG_SIZE = 100
IMG_CHANNEL = 3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_CHANNEL)

# loading image
img = cv2.imread(img_file)
img_px = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

# normalization
img = img / 255

# loading model
model = load_model(os.path.join(config.output_path(), "model_1.h5"))

print("Start Predicting.....")
# testy image predcition
test_image = np.ndarray((1, IMG_SIZE, IMG_SIZE, IMG_CHANNEL), dtype = np.float32)
test_image[0] = img_px
predictions = model.predict(test_image, 1, verbose = 2)


#formatting result; 
result = predictions[0]
predict_index = np.argmax(result)
print("Prediction: ", diseases_label[predict_index])
