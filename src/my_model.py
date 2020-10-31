from __future__ import print_function
import numpy as np
import os
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model

import PIL
from PIL import Image
import cv2
from .. import config

# constant and path varaibles
batch_size = 32
num_classes = 8
epochs = 100

# input image dimensions

print(os.listdir(config.dataset_path()))

diseases_label = os.listdir(config.dataset_path())

IMG_SIZE = 100
IMG_CHANNEL = 3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_CHANNEL)
lr = 0.001

def process_train_data():
    for folder_no, folder in enumerate(diseases_label):
        print(folder)

        work_folder = os.path.join(config.dataset_path(), folder)
        dis_images = [os.path.join(work_folder, path) for path in os.listdir(work_folder)]

        print("preparing %s folder. total images: %s" % (folder, str(len(dis_images))))
        #a = [print(i) for i in train_images]

        dis_data = np.ndarray((len(dis_images), IMG_SIZE, IMG_SIZE, 
            IMG_CHANNEL), dtype = np.uint8)

        label = []
        for img_no, image_file in enumerate(dis_images):
            """
            img = Image.open(image_file)
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_px = np.array(img)  #convert PIL image to numpy array
            """
            img = cv2.imread(image_file)
            img_px = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            dis_data[img_no] = img_px
            label.append(folder_no)
        
        #print(label)
       
       # 85% train data & 15% test data
        split = int(len(dis_data) * 0.85)
        if (folder_no == 0):
            X_train = dis_data[:split]; X_test = dis_data[split:]
            y_train = label[:split]; y_test = label[split:]

        else:
            X_train = np.vstack((X_train, dis_data[:split]))
            X_test = np.vstack((X_test, dis_data[split:]))

            y_train = y_train + label[:split]
            y_test = y_test + label[split:] 
        
    del dis_data, label
    return X_train, y_train, X_test, y_test


# the data, split between train and test sets
X_train, y_train, X_test, y_test = process_train_data()

np.save(os.path.join(config.output_path(), "X_train"), X_train)
np.save(os.path.join(config.output_path(), "X_test"), X_test)
np.save(os.path.join(config.output_path(), "y_train"), y_train)
np.save(os.path.join(config.output_path(), "y_test"), y_test)
"""

# loading from saved npz file
X_train  = np.load(os.path.join(config.output_path(), "X_train.npy")) 
X_test  = np.load(os.path.join(config.output_path(), "X_test.npy"))
y_train = np.load(os.path.join(config.output_path(), "y_train.npy"))
y_test = np.load(os.path.join(config.output_path(), "y_test.npy"))
"""

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("\ntrain_data shape:", X_train.shape)
print("train_label_shape:", y_train.shape)
print("\ntest_data shape:", X_test.shape)
print("test_label_shape:", y_test.shape)


model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3),
                 activation = 'relu',
                 input_shape = IMG_SHAPE))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(50, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
"""
model = load_model(os.path.join(config.output_path(), "my_model.h5"))
"""
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size = batch_size,
          epochs = epochs,
          verbose = 2,
	      shuffle = True,
          validation_data= (X_test, y_test))


score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save(os.path.join(config.output_path(), "model_1.h5"))


print("Start Predicting.....")
print("actulal label: ", y_test[300])
index = np.argmax(y_test[300])
print("Plant label: ", diseases_label[index])

test_image = np.ndarray((1, IMG_SIZE, IMG_SIZE, IMG_CHANNEL), dtype = np.float32)
test_image[0] = X_test[300]
predictions = model.predict(test_image, 1, verbose = 2)

#formatting result; 
result = predictions[0]
predict_index = np.argmax(result)
print("Prediction: ", diseases_label[predict_index])

