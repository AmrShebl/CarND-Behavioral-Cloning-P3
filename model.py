import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers import Lambda,Convolution2D,MaxPooling2D, Cropping2D
import matplotlib.pyplot as plt

def get_collection_data(images,steering_angles,file_name):
    lines = []
    with open(file_name) as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)
    angle_correction = 0.2
    angle = 0
    for line in lines:
        center_image, left_image, right_image = cv2.imread(line[0]), cv2.imread(line[1]), cv2.imread(line[2])
        images.extend([center_image, left_image, right_image])
        angle = float(line[3])
        steering_angles.extend([angle, angle + angle_correction, angle - angle_correction])
        center_image, left_image, right_image = cv2.flip(center_image, 1), cv2.flip(left_image, 1), cv2.flip(
            right_image, 1)
        images.extend([center_image, left_image, right_image])
        steering_angles.extend([-1.0 * angle, -1.0 * (angle + angle_correction), -1.0 * (angle - angle_correction)])

def train_model():
    images = []
    steering_angles = []
    n_collections=4
    for i in range(1,n_collections+1):
        file_name=".\Data\Collection"+str(i)+"\driving_log.csv"
        get_collection_data(images,steering_angles,file_name)
    x_train=np.array(images)
    y_train=np.array(steering_angles)
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(filters=24, kernel_size= 5, strides=2,padding='valid', activation='relu'))
    model.add(Convolution2D(filters=36, kernel_size= 5, strides=2,padding='valid', activation='relu'))
    model.add(Convolution2D(filters=48, kernel_size= 5, strides=2,padding='valid', activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile('adam','mse')
    model_history=model.fit(x_train,y_train,epochs=5,shuffle=True,validation_split=0.2, batch_size=500, verbose=2)
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Model mean square error loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean squared error loss')
    plt.legend(['training set', 'validation_set'], loc='upper right')
    plt.show()
    model.save('trained_model.h5')