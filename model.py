import os
import csv
import cv2
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt

### Image and Measurement lists
road_images = []
steering_angles = []

### Loop through multiple directories of training data
data_dirs = os.listdir('./data')
for dir in data_dirs:
    root_dir = './data/' + dir + '/'
    csv_file =  root_dir + 'driving_log.csv'
    ### Open csv file
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        iter_rows = iter(reader)
        # Skip first row of headers
        next(iter_rows)
        for row in iter_rows:
            steering_center = float(row[3])
            steering_center_flip = steering_center * -1
            
            # Create adjusted steering measurements for the side camera images
            correction = 0.2 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction
            
            # Read in images from center, center flipped, left and right cameras
            img_center = np.asarray(Image.open(root_dir + row[0].strip(' ')))
            img_center_flip = cv2.flip(img_center,1)
            img_left = np.asarray(Image.open(root_dir + row[1].strip(' ')))
            img_right = np.asarray(Image.open(root_dir + row[2].strip(' ')))
            
            # Add images and angles to data set
            road_images.extend([img_center, img_center_flip, img_left, img_right])
            steering_angles.extend([steering_center, steering_center_flip, steering_left, steering_right])

X_train = np.array(road_images)
y_train = np.array(steering_angles)

### NVIDIA CNN Architecture: https://devblogs.nvidia.com/deep-learning-self-driving-cars/
### Added dropout after fully connected layers to prevent overfitting

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(1))

### Mean Squared Error
### Adam Optimizer
### 80%/20% training/validation split
### Epochs: 2
model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2, verbose=1)
model.save('model.h5')

### Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
