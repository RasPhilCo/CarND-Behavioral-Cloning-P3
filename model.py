import csv
import cv2
import numpy as np

## Load training data
lines = []
# load lines from csv
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None) # skip the csv header
    for line in reader:
        lines.append(line)

X_images = []
y_headings = []
side_camera_correction = 0.2

def loadData(imagePath, heading):
    # Append heading to y list
    y_headings.append(heading)

    # Load & read image
    img = './data/' + imagePath.strip()
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Append image to X list
    X_images.append(img)

    # Create more diverse training data by flipping image
    X_images.append(cv2.flip(img, 1))
    y_headings.append(heading * -1.0)

# load center, left, & right image data with heading (steering angle)
for line in lines:
    heading = float(line[3])
    center, left, right = (line[0], line[1], line[2])
    loadData(center, heading)
    loadData(left, heading + side_camera_correction)
    loadData(right, heading - side_camera_correction)

X_train, y_train = (np.array(images), np.array(headings))

# Define model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

model = Sequential()
# Normalize image
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# Crop out top 50 and bottom 20 pixels (the sky & car hood)
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
# Use nVidea CNN
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Train model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

# Save model
model.save('model.h5')
exit()
