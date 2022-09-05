from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D
from keras.utils import np_utils; import numpy as np
from sklearn.metrics import accuracy_score

(X_train, y_train), (X_test, y_test)=mnist.load_data()

"""
The basic steps to build an image classification model using a neural network are:
1. Flatten the input image dimensions to 1D (width pixels x height pixels)
2. Normalize the image pixel values (divide by 255)
3. One-Hot Encode the categorical column
4. Build a model architecture (Sequential) with Dense layers
5. Train the model and make predictions
"""

# Flattening the images from the 28x28 pixels to 1D 784 pixels
X_train=X_train.reshape(60000, 784); X_test=X_test.reshape(10000, 784)
X_train=X_train.astype('float32'); X_test=X_test.astype('float32')
# normalizing the data:
X_train/=255; X_test/=255
# one-hot encoding:
n_classes = 10
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

# building a linear stack of layers with the sequential model
model=Sequential()
model.add(Dense(100, input_shape=(784,), activation='relu')) # hidden layer
model.add(Dense(n_classes, activation='softmax')) # output layer
# model summary
model.summary()
# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# training the model
model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
