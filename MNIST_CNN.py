from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
from sklearn.metrics import accuracy_score # calculate accuracy

(X_train, y_train), (X_test, y_test)=mnist.load_data()

# building the input vector from the 28x28 pixels
X_train=X_train.reshape(X_train.shape[0], 28, 28, 1) # 1: unique channel (gray scale)
# 3 -> red, blue, green
X_test=X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train=X_train.astype('float32'); X_test=X_test.astype('float32')
# normalizing the data:
X_train/=255; X_test/=255
# one-hot encoding:
n_classes=10
Y_train=np_utils.to_categorical(y_train, n_classes); Y_test=np_utils.to_categorical(y_test, n_classes)

# building a linear stack of layers with the sequential model
model=Sequential()
model.add(Conv2D(25,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',input_shape=(28,28,1))) # convolutional layer
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of convolution:
model.add(Flatten())
model.add(Dense(100, activation='relu')) # hidden layer
model.add(Dense(10, activation='softmax')) # output layer
# compiling the sequential model
model.summary(); model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# training the model for 15 epochs
model.fit(X_train, Y_train, batch_size=128, epochs=15, validation_data=(X_test, Y_test))

# evaluating the model:
#test_loss, test_accuracy=model.evaluate(X_test,Y_test)
