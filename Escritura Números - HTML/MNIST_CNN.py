from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
from sklearn.metrics import accuracy_score # calculate accuracy

(X_train, y_train), (X_val, y_val)=mnist.load_data()

# building the input vector from the 28x28 pixels
X_train=X_train.reshape(X_train.shape[0], 28, 28, 1) # 1: unique channel (gray scale)
# 3 -> red, blue, green
X_val=X_val.reshape(X_val.shape[0], 28, 28, 1)
X_train=X_train.astype('float32'); X_val=X_val.astype('float32')
# normalizing the data:
X_train/=255; X_val/=255
# one-hot encoding:
n_classes=10
Y_train=np_utils.to_categorical(y_train, n_classes); Y_val=np_utils.to_categorical(y_val, n_classes)

#Estructura CNN:
#
from tensorflow.keras.preprocessing.image import ImageDataGenerator
rango_rotacion=30; mov_ancho=0.25; mov_alto=0.25; rango_acercamiento=[0.5,1.5]
#rango_inclinacion=15; 

datagen = ImageDataGenerator(
    rotation_range = rango_rotacion,
    width_shift_range = mov_ancho,
    height_shift_range = mov_alto,
    zoom_range=rango_acercamiento,
    #shear_range=rango_inclinacion
)

datagen.fit(X_train)
#
epochs=50; model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',input_shape=(28,28,1))) # convolutional layer
model.add(MaxPool2D(pool_size=(2,2))); model.add(Dropout(0.25))
model.add(Conv2D(64,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu')) # convolutional layer
model.add(MaxPool2D(pool_size=(2,2))); model.add(Dropout(0.35))
# Flattening:
model.add(Flatten())
model.add(Dense(100, activation='relu')) # hidden layer
model.add(Dropout(0.35))
model.add(Dense(10, activation='softmax')) # output layer
# compiling the sequential model
model.summary(); model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# Entrenamiento: (Data Augmentation)
data_gen_training=datagen.flow(X_train, Y_train, batch_size=128)
model.fit(data_gen_training, batch_size=128, epochs=epochs, validation_data=(X_val, Y_val))
#X_train, Y_train
# evaluating the model:
#test_loss, test_accuracy=model.evaluate(X_test,Y_test)

model.save('model_MNIST_CNN.h5')
