from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils; import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import logging; from urllib import parse
from http.server import HTTPServer, BaseHTTPRequestHandler

logger = tf.get_logger()
logger.setLevel(logging.ERROR)
# Carga de datos de MNIST:
(X_train, y_train), (X_val, y_val)=mnist.load_data()

# Etiquetas:
class_names=['Cero', 'Uno', 'Dos', 'Tres', 'Cuatro', 'Cinco', 'Seis','Siete', 'Ocho', 'Nueve']

# Flattening the images from the 28x28 pixels to 1D 784 pixels:
#X_train=X_train.reshape(60000, 784); X_val=X_val.reshape(10000, 784)
X_train=X_train.astype('float32'); X_val=X_val.astype('float32')
# Normalizamos:
X_train/=255; X_val/=255
# One-hot encoding:
n_classes=10
Y_train=np_utils.to_categorical(y_train, n_classes)
Y_val=np_utils.to_categorical(y_val, n_classes)

"""
# Estructura ANN de la red:
model=Sequential()
model.add(Flatten(input_shape=(28,28,1)))
model.add(Dense(100, activation='relu')) # hidden layer #input_shape=(784,)
model.add(Dense(n_classes, activation='softmax')) # output layer
# Sumario:
model.summary()
# Compilamos:
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# Entrenamos:
model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_val, Y_val))
"""

#Estructura CNN:
#
from tensorflow.keras.preprocessing.image import ImageDataGenerator

rango_rotacion = 30
mov_ancho = 0.25
mov_alto = 0.25
#rango_inclinacion=15 #No uso este de momento pero si quieres puedes probar usandolo!
rango_acercamiento=[0.5,1.5]

datagen = ImageDataGenerator(
    rotation_range = rango_rotacion,
    width_shift_range = mov_ancho,
    height_shift_range = mov_alto,
    zoom_range=rango_acercamiento,
    #shear_range=rango_inclinacion #No uso este de momento pero si quieres puedes probar usandolo!
)

datagen.fit(X_train)
#
model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',input_shape=(28,28,1))) # convolutional layer
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu')) # convolutional layer
model.add(MaxPool2D(pool_size=(2,2))); model.add(Dropout(0.5))
# Flattening:
model.add(Flatten())
model.add(Dense(100, activation='relu')) # hidden layer
model.add(Dense(10, activation='softmax')) # output layer
# compiling the sequential model
model.summary(); model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# Entrenamiento: (Data Augmentation)
data_gen_training=datagen.flow(X_train, Y_train, batch_size=128)
model.fit(data_gen_training, batch_size=128, epochs=5, validation_data=(X_val, Y_val))
#X_train, Y_train

# Clase para definir el servidor http:
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        print("Peticion recibida")
        #Obtener datos de la peticion y limpiar los datos
        content_length=int(self.headers['Content-Length'])
        data=self.rfile.read(content_length)
        data=data.decode().replace('pixeles=', '')
        data=parse.unquote(data)
        #Realizar transformacion para dejar igual que los ejemplos que usa MNIST
        arr=np.fromstring(data, np.float32, sep=",")
        arr=arr.reshape(28,28); arr=np.array(arr); arr=arr.reshape(1,28,28,1)
        #Realizar y obtener la prediccion
        prediction_values=model.predict(arr, batch_size=1)
        prediction=str(np.argmax(prediction_values))
        print("Prediccion final: "+ prediction)
        #Regresar respuesta a la peticion HTTP
        self.send_response(200)
        #Evitar problemas con CORS
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers(); self.wfile.write(prediction.encode())

# Iniciar el servidor en el puerto 8000:
print("Iniciando el servidor...")
server = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler) #localhost8000
server.serve_forever()

############################################################################################
