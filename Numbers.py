from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import math; import numpy as np
import matplotlib.pyplot as plt
import logging; from urllib import parse
from http.server import HTTPServer, BaseHTTPRequestHandler

logger = tf.get_logger()
logger.setLevel(logging.ERROR)
# Carga de datos de MNIST:
dataset, metadata=tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset=dataset['train'], dataset['test']

# Etiquetas:
class_names=['Cero', 'Uno', 'Dos', 'Tres', 'Cuatro', 'Cinco', 'Seis','Siete', 'Ocho', 'Nueve']
num_train_examples=metadata.splits['train'].num_examples
num_test_examples=metadata.splits['test'].num_examples

# Normalizamos
def normalize(images, labels):
    images=tf.cast(images, tf.float32)
    images/=255
    return images, labels

train_dataset=train_dataset.map(normalize); test_dataset=test_dataset.map(normalize)

# Estructura de la red:
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # softmax para clasificacion
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Aprendizaje por lotes de 32:
BATCHSIZE=32; train_dataset=train_dataset.repeat().shuffle(num_train_examples).batch(BATCHSIZE)
test_dataset=test_dataset.batch(BATCHSIZE)
# Entrenamiento:
model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCHSIZE))

# Clase para definir el servidor http:
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        print("Peticion recibida")
        #Obtener datos de la peticion y limpiar los datos
        content_length = int(self.headers['Content-Length'])
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
"""
# Evaluating:
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/BATCHSIZE))
print('model.evaluate: ', model.evaluate(test_dataset, steps=math.ceil(num_test_examples/BATCHSIZE)), type(model.evaluate(test_dataset, steps=math.ceil(num_test_examples/BATCHSIZE))))

for test_images, test_labels in test_dataset.take(2):
	test_images = test_images.numpy()
	test_labels = test_labels.numpy()
	predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_labels, images):
	predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(img[...,0], cmap=plt.cm.binary)

	predicted_label = np.argmax(predictions_array)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel("Prediccion: {}".format(class_names[predicted_label]), color=color)
	print('class_names[predicted_label]: ', class_names[predicted_label], predicted_label)

def plot_value_array(i, predictions_array, true_label):
	predictions_array, true_label=predictions_array[i], true_label[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	thisplot=plt.bar(range(10), predictions_array, color="#888888")
	plt.ylim([0,1])
	predicted_label = np.argmax(predictions_array)

	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

numrows=5; numcols=3
numimages=numrows*numcols
plt.figure(figsize=(2*2*numcols, 2*numrows))
for i in range(numimages):
	plt.subplot(numrows, 2*numcols, 2*i+1)
	plot_image(i, predictions, test_labels, test_images)
	plt.subplot(numrows, 2*numcols, 2*i+2)
	plot_value_array(i, predictions, test_labels)

plt.show()
"""