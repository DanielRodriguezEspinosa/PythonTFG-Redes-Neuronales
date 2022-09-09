# (CNN) Clasification: astronomical images
import numpy as np; import math; import matplotlib.pyplot as plt
from astropy.io import ascii, fits; import h5py; import random
from keras.models import Sequential, Model; from plot_model import plot_model
from keras.utils.vis_utils import plot_model; import keras
from keras import optimizers; import tensorflow_addons as tfa
from keras.optimizers import SGD; import keras.layers
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Input, BatchNormalization
from keras.regularizers import L1, L2, L1L2
#from keras.layers import BatchNormalization, LeakyReLU
import tensorflow as tf; from keras.utils import to_categorical
from sklearn.metrics import accuracy_score; from neuralplot import ModelPlot
import pydotplus; #from keras.utils import plot_model
keras.utils.vis_utils.pydot = pydotplus
from ann_visualizer.visualize import ann_viz

class_names=['Circular', 'Cometa','Donut'] #, 'Circular (GSeeing)', 'Circular (BSeeing)'
# Load file:
def read_hdf5_array(fname,hdf5_dir='./'):
    # Open the HDF5 file
    file=h5py.File(hdf5_dir+fname+".h5", "r+")
    stamps=np.array(file["/stamps"])
    labels=np.array(file["/labels"]).astype(np.uint16)
    return stamps, labels

dataset_training, _metadata_training=read_hdf5_array('training_plus', hdf5_dir='./')
dataset_validation, _metadata_validation=read_hdf5_array('validation', hdf5_dir='./')
# -> numpy arrays:
_metadata_training_=[]; _metadata_validation_=[]
for i in range(_metadata_training.shape[0]):
    _metadata_training_.append(_metadata_training[i][0])
for j in range(_metadata_validation.shape[0]):
    _metadata_validation_.append(_metadata_validation[j][0])
metadata_training=np.array(_metadata_training_); metadata_validation=np.array(_metadata_validation_) #dtype=np.uint8

# building the input vector from the 60x60 pixels
dataset_training=dataset_training.reshape(dataset_training.shape[0], 60, 60, 1) # (..., 50, 50, 1)
dataset_validation=dataset_validation.reshape(dataset_validation.shape[0], 60, 60, 1) # (..., 50, 50, 1)
dataset_training=dataset_training.astype('float32'); dataset_validation=dataset_validation.astype('float32')
# One-hot encoding:
n_classes=len(class_names)
metadata_training=to_categorical(metadata_training, n_classes); metadata_validation=to_categorical(metadata_validation, n_classes)

###
BATCHSIZE=32 # -> tensors:
_metadata_training_=np.array(_metadata_training_); _metadata_validation_=np.array(_metadata_validation_)

_dataset_training=tf.data.Dataset.from_tensor_slices((dataset_training, _metadata_training_))
_dataset_validation=tf.data.Dataset.from_tensor_slices((dataset_validation, _metadata_validation_))
_dataset_training=_dataset_training.shuffle(dataset_training.shape[0])
_dataset_validation=_dataset_validation.shuffle(dataset_validation.shape[0]).batch(BATCHSIZE)

# Data augmentation:
"""
from keras.preprocessing.image import ImageDataGenerator
rango_rotacion=5; mov_ancho=0.15; mov_alto=0.10; rango_acercamiento=[0.05,0.25]; rango_inclinacion=10
datagen=ImageDataGenerator(rotation_range=rango_rotacion,width_shift_range=mov_ancho,height_shift_range=mov_alto,
    zoom_range=rango_acercamiento, shear_range=rango_inclinacion)
datagen.fit(dataset_training); data_gen_training=datagen.flow(dataset_training, metadata_training, batch_size=BATCHSIZE, shuffle=True)
"""

# PLOTS:
plt.figure(figsize=(10,10))
for i, (imagen, etiqueta) in enumerate(_dataset_training.take(15)):
  imagen=imagen.numpy().reshape((60,60))
  plt.subplot(5,5,i+1); plt.xticks([]); plt.yticks([])
  plt.grid(False); plt.imshow(imagen, cmap=plt.cm.binary)
  #plt.xlabel('{}'.format(class_names[etiqueta]))
#plt.savefig("stars-to-PIL.png")
plt.show()
###

# building a linear stack of layers with the sequential model:
opt='adam'
""" PRIMER MODELO (fail):
epochs=15; BATCHSIZE=64; model=Sequential()
model.add(Conv2D(32, kernel_size=(5,5), strides=(1,1), padding='same', kernel_regularizer=L2(1e-3), activation=tfa.activations.mish, input_shape=(60,60,1))) # convolutional layer#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2))); model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(5,5), strides=(1,1), padding='same', kernel_regularizer=L2(1e-3), activation=tfa.activations.mish)) # convolutional layer
model.add(MaxPool2D(pool_size=(2,2))); model.add(Dropout(0.35)) #,tfa.activations.mish
model.add(Conv2D(64, kernel_size=(5,5), strides=(1,1), padding='same', kernel_regularizer=L2(1e-3), activation=tfa.activations.mish)) # convolutional layer
model.add(MaxPool2D(pool_size=(3,3))); model.add(Dropout(0.35))
model.add(Flatten()) # flatten output of convolution
model.add(Dense(32, activation=tfa.activations.mish, kernel_regularizer=L2(1e-2))) # hidden layer #tfa.activations.mish
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax', kernel_regularizer=L2(1e-2))) # output layer
model.summary()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
"""
epochs=15; BATCHSIZE=32; model=Sequential() # Activation function: tfa.activations.mish
model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(60,60,1))) # convolutional layer #, kernel_regularizer=L2(1e-2)
model.add(MaxPool2D(pool_size=(2,2))); model.add(Dropout(0.25))
model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')) # convolutional layer , kernel_regularizer=L2(1e-2)
model.add(MaxPool2D(pool_size=(2,2))); model.add(Dropout(0.30)) #,tfa.activations.mish
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')) # convolutional layer #, kernel_regularizer=L2(1e-3), LeakyReLU(alpha=0.5)
model.add(MaxPool2D(pool_size=(2,2))); model.add(Dropout(0.30))
model.add(Flatten()) # flatten output of convolution
model.add(Dense(8, activation='relu')); model.add(Dropout(0.35)) # hidden layer #tfa.activations.mish
#
model.add(Dense(n_classes, activation='softmax')) # output layer
model.summary()#; plot_model(model, to_file='flujo_prev.png', show_shapes=True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']) # compiling the sequential model
#data_gen_training=datagen.flow(dataset_training, metadata_training, batch_size=BATCHSIZE, shuffle=True)
# MODEL PLOTS: 
#modelplot=ModelPlot(model=model,grid=False,connection=True,linewidth=0.1); modelplot.show()
#view_model=ann_viz(model, view=True, title="Convolutional Neural Network, TFG")
# TRAINING:
print("Training the model...")
train_model=model.fit(dataset_training, metadata_training, batch_size=BATCHSIZE, epochs=epochs, steps_per_epoch=int(np.ceil(dataset_training.shape[0]/float(BATCHSIZE))), 
    verbose=2,validation_data=(dataset_validation, metadata_validation), validation_steps=int(np.ceil(metadata_validation.shape[0]/float(BATCHSIZE))))
print("Model trained!"); model.save('model_trained.h5') # model saved

# PLOTS - ACCURACY, LOSS:
_acc=train_model.history['accuracy']; _val_acc=train_model.history['val_accuracy']
_loss=train_model.history['loss']; _val_loss=train_model.history['val_loss']
epochs=range(1,len(_acc)+1)

plt.plot(epochs, _acc,'r',label='Training accuracy')
plt.plot(epochs, _val_acc,'b',label='Validation accuracy')
plt.yticks(np.arange(0.5, 1, 0.1))
plt.title('Training and Validation: accuracy')
plt.legend(); plt.figure()

plt.plot(epochs, _loss,'r',label='Training loss')
plt.plot(epochs, _val_loss,'b',label='Validation loss')
#plt.yticks(np.arange(0, 1, 0.1))
plt.title('Training and Validation: loss')
plt.legend()#; plt.figure()
plt.show()
