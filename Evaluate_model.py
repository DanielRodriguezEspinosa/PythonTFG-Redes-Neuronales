import numpy as np; import math; import matplotlib.pyplot as plt
from astropy.io import ascii, fits; import h5py
from keras.models import Sequential; import random
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, MaxPooling2D
import tensorflow as tf; from keras.utils import to_categorical
import tensorflow_addons as tfa
#from sklearn.metrics import accuracy_score #to calculate accuracy

model=tf.keras.models.load_model('model_trained.h5')
class_names=['Circular', 'Cometa', 'Donut']
# Load file:
def read_hdf5_array(fname,hdf5_dir='./'):
    # Open the HDF5 file
    file=h5py.File(hdf5_dir+fname+".h5", "r+")
    stamps=np.array(file["/stamps"])
    labels=np.array(file["/labels"]).astype(np.uint16)
    return stamps, labels

dataset_evaluation, _metadata_evaluation=read_hdf5_array('evaluation', hdf5_dir='./')
# -> numpy array:
_metadata_evaluation_=[]
for j in range(_metadata_evaluation.shape[0]):
    _metadata_evaluation_.append(_metadata_evaluation[j][0])
metadata_evaluation=np.array(_metadata_evaluation_); _metadata_evaluation_=np.array(_metadata_evaluation_)
# building the input vector from the 60x60 pixels
dataset_evaluation=dataset_evaluation.reshape(dataset_evaluation.shape[0], 60, 60, 1); dataset_evaluation=dataset_evaluation.astype('float32')
n_classes=len(class_names); BATCHSIZE=32
metadata_evaluation=to_categorical(metadata_evaluation, n_classes)
# -> tensor:
_dataset_evaluation=tf.data.Dataset.from_tensor_slices((dataset_evaluation, _metadata_evaluation_))
_dataset_evaluation=_dataset_evaluation.shuffle(dataset_evaluation.shape[0]).batch(BATCHSIZE)

# evaluating the model:
test_loss, test_accuracy=model.evaluate(dataset_evaluation, metadata_evaluation, steps=int(np.ceil(metadata_evaluation.shape[0]/BATCHSIZE)))

for test_images, test_labels in _dataset_evaluation.take(12):
    test_images=test_images.numpy(); test_labels=test_labels.numpy()
    predictions=model.predict(test_images)

def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img=predictions_array[i], true_labels[i], images[i]
    plt.grid(False); plt.xticks([]); plt.yticks([]); plt.imshow(img[...,0], cmap=plt.cm.binary)
    predicted_label=np.argmax(predictions_array)
    if predicted_label==true_label:
        color='blue'
    else:
        color='red'
    plt.xlabel("Prediction: {}".format(class_names[predicted_label]), color=color)
    plt.ylabel("{}".format(class_names[true_label]), fontsize=10)  

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label=predictions_array[i], true_label[i]
    plt.grid(False); plt.xticks([]); plt.yticks([])
    thisplot=plt.bar(range(n_classes), predictions_array, color="#888888")
    predicted_label=np.argmax(predictions_array); thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

numrows=4; numcols=3; numimages=numrows*numcols; plt.figure(figsize=(2*2*numcols, 2*numrows))
for i in range(numimages):
    plt.subplot(numrows, 2*numcols, 2*i+1); plot_image(i, predictions, test_labels, test_images)
    plt.subplot(numrows, 2*numcols, 2*i+2); plot_value_array(i, predictions, test_labels)
plt.show()
