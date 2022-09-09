from keras import models  
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation, Input
from keras.models import Sequential, Model
from keras_visualizer import visualizer 
from keras import layers
from neuralplot import ModelPlot

# Creating Model (Example):
X_input=Input(shape=(32,32,3))
X=Conv2D(4, 3, activation='relu')(X_input)
X=MaxPool2D(2,2)(X)
X=Conv2D(16, 3, activation='relu')(X)
X=MaxPool2D(2,2)(X)
X=Conv2D(8, 3, activation='relu')(X)
X=MaxPool2D(2,2)(X)
X=Flatten()(X)
X=Dense(10, activation='relu')(X)
X=Dense(2, activation='softmax')(X)

model=Model(inputs=X_input, outputs=X)
modelplot=ModelPlot(model=model, grid=True, connection=True, linewidth=0.15)
modelplot.show()
