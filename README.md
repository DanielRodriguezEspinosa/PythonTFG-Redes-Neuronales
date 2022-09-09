# Programas Python - TFG, Redes Neuronales

En el presente repositorio se muestran los programas principales utilizados en la memoria del **Trabajo Fin de Grado en Matemáticas**: *On the use of Machine Learning techniques to discriminate astronomical images*.

Para usar los programas, descargar el repositorio (`Code` + `Download ZIP`) para usarlo posteriormente en Google Colab o en el propio IDLE correspondiente en su ordenador (SublimeText ha sido usado para los programas aquí presentes).

## Programas principales:

- *make_stamps.py*: (este programa necesita de las imágenes *.fits*) se usan las listas: imglist_training.lst, imglist_validation.lst y imglist_evaluation.lst. Detecta las fuentes estelares y las recorta en cajas para la posterior clasificación, generando los archivos *.h5*: *training.h5*, *validation.h5*, *evaluation.h5*, *training_plus.h5*(>25Mb) (dado que no se permiten en GitHub ficheros >25Mb, se ha hecho un *training_pruebas.h5* con menos imágenes por si se quiere probar el siguiente programa).
- *Clasification.py*: clasifica los recortes de imagen mediante una red neuronal convolucional en 3 grupos diferentes: tipo *Circular*, tipo *Cometa* y tipo *Donut*. Se guarda el modelo entrenado en *model_trained.h5*.
- *Evaluate_model.py*: se cargan los datos y metadatos de *model_trained.h5* para evaluar los resultados obtenidos.

## Programas secundarios:

- *Make_Circles.py*: red neuronal multicapa, clasificación binaria.

- Escritura Números - HTML (*MNIST_CNN.py* + *index.html*): https://danielrodriguezespinosa.github.io/N-meros/ para probar la predicción de nuestra propia escritura en tiempo real.

- *Gradient_Descent+.py*: se visualiza el algoritmo del descenso del gradiente.

## Programas extras:

- *ModelPlot.py*: muestra la estructura de una *CNN*.

- *kernels_PIL.py*: se aplican filtros a la imagen *donut.png*.

- *Activation_functions.py*: se grafican las funciones de activación sigmoide, *tanh*, mish, etcétera.
