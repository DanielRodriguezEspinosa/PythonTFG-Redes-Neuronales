# Programas Python - TFG, Redes Neuronales

En el presente repositorio se muestran los programas principales utilizados en la memoria del **Trabajo Fin de Grado en Matemáticas**: *On the use of Machine Learning techniques to discriminate astronomical images*.

Para usar los programas, descargar el repositorio *.zip* (`Code` + `Download ZIP`) para usarlo posteriormente en Google Colab o en el propio IDLE correspondiente (SublimeText ha sido usado para los programas aquí presentes).

## Programas principales:

- *make_stamps.py* se usan las listas: imglist_training.lst, imglist_validation.lst y imglist_evaluation.lst. Detecta las fuentes estelares y las recorta en cajas para la posterior clasificación, generando los archivos *.h5*: *training.h5*, *validation.h5*, *evaluation.h5*, *training_plus.h5* (dado que no se permiten aquí ficheros >25Mb, se ha hecho un *training_pruebas.h5* con menos imágenes por si se quiere probar el programa).
- *Clasification.py* clasifica los recortes de imagen mediante una red neuronal convolucional en 3 grupos diferentes: tipo *Circular*, tipo *Cometa* y tipo *Donut*. Se guarda el modelo entrenado en *model_trained.h5*.
- *Evaluate_model.py* se cargan los datos y metadatos de *model_trained.h5* para evaluar los resultados obtenidos.

## Programas secundarios:

- *Numbers.py* + *Numbers.html*: primero correr el programa python


## Programas extras:

- *make_stamps.py*

- *make_stamps.py*

- *make_stamps.py*

- *make_stamps.py*
