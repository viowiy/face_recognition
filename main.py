# Importando las librerías...

import os

import numpy as np
from numpy import genfromtxt
import pandas as pd

import PIL
from PIL import Image

import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.keras.models import model_from_json

K.set_image_data_format('channels_last')

import streamlit as st
import json

import cv2

import tempfile

# Cargando el modelo...

json_file = open('./keras-facenet-h5/model.json','r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights('./keras-facenet-h5/model.h5')

# Instanciando el modelo preentrenado...

FRmodel = model

# Función de codificación de la imagen a un vector de 128...
#tf.keras.backend.set_image_data_format('channels_last')
def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)


# Cargando la Base de Datos de rostros...
# Leer el archivo JSON
with open('./keras-facenet-h5/db_facer2.json', 'r') as file:
    database = json.load(file)

# Función para convertir listas a ndarrays
def convert_list_to_ndarray(d):
    for key, value in d.items():
        if isinstance(value, list):
            d[key] = np.array(value)
        elif isinstance(value, dict):
            convert_list_to_ndarray(value)

convert_list_to_ndarray(database)


# Código de reconocimiento de rostros...
# UNQ_C3(UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: who_is_it

def who_is_it(image_path, database, model):
    """
    Implements face recognition for the office by finding who is the person on the image_path image.

    Arguments:
        image_path -- path to an image
        database -- database containing image encodings along with the name of the person on the image
        model -- your Inception model instance in Keras

    Returns:
        min_dist -- the minimum distance between image_path encoding and the encodings from the database
        identity -- string, the name prediction for the person on image_path
    """

    ### START CODE HERE

    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding =   img_to_encoding(image_path, model)

    ## Step 2: Find the closest encoding ##

    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
        dist =  np.linalg.norm(encoding - db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist<min_dist:
            min_dist = dist
            identity = name
    ### END CODE HERE

    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity

# Función para captura de la imagen

def capture_image():
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    # Verificar si la cámara se abrió correctamente
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return None

    # Capturar un frame de la cámara
    ret, frame = cap.read()

    # Verificar si se capturó correctamente el frame
    if not ret:
        print("Error: No se pudo capturar la imagen.")
        return None

    # Liberar la cámara
    cap.release()

    return frame

# Capturar una imagen
captured_image = capture_image()

# Verificar si se capturó la imagen
if captured_image is not None:
    # Guardar la imagen capturada en un archivo
    cv2.imwrite("captured_image.jpg", captured_image)
    print("Imagen guardada como 'captured_image.jpg'")
else:
    print("No se pudo capturar la imagen.")





# Crear una aplicación Streamlit
def main():
    st.title('Reconocimiento Facial')

    # Permitir al usuario cargar una imagen
    uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

    # Verificar si se cargó una imagen
    if uploaded_file is not None:
        # Guardar temporalmente la imagen cargada
        temp_image = tempfile.NamedTemporaryFile(delete=False)
        temp_image.write(uploaded_file.read())
        temp_image.close()

        # Mostrar la imagen cargada
        image = Image.open(temp_image.name)
        st.image(image, caption='Imagen cargada', use_column_width=True)

        # Verificar si el usuario quiere verificar la identidad
        if st.button('Verificar Identidad'):
            with st.spinner('Verificando identidad...'):
                # Llamar a la función who_is_it para reconocimiento facial
                min_dist, identity = who_is_it(temp_image.name, database, FRmodel)
                # Mostrar el resultado del reconocimiento facial
                if min_dist < 1.35:  # Umbral de distancia... antes 0.7... Actual 1.35..
                    if identity == "ernesto":
                        acceso = "No Autorizado: Ya no cursa la Maestría."
                    else:
                        acceso = "Autorizado: Estudiante de la Maestría."
                    st.success(f'Identificado como: {identity} (Distancia: {min_dist}). {acceso}')
                else:
                    st.error(f'No Identificado. Pero se parece a {identity} (Distancia: {min_dist})')

# Ejecutar la aplicación
if __name__ == "__main__":
    main()