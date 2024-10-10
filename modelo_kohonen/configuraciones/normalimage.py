# normalization.py
from PIL import Image
import numpy as np
import os
import pandas as pd

# Función para procesar una imagen y devolver la suma de sus columnas
def procesar_imagen(filepath):
    # Abrir la imagen en escala de grises
    img = Image.open(filepath).convert('L')
    # Convertir la imagen a un array numpy
    img_array = np.array(img)

    # Binarizar la imagen (0 para blanco, 1 para negro)
    binarizada = (img_array < 128).astype(int)

    # Sumar las columnas
    sumas_columnas = binarizada.sum(axis=0)
    return sumas_columnas

# Función para procesar imágenes en una carpeta y guardar en CSV
def procesar_imagenes_y_guardar(carpeta_imagenes, output_filepath):
    datos_imagenes = []

    # Procesar cada imagen en la carpeta
    for filename in os.listdir(carpeta_imagenes):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            filepath = os.path.join(carpeta_imagenes, filename)
            sumas_columnas = procesar_imagen(filepath)
            datos_imagenes.append(sumas_columnas)

    # Convertir los datos a un DataFrame de Pandas
    df = pd.DataFrame(datos_imagenes)

    # Guardar el DataFrame en un archivo CSV
    df.to_csv(output_filepath, index=False)
