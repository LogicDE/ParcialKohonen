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

# Función para procesar imágenes en una carpeta de letras y guardar en CSV
def procesar_imagenes_y_guardar(carpeta_imagenes, output_filepath):
    datos_imagenes = []
    etiquetas = []

    # Procesar cada carpeta de letra
    for letra in os.listdir(carpeta_imagenes):
        ruta_letra = os.path.join(carpeta_imagenes, letra)
        if os.path.isdir(ruta_letra):  # Asegurarse de que es un directorio
            for filename in os.listdir(ruta_letra):
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    filepath = os.path.join(ruta_letra, filename)
                    sumas_columnas = procesar_imagen(filepath)
                    datos_imagenes.append(sumas_columnas)
                    etiquetas.append(letra)  # Guardar la etiqueta de la letra

    # Convertir los datos a un DataFrame de Pandas
    df = pd.DataFrame(datos_imagenes)

    # Agregar la columna de etiquetas
    df['Etiqueta'] = etiquetas

    # Guardar el DataFrame en un archivo CSV
    df.to_csv(output_filepath, index=False)

    print(f"Datos procesados y guardados en {output_filepath}.")
