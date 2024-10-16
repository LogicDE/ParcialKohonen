import pandas as pd
import numpy as np
import os
import random
from PIL import Image
from tkinter import Tk
from tkinter import filedialog

# Función para procesar una imagen y devolver la suma de sus columnas
def procesar_imagen(filepath):
    img = Image.open(filepath).convert('L')  # Abrir la imagen en escala de grises
    img_array = np.array(img)  # Convertir la imagen a un array numpy
    binarizada = (img_array < 128).astype(int)  # Binarizar la imagen
    sumas_columnas = binarizada.sum(axis=0)  # Sumar las columnas
    return sumas_columnas

# Función para procesar imágenes en una carpeta y guardar en CSV
def procesar_imagenes_y_guardar(carpeta_imagenes, output_filepath, porcentaje_entrenamiento=0.8):
    datos_imagenes = []
    etiquetas = []

    # Obtener todas las imágenes en la carpeta
    imagenes = [f for f in os.listdir(carpeta_imagenes) if f.endswith(".png") or f.endswith(".jpg")]
    
    # Calcular cuántas imágenes serán seleccionadas (80%)
    cantidad_entrenamiento = int(len(imagenes) * porcentaje_entrenamiento)

    # Seleccionar aleatoriamente el 80% de las imágenes
    imagenes_seleccionadas = random.sample(imagenes, cantidad_entrenamiento)

    for filename in imagenes_seleccionadas:
        filepath = os.path.join(carpeta_imagenes, filename)
        sumas_columnas = procesar_imagen(filepath)
        datos_imagenes.append(sumas_columnas)  # Agregar las sumas
        etiquetas.append(filename)  # Agregar la etiqueta

    # Verifica los datos antes de crear el DataFrame
    print("Datos de imágenes (sumas):", datos_imagenes)
    print("Etiquetas:", etiquetas)

    # Convertir los datos a un DataFrame de Pandas
    df = pd.DataFrame(datos_imagenes)

    # Verifica el DataFrame antes de guardar
    print("DataFrame antes de guardar:")
    print(df)

    # Guardar el DataFrame en un archivo CSV
    df.to_csv(output_filepath, sep=',', index=False)

    print(f"Datos procesados y guardados en {output_filepath}.")

# Función para seleccionar una carpeta
def seleccionar_carpeta():
    root = Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter
    carpeta_seleccionada = filedialog.askdirectory()  # Abrir diálogo para seleccionar carpeta
    return carpeta_seleccionada

# Ejemplo de uso
carpeta_base_datos = seleccionar_carpeta()  # Seleccionar carpeta interactiva
archivo_salida = "entrenamiento.csv"
procesar_imagenes_y_guardar(carpeta_base_datos, archivo_salida)
