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
def procesar_imagenes_y_guardar(carpeta_imagenes, output_filepath_entrenamiento, output_filepath_prueba, porcentaje_entrenamiento=0.8):
    datos_imagenes = []
    etiquetas = []

    # Obtener todas las imágenes en la carpeta
    imagenes = [f for f in os.listdir(carpeta_imagenes) if f.endswith(".png") or f.endswith(".jpg")]

    # Calcular cuántas imágenes serán seleccionadas (80%)
    cantidad_entrenamiento = int(len(imagenes) * porcentaje_entrenamiento)

    # Seleccionar aleatoriamente el 80% de las imágenes
    imagenes_seleccionadas = random.sample(imagenes, cantidad_entrenamiento)

    # Las imágenes restantes serán el 20%
    imagenes_restantes = list(set(imagenes) - set(imagenes_seleccionadas))

    # Procesar las imágenes seleccionadas
    for filename in imagenes_seleccionadas:
        filepath = os.path.join(carpeta_imagenes, filename)
        sumas_columnas = procesar_imagen(filepath)
        datos_imagenes.append(sumas_columnas)  # Agregar las sumas
        etiquetas.append(filename)  # Agregar la etiqueta

    # Verifica los datos antes de crear el DataFrame de entrenamiento
    print("Datos de imágenes (sumas) para entrenamiento:", datos_imagenes)
    print("Etiquetas para entrenamiento:", etiquetas)

    # Convertir los datos a un DataFrame de Pandas para entrenamiento
    df_entrenamiento = pd.DataFrame(datos_imagenes)

    # Verifica el DataFrame de entrenamiento antes de guardar
    print("DataFrame de entrenamiento antes de guardar:")
    print(df_entrenamiento)

    # Guardar el DataFrame en un archivo CSV para entrenamiento
    df_entrenamiento.to_csv(output_filepath_entrenamiento, sep=',', index=False)

    print(f"Datos de entrenamiento procesados y guardados en {output_filepath_entrenamiento}.")

    # Procesar las imágenes restantes para el 20%
    datos_imagenes_restantes = []
    etiquetas_restantes = []

    for filename in imagenes_restantes:
        filepath = os.path.join(carpeta_imagenes, filename)
        sumas_columnas = procesar_imagen(filepath)
        datos_imagenes_restantes.append(sumas_columnas)  # Agregar las sumas
        etiquetas_restantes.append(filename)  # Agregar la etiqueta

    # Verifica los datos restantes antes de crear el DataFrame de prueba
    print("Datos de imágenes (sumas) para prueba:", datos_imagenes_restantes)
    print("Etiquetas para prueba:", etiquetas_restantes)

    # Convertir los datos a un DataFrame de Pandas para prueba
    df_prueba = pd.DataFrame(datos_imagenes_restantes)

    # Verifica el DataFrame de prueba antes de guardar
    print("DataFrame de prueba antes de guardar:")
    print(df_prueba)

    # Guardar el DataFrame en un archivo CSV para prueba
    df_prueba.to_csv(output_filepath_prueba, sep=',', index=False)

    print(f"Datos de prueba procesados y guardados en {output_filepath_prueba}.")

# Función para seleccionar una carpeta
def seleccionar_carpeta():
    root = Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter
    carpeta_seleccionada = filedialog.askdirectory()  # Abrir diálogo para seleccionar carpeta
    return carpeta_seleccionada

# Ejemplo de uso
carpeta_base_datos = seleccionar_carpeta()  # Seleccionar carpeta interactiva
archivo_salida_entrenamiento = "entrenamiento.csv"
archivo_salida_prueba = "entrenamiento20.csv"
procesar_imagenes_y_guardar(carpeta_base_datos, archivo_salida_entrenamiento, archivo_salida_prueba)
