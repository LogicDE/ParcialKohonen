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

# Función para procesar imágenes en una carpeta y guardar en un archivo CSV
def procesar_imagenes_y_guardar(carpeta_letra, output_filepath):
    datos_imagenes = []
    etiquetas = []

    # Obtener todas las imágenes en la carpeta
    imagenes = [f for f in os.listdir(carpeta_letra) if f.endswith(".png") or f.endswith(".jpg")]

    for filename in imagenes:
        filepath = os.path.join(carpeta_letra, filename)
        sumas_columnas = procesar_imagen(filepath)
        datos_imagenes.append(sumas_columnas)
        etiquetas.append(filename)  # Guardar el nombre del archivo como etiqueta (puedes cambiarlo)

    # Convertir los datos a un DataFrame de Pandas
    df = pd.DataFrame(datos_imagenes)

    # Agregar la columna de etiquetas
    df['Etiqueta'] = etiquetas

    # Guardar el DataFrame en un archivo CSV
    df.to_csv(output_filepath, index=False)

    print(f"Datos de la carpeta {carpeta_letra} procesados y guardados en {output_filepath}.")

# Función para procesar múltiples carpetas
def procesar_carpetas(base_path):
    # Letras o carpetas que quieres procesar
    letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'O', 'T', 'U']

    for letra in letras:
        carpeta_letra = os.path.join(base_path, f"letra{letra}")
        archivo_salida = f"entrenamiento_{letra}.csv"

        # Asegurarse de que la carpeta existe
        if os.path.exists(carpeta_letra):
            procesar_imagenes_y_guardar(carpeta_letra, archivo_salida)
        else:
            print(f"La carpeta {carpeta_letra} no existe, saltando...")

# Ejemplo de uso
base_path = r"C:\Users\themo\OneDrive\Desktop\ParcialKohonen\letras_organizadas"
procesar_carpetas(base_path)
