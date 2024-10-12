import os
import random
import shutil

# Directorio donde están las imágenes organizadas
directorio_origen = 'letras_organizadas'

# Directorios para entrenamiento y simulación
directorio_entrenamiento = 'entrenamiento'
directorio_simulacion = 'simulacion'

# Crear directorios si no existen
for directorio in [directorio_entrenamiento, directorio_simulacion]:
    if not os.path.exists(directorio):
        os.makedirs(directorio)

# Obtener todas las imágenes
todas_las_imagenes = []
for letra_carpeta in os.listdir(directorio_origen):
    ruta_letra = os.path.join(directorio_origen, letra_carpeta)
    if os.path.isdir(ruta_letra):
        todas_las_imagenes.extend([os.path.join(ruta_letra, img) for img in os.listdir(ruta_letra)])

# Mezclar aleatoriamente las imágenes
random.shuffle(todas_las_imagenes)

# Letras que deben estar en la simulación
letras_obligatorias = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'O', 'T', 'U']

# Seleccionar al menos una imagen de cada letra obligatoria para la simulación
imagenes_simulacion = []
for letra in letras_obligatorias:
    for img in todas_las_imagenes:
        if letra in img:  # Verificar si la imagen corresponde a la letra
            imagenes_simulacion.append(img)
            todas_las_imagenes.remove(img)  # Eliminar la imagen de la lista para no seleccionarla de nuevo
            break  # Salir del bucle una vez que se ha encontrado una imagen

# Completar con imágenes aleatorias hasta llegar a 19
imagenes_simulacion += random.sample(todas_las_imagenes, 19 - len(imagenes_simulacion))

# Asegurarse de que las imágenes de simulación no se repitan en el entrenamiento
# Seleccionar 80 imágenes para entrenamiento, asegurando que no se repitan letras
imagenes_entrenamiento = []
letras_usadas = set()

for img in todas_las_imagenes:
    letra = img.split(os.path.sep)[-2]  # Obtener la letra de la ruta de la imagen
    if letra not in letras_usadas and len(imagenes_entrenamiento) < 80:
        imagenes_entrenamiento.append(img)
        letras_usadas.add(letra)  # Marcar la letra como utilizada

# Completar el entrenamiento con imágenes aleatorias si no se alcanzan 80
if len(imagenes_entrenamiento) < 80:
    imagenes_entrenamiento += random.sample(todas_las_imagenes, 80 - len(imagenes_entrenamiento))

# Copiar imágenes a los directorios correspondientes
for img in imagenes_entrenamiento:
    shutil.copy2(img, directorio_entrenamiento)

for img in imagenes_simulacion:
    shutil.copy2(img, directorio_simulacion)

print(f"Se han seleccionado {len(imagenes_entrenamiento)} imágenes para entrenamiento y {len(imagenes_simulacion)} para simulación.")