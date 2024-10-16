import os
import shutil
# Directorio donde están actualmente todas las imágenes
directorio_origen = 'letras'
# Directorio donde se crearán las nuevas carpetas
directorio_destino = 'letras_organizadas'
# Lista de letras (todas en mayúsculas, con 'I' en lugar de 'L')
letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'O', 'U', 'T']
# Crear las carpetas para cada letra si no existen
for letra in letras:
    nueva_carpeta = os.path.join(directorio_destino, f'letra{letra}')
    if not os.path.exists(nueva_carpeta):
        os.makedirs(nueva_carpeta)

# Mover las imágenes a las carpetas correspondientes
for archivo in os.listdir(directorio_origen):
    if archivo.endswith('.jpg') or archivo.endswith('.png'):  # Ajusta esto según el formato de tus imágenes
        primera_letra = archivo[0].upper()  # Tomar la primera letra y convertirla a mayúscula
        if primera_letra in letras:
            origen = os.path.join(directorio_origen, archivo)
            destino = os.path.join(directorio_destino, f'letra{primera_letra}', archivo)
            shutil.copy2(origen, destino)  # Usamos copy2 en lugar de move para mantener los originales

print("Reorganización de imágenes completada.")