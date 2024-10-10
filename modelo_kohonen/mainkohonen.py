import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from configuraciones.normalimage import procesar_imagenes_y_guardar  # Asegúrate de que la ruta es correcta

# Función para cargar el dataset desde un archivo CSV
def cargar_dataset():
    try:
        # Abrir diálogo para seleccionar el archivo CSV
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            # Leer el archivo CSV
            dataset = pd.read_csv(filepath)

            # Obtener el número de entradas (columnas) y patrones (filas)
            num_entradas = dataset.shape[1]  # Columnas
            num_patrones = dataset.shape[0]  # Filas

            # Mostrar en la interfaz
            label_entradas.config(text=f"Número de entradas: {num_entradas}")
            label_patrones.config(text=f"Número de patrones: {num_patrones}")

    except Exception as e:
        messagebox.showerror("Error", f"Error al cargar el dataset: {e}")

# Función para procesar imágenes y guardar en CSV
def procesar_imagenes():
    try:
        # Seleccionar la carpeta donde se encuentran las imágenes
        carpeta_imagenes = filedialog.askdirectory(title="Seleccionar carpeta con imágenes")
        if carpeta_imagenes:
            # Seleccionar la ubicación para guardar el archivo CSV
            output_filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if output_filepath:
                # Llamar a la función del archivo normalization.py
                procesar_imagenes_y_guardar(carpeta_imagenes, output_filepath)
                messagebox.showinfo("Éxito", "Las imágenes se han procesado y almacenado en el archivo CSV.")
    
    except Exception as e:
        messagebox.showerror("Error", f"Error al procesar las imágenes: {e}")

# Configuración de la interfaz principal
root = tk.Tk()
root.title("Kohonen - Cargar Dataset y Procesar Imágenes")

# Botón para cargar el dataset desde un archivo CSV
button_cargar = tk.Button(root, text="Cargar Dataset", command=cargar_dataset)
button_cargar.pack(padx=10, pady=10)

# Botón para procesar imágenes y guardar los datos en un archivo CSV
button_procesar_imagenes = tk.Button(root, text="Procesar Imágenes", command=procesar_imagenes)
button_procesar_imagenes.pack(padx=10, pady=10)

# Etiquetas para mostrar el número de entradas y patrones
label_entradas = tk.Label(root, text="Número de entradas: -")
label_entradas.pack(padx=10, pady=5)

label_patrones = tk.Label(root, text="Número de patrones: -")
label_patrones.pack(padx=10, pady=5)

root.mainloop()
