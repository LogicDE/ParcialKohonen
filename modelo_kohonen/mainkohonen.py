import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from configuraciones.normalimage import procesar_imagenes_y_guardar  # Asegúrate de que la ruta es correcta
from configuraciones.creaciondearchivoporletra import procesar_imagenes_y_guardar, procesar_carpetas
from configuraciones.creacionred import RedKohonen  # Asegúrate de que la ruta es correcta

# Variable global para almacenar la red
red_kohonen = None

# Función para cargar el dataset desde un archivo CSV
def cargar_dataset():
    global red_kohonen  # Usar la variable global
    try:
        # Abrir diálogo para seleccionar el archivo CSV
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            # Leer el archivo CSV
            dataset = pd.read_csv(filepath)

            # Verificar si el dataset tiene datos numéricos
            if not np.issubdtype(dataset.dtypes.iloc[0], np.number):
                messagebox.showerror("Error", "El dataset debe contener solo datos numéricos.")
                return

            # Obtener el número de entradas (columnas) y patrones (filas)
            num_entradas = dataset.shape[1]  # Columnas
            num_patrones = dataset.shape[0]  # Filas

            # Mostrar información en la interfaz
            label_entradas.config(text=f"Número de entradas: {num_entradas}")
            label_patrones.config(text=f"Número de patrones: {num_patrones}")

            # Crear la red de Kohonen
            tipo_competencia = competencia_var.get()  # Obtener tipo de competencia
            
            try:
                tasa_aprendizaje = float(tasa_aprendizamiento_entry.get())
                if tasa_aprendizaje <= 0 or tasa_aprendizaje > 1:
                    raise ValueError("La tasa de aprendizaje debe estar entre 0 y 1.")
            except ValueError as ve:
                messagebox.showerror("Error", f"Tasa de Aprendizaje: {ve}")
                return
            
            try:
                num_iteraciones = int(iteraciones_entry.get())
                if num_iteraciones <= 0:
                    raise ValueError("El número de iteraciones debe ser mayor que 0.")
            except ValueError as ve:
                messagebox.showerror("Error", f"Número de Iteraciones: {ve}")
                return

            # Crear la red de Kohonen
            red_kohonen = RedKohonen(
                num_entradas=num_entradas,
                tipo_competencia=tipo_competencia,
                tasa_aprendizaje=tasa_aprendizaje,
                num_iteraciones=num_iteraciones
            )

            messagebox.showinfo("Carga Completa", "Dataset cargado con éxito. Ahora puede entrenar la red.")

    except Exception as e:
        messagebox.showerror("Error", f"Error al cargar el dataset: {e}")

# Función para entrenar la red
def entrenar_red():
    global red_kohonen  # Usar la variable global
    if red_kohonen is None:
        messagebox.showerror("Error", "Primero debe cargar un dataset.")
        return

    try:
        # Entrenar la red de Kohonen con el dataset previamente cargado
        dataset = pd.read_csv(filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])).values
        if not np.issubdtype(dataset.dtype, np.number):
            messagebox.showerror("Error", "El dataset debe contener solo datos numéricos.")
            return
        red_kohonen.entrenar(dataset)
        messagebox.showinfo("Entrenamiento Completo", "La red ha sido entrenada con éxito.")
    except Exception as e:
        messagebox.showerror("Error", f"Error durante el entrenamiento: {e}")

# Configuración de la interfaz principal
root = tk.Tk()
root.title("Kohonen - Cargar Dataset y Procesar Imágenes")

# Botón para cargar el dataset desde un archivo CSV
button_cargar = tk.Button(root, text="Cargar Dataset", command=cargar_dataset)
button_cargar.pack(padx=10, pady=10)

# Botón para entrenar la red
button_entrenar = tk.Button(root, text="Entrenar Red", command=entrenar_red)
button_entrenar.pack(padx=10, pady=10)

# Etiquetas para mostrar el número de entradas y patrones
label_entradas = tk.Label(root, text="Número de entradas: -")
label_entradas.pack(padx=10, pady=5)

label_patrones = tk.Label(root, text="Número de patrones: -")
label_patrones.pack(padx=10, pady=5)

# Paso 6: Tipo de competencia
competencia_var = tk.StringVar(value='dura')  # Valor por defecto
tk.Radiobutton(root, text="Competencia Dura", variable=competencia_var, value='dura').pack()
tk.Radiobutton(root, text="Competencia Blanda", variable=competencia_var, value='blanda').pack()

# Paso 10: Tasa de aprendizaje
tk.Label(root, text="Tasa de Aprendizaje:").pack()
tasa_aprendizamiento_entry = tk.Entry(root)
tasa_aprendizamiento_entry.pack(padx=10, pady=5)
tasa_aprendizamiento_entry.insert(0, "0.1")  # Valor por defecto

# Paso 9: Número de iteraciones
tk.Label(root, text="Número de Iteraciones:").pack()
iteraciones_entry = tk.Entry(root)
iteraciones_entry.pack(padx=10, pady=5)
iteraciones_entry.insert(0, "100")  # Valor por defecto

root.mainloop()
