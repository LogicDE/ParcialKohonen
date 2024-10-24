import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from configuraciones.normalimage import procesar_imagenes_y_guardar
from configuraciones.creaciondearchivoporletra import procesar_imagenes_y_guardar, procesar_carpetas
from configuraciones.creacionred import RedKohonen

# Variables globales para almacenar la red y el dataset
red_kohonen = None
dataset_global = None  # Nueva variable global para el dataset

# Función para cargar los pesos desde un archivo
def cargar_pesos():
    global red_kohonen
    try:
        filepath = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy")])
        if filepath:
            red_kohonen.pesos = np.load(filepath)
            messagebox.showinfo("Carga de Pesos Completa", "Pesos cargados con éxito.")
    except Exception as e:
        messagebox.showerror("Error", f"Error al cargar los pesos: {e}")

# Función para cargar el dataset desde un archivo CSV
def cargar_dataset():
    global red_kohonen, tasa_aprendizaje_entry, iteraciones_entry, dataset_global
    try:
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            dataset_global = pd.read_csv(filepath)  # Cargar el dataset en la variable global

            if not np.issubdtype(dataset_global.dtypes.iloc[0], np.number):
                messagebox.showerror("Error", "El dataset debe contener solo datos numéricos.")
                return

            num_entradas = dataset_global.shape[1]
            num_patrones = dataset_global.shape[0]

            label_entradas.config(text=f"Número de entradas: {num_entradas}")
            label_patrones.config(text=f"Número de patrones: {num_patrones}")

            tipo_competencia = competencia_var.get()

            try:
                tasa_aprendizaje = float(tasa_aprendizaje_entry.get())
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
    global red_kohonen, dataset_global  # Usar la variable global del dataset
    if red_kohonen is None:
        messagebox.showerror("Error", "Primero debe cargar un dataset.")
        return

    try:
        if dataset_global is None:
            messagebox.showerror("Error", "Primero debe cargar un dataset.")
            return

        # Entrenar la red con el dataset cargado previamente
        red_kohonen.entrenar(dataset_global.values)  # Usar el dataset global
        messagebox.showinfo("Entrenamiento Completo", "La red ha sido entrenada con éxito.")
        messagebox.showinfo("DM Total", f"El DM total es: {red_kohonen.mejor_dm}")
    except Exception as e:
        messagebox.showerror("Error", f"Error durante el entrenamiento: {e}")

# Función para simular la red y mostrar las salidas junto con las neuronas vencedoras
def simular_red():
    global red_kohonen
    if red_kohonen is None:
        messagebox.showerror("Error", "Primero debe cargar y entrenar la red.")
        return
    
    try:
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            data = pd.read_csv(filepath, header=None).values
            print(f"Forma del data: {data.shape}")
            neuronas_vencedoras = []
            distancias = []  # Almacenar distancias para análisis
            diferencias_promedio = []  # Almacenar diferencias promedio

            for i in range(data.shape[0]):
                patron = data[i]
                print(f"Longitud del patrón: {len(patron)}")
                print(f"Número de entradas de la red: {red_kohonen.num_entradas}")

                if not np.issubdtype(patron.dtype, np.number):
                    messagebox.showerror("Error", "El patrón debe contener solo datos numéricos.")
                    return

                if len(patron) != red_kohonen.num_entradas:
                    messagebox.showerror("Error", f"El patrón debe tener {red_kohonen.num_entradas} entradas.")
                    return

                # Simular la red con el patrón cargado
                neurona_vencedora = red_kohonen.simular(patron)
                neuronas_vencedoras.append(neurona_vencedora)

                # Calcular distancia entre el patrón y la neurona vencedora
                distancia = red_kohonen.calcular_distancias(patron)
                distancias.append(distancia)

                # Identificar neurona vencedora
                index_vencedora = np.argmin(distancia)
                pesos_vencedora = red_kohonen.pesos[:, index_vencedora]

                # Calcular diferencia promedio entre el patrón y la neurona vencedora
                diferencia_promedio = np.mean(np.abs(patron - pesos_vencedora))
                diferencias_promedio.append(diferencia_promedio)

                salida_str = ', '.join(map(str, neurona_vencedora))
                messagebox.showinfo(f"Simulación Completa - Imagen {i+1}", 
                                    f"Neurona ganadora en coordenadas: {salida_str}, "
                                    f"Diferencia promedio: {diferencia_promedio:.4f}")

                print(neuronas_vencedoras)  # Asegúrate de que contiene pares (x, y)

            # Análisis de similitud
            for j in range(len(neuronas_vencedoras)):
                neurona = neuronas_vencedoras[j]
                distancia_neurona = distancias[j]
                diferencia_promedio = diferencias_promedio[j]

                print(f"Patrón {j+1}: Neurona vencedora: {neurona}, Distancia: {distancia_neurona}, "
                      f"Diferencia promedio: {diferencia_promedio:.4f}")

                # Evaluar similitudes
                if diferencia_promedio < 0.1:  # Umbral ejemplo
                    print(f"Patrón {j+1} es similar a la neurona vencedora.")
                else:
                    print(f"Patrón {j+1} no es similar a la neurona vencedora.")

    except Exception as e:
        messagebox.showerror("Error", f"Error durante la simulación: {e}")


#INTERFAZ-----------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Configuración de la interfaz principal
root = tk.Tk()
root.title("Kohonen - Cargar Dataset y Procesar Imágenes")
root.geometry("600x600")  # Aumentar el tamaño de la ventana
root.configure(bg="#f0f0f0")  # Cambiar el color de fondo

# Crear un marco para organizar los elementos
frame = tk.Frame(root, bg="#ffffff", bd=5, relief=tk.GROOVE)
frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

# Configurar la cuadrícula para que sea responsive
frame.grid_rowconfigure(0, weight=1)
frame.grid_rowconfigure(1, weight=1)
frame.grid_rowconfigure(2, weight=1)
frame.grid_rowconfigure(3, weight=1)
frame.grid_rowconfigure(4, weight=1)
frame.grid_rowconfigure(5, weight=1)
frame.grid_rowconfigure(6, weight=1)
frame.grid_columnconfigure(0, weight=1)

# Estilo de fuente
fuente_titulo = ("Arial", 20, "bold")
fuente_label = ("Arial", 12)
fuente_boton = ("Arial", 12)

# Título
titulo = tk.Label(frame, text="Kohonen Network", font=fuente_titulo, bg="#ffffff", fg="#333333")
titulo.grid(row=0, column=0, pady=10)

# Botón para cargar el dataset desde un archivo CSV
button_cargar = tk.Button(frame, text="Cargar Dataset", command=cargar_dataset, font=fuente_boton, bg="#4CAF50", fg="white", relief=tk.RAISED)
button_cargar.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

# Botón para entrenar la red
button_entrenar = tk.Button(frame, text="Entrenar Red", command=entrenar_red, font=fuente_boton, bg="#2196F3", fg="white", relief=tk.RAISED)
button_entrenar.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

# Botón para simular la red
button_simular = tk.Button(frame, text="Simular Red", command=simular_red, font=fuente_boton, bg="#FF9800", fg="white", relief=tk.RAISED)
button_simular.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

# Etiquetas para mostrar el número de entradas y patrones
label_entradas = tk.Label(frame, text="Número de entradas: -", font=fuente_label, bg="#ffffff", fg="#333333")
label_entradas.grid(row=4, column=0, padx=10, pady=5)

label_patrones = tk.Label(frame, text="Número de patrones: -", font=fuente_label, bg="#ffffff", fg="#333333")
label_patrones.grid(row=5, column=0, padx=10, pady=5)

# Paso 6: Tipo de competencia
competencia_var = tk.StringVar(value='dura')  # Valor por defecto
tk.Label(frame, text="Tipo de Competencia:", font=fuente_label, bg="#ffffff", fg="#333333").grid(row=6, column=0, pady=5)

# Centrar los botones de opción
radiobutton_frame = tk.Frame(frame, bg="#ffffff")  # Crear un marco para los radiobuttons
radiobutton_frame.grid(row=7, column=0, pady=5)  # Empaquetar el marco

tk.Radiobutton(radiobutton_frame, text="Competencia Dura", variable=competencia_var, value='dura', bg="#ffffff", fg="#333333").pack(anchor=tk.W, padx=20)  # Ajustar el padding
tk.Radiobutton(radiobutton_frame, text="Competencia Blanda", variable=competencia_var, value='blanda', bg="#ffffff", fg="#333333").pack(anchor=tk.W, padx=20)  # Ajustar el padding

# Paso 10: Tasa de aprendizaje
tk.Label(frame, text="Tasa de Aprendizaje:", font=fuente_label, bg="#ffffff", fg="#333333").grid(row=8, column=0, pady=5)
tasa_aprendizaje_entry = tk.Entry(frame, font=fuente_label, bd=2, relief=tk.SUNKEN)
tasa_aprendizaje_entry.grid(row=9, column=0, padx=10, pady=5)
tasa_aprendizaje_entry.insert(0, "0.1")  # Valor por defecto

# Paso 9: Número de iteraciones
tk.Label(frame, text="Número de Iteraciones:", font=fuente_label, bg="#ffffff", fg="#333333").grid(row=10, column=0, pady=5)
iteraciones_entry = tk.Entry(frame, font=fuente_label, bd=2, relief=tk.SUNKEN)
iteraciones_entry.grid(row=11, column=0, padx=10, pady=5)
iteraciones_entry.insert(0, "100")  # Valor por defecto

root.mainloop()
