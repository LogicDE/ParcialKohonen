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
    global red_kohonen, tasa_aprendizaje_entry, iteraciones_entry  # Usar las variables globales
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
    global red_kohonen
    if red_kohonen is None:
        messagebox.showerror("Error", "Primero debe cargar un dataset.")
        return

    try:
        # Crear label para mostrar DM si no existe
        if not hasattr(entrenar_red, 'dm_label'):
            entrenar_red.dm_label = tk.Label(frame, text="DM: -", font=fuente_label, bg="#ffffff")
            entrenar_red.dm_label.pack(padx=10, pady=5)

        # Función callback para actualizar el DM en la interfaz
        def actualizar_dm(texto):
            entrenar_red.dm_label.config(text=texto)
            root.update()

        # Asignar callback a la red
        red_kohonen.set_callback(actualizar_dm)

        # Entrenar la red
        dataset = pd.read_csv(filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])).values
        if not np.issubdtype(dataset.dtype, np.number):
            messagebox.showerror("Error", "El dataset debe contener solo datos numéricos.")
            return
            
        red_kohonen.entrenar(dataset)
        messagebox.showinfo("Entrenamiento Completo", f"La red ha sido entrenada con éxito.\nMejor DM alcanzado: {red_kohonen.mejor_dm:.6f}")
    except Exception as e:
        messagebox.showerror("Error", f"Error durante el entrenamiento: {e}")

# Función para simular la red
def simular_red():
    global red_kohonen
    if red_kohonen is None:
        messagebox.showerror("Error", "Primero debe cargar y entrenar la red.")
        return

    try:
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            # Leer el CSV sin encabezados
            data = pd.read_csv(filepath, header=None).values

            # Comprobar la forma del data
            print(f"Forma del data: {data.shape}")  # Para verificar cuántas filas y columnas tiene

            # Iterar sobre las filas (si quieres simular cada imagen)
            for i in range(data.shape[0]):
                patron = data[i]  # Extraer el vector de la fila i

                print(f"Longitud del patrón: {len(patron)}")
                print(f"Número de entradas de la red: {red_kohonen.num_entradas}")

                # Verificar que el patrón contenga solo datos numéricos
                if not np.issubdtype(patron.dtype, np.number):
                    messagebox.showerror("Error", "El patrón debe contener solo datos numéricos.")
                    return

                # Verificar la dimensión del patrón
                if len(patron) != red_kohonen.num_entradas:
                    messagebox.showerror("Error", f"El patrón debe tener {red_kohonen.num_entradas} entradas.")
                    return

                # Simular la red con el patrón cargado
                salida = red_kohonen.simular(patron)

                # Convertir la salida a cadena para mostrarla
                salida_str = ', '.join(map(str, salida))
                messagebox.showinfo(f"Simulación Completa - Imagen {i+1}", f"Salida de la red: {salida_str}")

    except Exception as e:
        messagebox.showerror("Error", f"Error durante la simulación: {e}")

# Configuración de la interfaz principal
root = tk.Tk()
root.title("Kohonen - Cargar Dataset y Procesar Imágenes")
root.geometry("500x500")  # Aumentar el tamaño de la ventana
root.configure(bg="#eaeaea")  # Cambiar el color de fondo

# Crear un marco para organizar los elementos
frame = tk.Frame(root, bg="#ffffff", bd=2, relief=tk.RAISED)
frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

# Estilo de fuente
fuente_titulo = ("Helvetica", 18, "bold")
fuente_label = ("Helvetica", 12)
fuente_boton = ("Helvetica", 12)

# Título
titulo = tk.Label(frame, text="Kohonen Network", font=fuente_titulo, bg="#ffffff")
titulo.pack(pady=10)

# Botón para cargar el dataset desde un archivo CSV
button_cargar = tk.Button(frame, text="Cargar Dataset", command=cargar_dataset, font=fuente_boton, bg="#4CAF50", fg="white", relief=tk.FLAT)
button_cargar.pack(padx=10, pady=10, fill=tk.X)

# Botón para entrenar la red
button_entrenar = tk.Button(frame, text="Entrenar Red", command=entrenar_red, font=fuente_boton, bg="#2196F3", fg="white", relief=tk.FLAT)
button_entrenar.pack(padx=10, pady=10, fill=tk.X)

# Botón para simular la red
button_simular = tk.Button(frame, text="Simular Red", command=simular_red, font=fuente_boton, bg="#FF9800", fg="white", relief=tk.FLAT)
button_simular.pack(padx=10, pady=10, fill=tk.X)

# Etiquetas para mostrar el número de entradas y patrones
label_entradas = tk.Label(frame, text="Número de entradas: -", font=fuente_label, bg="#ffffff")
label_entradas.pack(padx=10, pady=5)

label_patrones = tk.Label(frame, text="Número de patrones: -", font=fuente_label, bg="#ffffff")
label_patrones.pack(padx=10, pady=5)

# Paso 6: Tipo de competencia
competencia_var = tk.StringVar(value='dura')  # Valor por defecto
tk.Label(frame, text="Tipo de Competencia:", font=fuente_label, bg="#ffffff").pack(pady=5)

# Centrar los botones de opción
radiobutton_frame = tk.Frame(frame, bg="#ffffff")  # Crear un marco para los radiobuttons
radiobutton_frame.pack(pady=5)  # Empaquetar el marco

tk.Radiobutton(radiobutton_frame, text="Competencia Dura", variable=competencia_var, value='dura', bg="#ffffff").pack(anchor=tk.W, padx=20)  # Ajustar el padding
tk.Radiobutton(radiobutton_frame, text="Competencia Blanda", variable=competencia_var, value='blanda', bg="#ffffff").pack(anchor=tk.W, padx=20)  # Ajustar el padding

# Paso 10: Tasa de aprendizaje
tk.Label(frame, text="Tasa de Aprendizaje:", font=fuente_label, bg="#ffffff").pack(pady=5)
tasa_aprendizaje_entry = tk.Entry(frame, font=fuente_label)
tasa_aprendizaje_entry.pack(padx=10, pady=5)
tasa_aprendizaje_entry.insert(0, "0.1")  # Valor por defecto

# Paso 9: Número de iteraciones
tk.Label(frame, text="Número de Iteraciones:", font=fuente_label, bg="#ffffff").pack(pady=5)
iteraciones_entry = tk.Entry(frame, font=fuente_label)
iteraciones_entry.pack(padx=10, pady=5)
iteraciones_entry.insert(0, "100")  # Valor por defecto

root.mainloop()
