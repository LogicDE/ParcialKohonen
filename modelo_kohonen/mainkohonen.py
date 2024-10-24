import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from configuraciones.normalimage import procesar_imagenes_y_guardar
from configuraciones.creaciondearchivoporletra import procesar_imagenes_y_guardar, procesar_carpetas
from configuraciones.creacionred import RedKohonen

# Variable global para almacenar la red
red_kohonen = None

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
    global red_kohonen, tasa_aprendizaje_entry, iteraciones_entry
    try:
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            dataset = pd.read_csv(filepath)

            if not np.issubdtype(dataset.dtypes.iloc[0], np.number):
                messagebox.showerror("Error", "El dataset debe contener solo datos numéricos.")
                return

            num_entradas = dataset.shape[1]
            num_patrones = dataset.shape[0]

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
    global red_kohonen
    if red_kohonen is None:
        messagebox.showerror("Error", "Primero debe cargar un dataset.")
        return

    try:
        dataset = pd.read_csv(filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])).values
        if not np.issubdtype(dataset.dtype, np.number):
            messagebox.showerror("Error", "El dataset debe contener solo datos numéricos.")
            return
        red_kohonen.entrenar(dataset)
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

                salida_str = ', '.join(map(str, neurona_vencedora))
                messagebox.showinfo(f"Simulación Completa - Imagen {i+1}", f"Neurona ganadora en coordenadas: {salida_str}")

                print(neuronas_vencedoras)  # Asegúrate de que contiene pares (x, y)

            # Análisis de similitud
            for j in range(len(neuronas_vencedoras)):
                neurona = neuronas_vencedoras[j]
                distancia_neurona = distancias[j]

                print(f"Patrón {j+1}: Neurona vencedora: {neurona}, Distancia: {distancia_neurona}")

                # Aquí puedes implementar condiciones para evaluar similitudes
                # Por ejemplo, puedes definir un umbral para considerar que son similares
                if distancia_neurona[neurona] < 0.1:  # Umbral ejemplo
                    print(f"Patrón {j+1} es similar a la neurona vencedora.")
                else:
                    print(f"Patrón {j+1} no es similar a la neurona vencedora.")

            # Graficar el mapa de Kohonen
            graficar_mapa_kohonen(neuronas_vencedoras)

    except Exception as e:
        messagebox.showerror("Error", f"Error durante la simulación: {e}")


# Función para graficar el mapa de Kohonen y las activaciones
def graficar_mapa_kohonen(neuronas_vencedoras, salidas_neuronas, mapa_dim=10):
    mapa_kohonen = np.zeros((mapa_dim, mapa_dim))

    for i, (x, y) in enumerate(neuronas_vencedoras):
        print(f"Neurona vencedora para imagen {i+1}: ({x}, {y}) - Salida: {salidas_neuronas[i]}")
        mapa_kohonen[y, x] = 1

    plt.figure(figsize=(8, 8))
    plt.imshow(mapa_kohonen, cmap='viridis')
    plt.title("Mapa Autoorganizado de Kohonen (SOM)")
    plt.colorbar(label='Activación de la neurona vencedora')
    plt.show()


#INTERFAZ-----------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
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
