import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Importar matplotlib para la visualización
from configuraciones.normalimage import procesar_imagenes_y_guardar
from configuraciones.creaciondearchivoporletra import procesar_imagenes_y_guardar, procesar_carpetas
from configuraciones.creacionred import RedKohonen

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
        # Entrenar la red de Kohonen con el dataset previamente cargado
        dataset = pd.read_csv(filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])).values
        if not np.issubdtype(dataset.dtype, np.number):
            messagebox.showerror("Error", "El dataset debe contener solo datos numéricos.")
            return
        red_kohonen.entrenar(dataset)
        messagebox.showinfo("Entrenamiento Completo", "La red ha sido entrenada con éxito.")
        messagebox.showinfo("DM Total", f"El DM total es: {red_kohonen.mejor_dm}")
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
            neuronas_vencedoras = []

            # Iterar sobre las filas (si quieres simular cada imagen)
            for i in range(data.shape[0]):
                patron = data[i]  # Extraer el vector de la fila i

                # Simular la red con el patrón cargado
                salida = red_kohonen.simular(patron)
                neuronas_vencedoras.append(salida)

                # Graficar la letra en el mapa de Kohonen
                graficar_letra_en_mapa_kohonen(patron, salida)
                
                # Comparar la salida con los patrones
                letra_reconocida = comparar_con_patrones(salida, data)
                print(f"Letra reconocida: {letra_reconocida}")


    except Exception as e:
        messagebox.showerror("Error", f"Error durante la simulación: {e}")
        
def comparar_con_patrones(salida, patrones):
    """Compara la salida de la red con los patrones y devuelve la letra reconocida."""
    # Aquí asumimos que los patrones están en la misma forma que la salida
    # Puedes ajustar la lógica de comparación según sea necesario
    distancias = np.linalg.norm(patrones - salida, axis=1)  # Calcular distancias
    indice_menor_distancia = np.argmin(distancias)  # Índice del patrón más cercano
    letra_reconocida = f"Patrón {indice_menor_distancia}"  # Ajusta esto según tu estructura de etiquetas
    return letra_reconocida

def graficar_mapa_kohonen(neuronas_vencedoras):
    """Grafica el mapa de Kohonen basado en las neuronas vencedoras."""

    # Convertir la lista de neuronas vencedoras a un arreglo NumPy
    neuronas_vencedoras = np.array(neuronas_vencedoras)

    # Graficar el mapa de Kohonen
    plt.figure(figsize=(10, 6))
    plt.scatter(neuronas_vencedoras[:, 0], neuronas_vencedoras[:, 1], c='blue', marker='o')
    plt.title("Mapa de Kohonen")
    plt.xlabel("Neuronas Vencedoras - Dimensión 1")
    plt.ylabel("Neuronas Vencedoras - Dimensión 2")
    plt.grid()
    plt.show()

def graficar_letra_en_mapa_kohonen(patron, salida):
    """Grafica la letra en el mapa de Kohonen basado en el patrón y la salida."""
    plt.figure(figsize=(5, 5))  # Tamaño de figura revertido a 5x5
    
    # Asegúrate de que el patrón tenga 10 elementos para poder representarlo como 2x5
    if patron.size == 10:
        plt.imshow(patron.reshape(2, 5), cmap='gray', interpolation='nearest')  # Asumiendo que el patrón es de 2x5
    else:
        messagebox.showerror("Error", "El patrón debe tener exactamente 10 entradas.")
        return

    plt.title("Letra Simulada")
    plt.colorbar()
    plt.scatter(salida[0], salida[1], color='red', s=100)  # Marcar la neurona vencedora
    plt.axis('off')  # Ocultar los ejes para una mejor visualización
    plt.show()

#INTERFAZ-----------------------------------------------------------------------------

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
