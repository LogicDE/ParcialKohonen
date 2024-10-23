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

# Configuración de estilos y colores
COLOR_FONDO = "#F0F2F6"
COLOR_FRAME = "#FFFFFF"
COLOR_BOTON_CARGAR = "#4CAF50"
COLOR_BOTON_ENTRENAR = "#2196F3" 
COLOR_BOTON_SIMULAR = "#FF9800"
PADDING_X = 30
PADDING_Y = 20

# Configuración de la interfaz principal
root = tk.Tk()
root.title("Red Neuronal de Kohonen")
root.geometry("800x700")
root.configure(bg=COLOR_FONDO)

# Frame principal con sombra y bordes redondeados
frame = tk.Frame(root, bg=COLOR_FRAME, bd=0)
frame.pack(padx=PADDING_X, pady=PADDING_Y, fill=tk.BOTH, expand=True)

# Estilos de fuente mejorados
fuente_titulo = ("Helvetica", 24, "bold")
fuente_subtitulo = ("Helvetica", 14, "bold")
fuente_label = ("Helvetica", 12)
fuente_boton = ("Helvetica", 12, "bold")

# Título con mejor espaciado y diseño
titulo_frame = tk.Frame(frame, bg=COLOR_FRAME)
titulo_frame.pack(pady=20, fill=tk.X)

titulo = tk.Label(titulo_frame, 
                 text="Red Neuronal de Kohonen",
                 font=fuente_titulo,
                 bg=COLOR_FRAME,
                 fg="#2C3E50")
titulo.pack()

subtitulo = tk.Label(titulo_frame,
                    text="Sistema de Reconocimiento de Patrones",
                    font=fuente_subtitulo,
                    bg=COLOR_FRAME,
                    fg="#7F8C8D")
subtitulo.pack(pady=(5,20))

# Frame para los botones principales
botones_frame = tk.Frame(frame, bg=COLOR_FRAME)
botones_frame.pack(pady=20, padx=50)

# Estilo común para botones
button_style = {
    'font': fuente_boton,
    'fg': 'white',
    'relief': tk.FLAT,
    'padx': 20,
    'pady': 10,
    'width': 20,
    'cursor': 'hand2'
}

button_cargar = tk.Button(botones_frame, 
                         text="Cargar Dataset",
                         command=cargar_dataset,
                         bg=COLOR_BOTON_CARGAR,
                         **button_style)
button_cargar.pack(pady=10)

button_entrenar = tk.Button(botones_frame,
                          text="Entrenar Red",
                          command=entrenar_red,
                          bg=COLOR_BOTON_ENTRENAR,
                          **button_style)
button_entrenar.pack(pady=10)

button_simular = tk.Button(botones_frame,
                         text="Simular Red",
                         command=simular_red,
                         bg=COLOR_BOTON_SIMULAR,
                         **button_style)
button_simular.pack(pady=10)

# Frame para información y parámetros
info_frame = tk.Frame(frame, bg=COLOR_FRAME)
info_frame.pack(pady=20, fill=tk.X, padx=50)

# Estilo para las etiquetas de información
info_style = {
    'font': fuente_label,
    'bg': COLOR_FRAME,
    'pady': 5
}

# Información del dataset con mejor formato
label_entradas = tk.Label(info_frame, text="Número de entradas: -", **info_style)
label_entradas.pack()

label_patrones = tk.Label(info_frame, text="Número de patrones: -", **info_style)
label_patrones.pack()

# Frame para parámetros de configuración
config_frame = tk.Frame(frame, bg=COLOR_FRAME)
config_frame.pack(pady=20, fill=tk.X, padx=50)

# Tipo de competencia con mejor diseño
tk.Label(config_frame, text="Tipo de Competencia:", font=fuente_subtitulo, bg=COLOR_FRAME).pack(pady=(10,5))

competencia_frame = tk.Frame(config_frame, bg=COLOR_FRAME)
competencia_frame.pack()

competencia_var = tk.StringVar(value='dura')
tk.Radiobutton(competencia_frame, 
               text="Competencia Dura",
               variable=competencia_var,
               value='dura',
               bg=COLOR_FRAME,
               font=fuente_label).pack(side=tk.LEFT, padx=10)
tk.Radiobutton(competencia_frame,
               text="Competencia Blanda",
               variable=competencia_var,
               value='blanda',
               bg=COLOR_FRAME,
               font=fuente_label).pack(side=tk.LEFT, padx=10)

# Entradas de parámetros con mejor diseño
params_frame = tk.Frame(config_frame, bg=COLOR_FRAME)
params_frame.pack(pady=20)

# Estilo común para entradas
entry_style = {
    'font': fuente_label,
    'width': 15,
    'relief': tk.SOLID,
    'bd': 1
}

# Tasa de aprendizaje
tk.Label(params_frame, text="Tasa de Aprendizaje:", font=fuente_label, bg=COLOR_FRAME).pack()
tasa_aprendizaje_entry = tk.Entry(params_frame, **entry_style)
tasa_aprendizaje_entry.pack(pady=(5,15))
tasa_aprendizaje_entry.insert(0, "0.1")

# Número de iteraciones
tk.Label(params_frame, text="Número de Iteraciones:", font=fuente_label, bg=COLOR_FRAME).pack()
iteraciones_entry = tk.Entry(params_frame, **entry_style)
iteraciones_entry.pack(pady=5)
iteraciones_entry.insert(0, "100")

def on_enter(e, button):
    button['background'] = {
        COLOR_BOTON_CARGAR: '#45a049',
        COLOR_BOTON_ENTRENAR: '#1976D2',
        COLOR_BOTON_SIMULAR: '#F57C00'
    }[button['background']]

def on_leave(e, button):
    button['background'] = {
        '#45a049': COLOR_BOTON_CARGAR,
        '#1976D2': COLOR_BOTON_ENTRENAR,
        '#F57C00': COLOR_BOTON_SIMULAR
    }[button['background']]

# Vincular efectos hover a los botones
for btn in [button_cargar, button_entrenar, button_simular]:
    btn.bind("<Enter>", lambda e, b=btn: on_enter(e, b))
    btn.bind("<Leave>", lambda e, b=btn: on_leave(e, b))

root.mainloop()
