import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt

# Función para graficar los datos ingresados
def graficar():
    try:
        datos = [int(i) for i in entry_datos.get().split(',')]
        plt.plot(datos)
        plt.title('Gráfico de Datos Ingresados')
        plt.show()
    except ValueError:
        messagebox.showerror("Error", "Ingresa datos numéricos separados por comas.")

# Configuración de la ventana principal
root = tk.Tk()
root.title("Interfaz de Gráfico")

label = tk.Label(root, text="Ingresa los datos separados por comas:")
label.pack(padx=10, pady=5)

entry_datos = tk.Entry(root, width=40)
entry_datos.pack(padx=10, pady=5)

button_graficar = tk.Button(root, text="Graficar", command=graficar)
button_graficar.pack(padx=10, pady=10)

root.mainloop()
