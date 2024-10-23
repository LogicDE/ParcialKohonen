import numpy as np
import matplotlib.pyplot as plt

class RedKohonen:
    def __init__(self, num_entradas, tipo_competencia, tasa_aprendizaje, num_iteraciones):
        self.num_entradas = num_entradas
        self.num_neuronas = max(4, num_entradas * 6)  # Al menos 4 neuronas
        self.tasa_aprendizaje_inicial = tasa_aprendizaje
        self.tasa_aprendizaje = tasa_aprendizaje
        self.num_iteraciones = num_iteraciones
        self.pesos = np.random.uniform(-1, 1, (self.num_entradas, self.num_neuronas))
        self.tipo_competencia = tipo_competencia
        self.dm_values = []  # Valores de Distancia Media (DM) para cada iteración

    def entrenar(self, dataset):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # Dos gráficas lado a lado

        for iteracion in range(1, self.num_iteraciones + 1):
            distancias_total = []  # Distancias para cada patrón

            for patron in dataset:
                distancias = self.calcular_distancias(patron)
                neurona_vencedora = np.argmin(distancias)  # Neurona con menor distancia
                distancias_total.append(np.mean(distancias))  # Guardar distancia promedio

                # Actualizar los pesos
                self.actualizar_pesos(patron, neurona_vencedora, iteracion)

            # Actualizar la tasa de aprendizaje
            self.tasa_aprendizaje = self.tasa_aprendizaje_inicial / (1 + iteracion * 0.005)

            # Guardar y graficar DM
            dm = np.mean(distancias_total)
            self.dm_values.append(dm)
            self.actualizar_graficos(axs, iteracion)

            if self.verificar_condiciones_parada(dm, iteracion):
                print(f"Entrenamiento completado en iteración {iteracion}")
                break

        plt.ioff()
        plt.show()

    def calcular_distancias(self, patron):
        return np.sqrt(np.sum((self.pesos - patron[:, np.newaxis]) ** 2, axis=0))

    def actualizar_pesos(self, patron, neurona_vencedora, iteracion):
        if self.tipo_competencia == 'blanda':
            coef_vecindad = 0.2 / np.sqrt(iteracion)  # Coeficiente de vecindad dinámico
            for i in range(self.num_neuronas):
                distancia_neurona = np.linalg.norm(self.pesos[:, neurona_vencedora] - self.pesos[:, i])
                if distancia_neurona < coef_vecindad:
                    self.pesos[:, i] += self.tasa_aprendizaje * (patron - self.pesos[:, i])
        else:  # Competencia dura
            self.pesos[:, neurona_vencedora] += self.tasa_aprendizaje * (patron - self.pesos[:, neurona_vencedora])

    def actualizar_graficos(self, axs, iteracion):
        axs[0].cla()
        axs[0].imshow(self.pesos, aspect='auto', cmap='viridis')
        axs[0].set_title(f'Pesos en la iteración {iteracion}')
        axs[0].set_xlabel('Neuronas')
        axs[0].set_ylabel('Entradas')

        axs[1].cla()
        axs[1].plot(self.dm_values, label="DM")
        axs[1].set_title("Distancia Media (DM) vs Iteraciones")
        axs[1].set_xlabel("Iteraciones")
        axs[1].set_ylabel("DM")
        axs[1].legend()
        axs[1].grid(True)

        plt.pause(0.1)  # Actualización de gráficos

    def verificar_condiciones_parada(self, dm, iteracion):
        return dm < 0.1 or iteracion >= self.num_iteraciones
    
    def cargar_pesos(self, pesos):
        self.pesos = pesos  # Cargar pesos óptimos desde un archivo o variable

    def simular(self, patron):
        distancias = self.calcular_distancias(patron)  # Calcular distancias
        neurona_vencedora = np.argmin(distancias)  # Neurona con menor distancia
        return self.pesos[:, neurona_vencedora]  # Retornar los pesos de la neurona vencedora
    
