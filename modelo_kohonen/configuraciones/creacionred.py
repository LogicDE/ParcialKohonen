import numpy as np
import matplotlib.pyplot as plt

class RedKohonen:
    def __init__(self, num_entradas, tipo_competencia, tasa_aprendizaje, num_iteraciones):
        self.num_entradas = num_entradas
        # Calcular el número de neuronas (mínimo el doble del número de entradas)
        self.num_neuronas = max(4, num_entradas * 2)  
        self.tasa_aprendizaje = tasa_aprendizaje
        self.num_iteraciones = num_iteraciones
        self.pesos = np.random.uniform(-1, 1, (self.num_entradas, self.num_neuronas))
        self.tipo_competencia = tipo_competencia
        self.dm_values = []  # Para almacenar valores DM

    def entrenar(self, dataset):
        for iteracion in range(1, self.num_iteraciones + 1):
            distancias_total = []
            for patron in dataset:
                # Calcular distancias
                distancias = self.calcular_distancias(patron)
                neurona_vencedora = np.argmin(distancias)
                distancias_total.append(np.mean(distancias))
                self.actualizar_pesos(patron, neurona_vencedora)

            # Actualizar tasa de aprendizaje
            self.tasa_aprendizaje = 1 / iteracion

            # Calcular DM y almacenar
            dm = np.mean(distancias_total)
            self.dm_values.append(dm)

        # Graficar comportamiento de los pesos y DM al final del entrenamiento
        self.graficar_pesos_final()
        self.graficar_dm_final()

        # Verificar condiciones de parada
        if self.verificar_condiciones_parada(dm, iteracion):
            print(f"Entrenamiento completado en iteración {iteracion}")

    def calcular_distancias(self, patron):
        return np.sqrt(np.sum((self.pesos - patron[:, np.newaxis]) ** 2, axis=0))

    def actualizar_pesos(self, patron, neurona_vencedora):
        if self.tipo_competencia == 'blanda':
            for i in range(self.num_neuronas):
                distancia_neurona = np.linalg.norm(self.pesos[:, neurona_vencedora] - self.pesos[:, i])
                coef_vecindad = 0.2  # Este valor puede ser ajustado
                if distancia_neurona < coef_vecindad:
                    self.pesos[:, i] += self.tasa_aprendizaje * (patron - self.pesos[:, i])
        elif self.tipo_competencia == 'dura':
            self.pesos[:, neurona_vencedora] += self.tasa_aprendizaje * (patron - self.pesos[:, neurona_vencedora])

    def graficar_pesos_final(self):
        plt.figure(figsize=(12, 6))
        plt.imshow(self.pesos, aspect='auto', cmap='viridis')
        plt.title('Pesos de la Red al Final del Entrenamiento', fontsize=16)
        plt.colorbar(label='Valor de Peso')
        plt.xlabel('Neuronas', fontsize=12)
        plt.ylabel('Entradas', fontsize=12)
        plt.grid(False)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.show()

    def graficar_dm_final(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.dm_values, label="DM", color='blue', marker='o', linestyle='-', markersize=5)
        plt.title("Distancia Media (DM) vs Iteraciones", fontsize=16)
        plt.xlabel("Iteraciones", fontsize=12)
        plt.ylabel("DM", fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.show()
    
    def verificar_condiciones_parada(self, dm, iteracion):
        if dm < 0.1 or iteracion >= self.num_iteraciones:
            return True
        return False
    
    def guardar_configuracion(self):
    # Guardar pesos y configuración en un archivo (por ejemplo, usando np.save)
        np.save('pesos_red_kohonen.npy', self.pesos)
        with open('configuracion_red.txt', 'w') as f:
            f.write(f'Número de entradas: {self.num_entradas}\n')
            f.write(f'Tasa de aprendizaje: {self.tasa_aprendizaje}\n')
            f.write(f'Número de iteraciones: {self.num_iteraciones}\n')




