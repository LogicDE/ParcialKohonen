import numpy as np
import matplotlib.pyplot as plt

class RedKohonen:
    def __init__(self, num_entradas, tipo_competencia, tasa_aprendizaje, num_iteraciones):
        # Inicialización de parámetros
        self.num_entradas = num_entradas
        self.num_neuronas = max(4, num_entradas * 2)  # Mínimo 4 neuronas
        self.tasa_aprendizaje_inicial = tasa_aprendizaje
        self.tasa_aprendizaje = tasa_aprendizaje
        self.num_iteraciones = num_iteraciones
        self.pesos = np.random.uniform(-1, 1, (self.num_entradas, self.num_neuronas))
        self.tipo_competencia = tipo_competencia
        self.dm_values = []  # Para almacenar los valores de Distancia Media (DM)
        
        # Configuración para las gráficas
        self.fig, self.ax = plt.subplots(1, 2, figsize=(15, 5))  

    def entrenar(self, dataset):
        # Entrenamiento de la red para cada iteración
        for iteracion in range(1, self.num_iteraciones + 1):
            distancias_total = []  # Almacenar las distancias para cada patrón
            
            for patron in dataset:
                distancias = self.calcular_distancias(patron)  # Calcular distancias
                neurona_vencedora = np.argmin(distancias)  # Encontrar neurona vencedora
                distancias_total.append(np.mean(distancias))  # Promedio de distancias
                
                # Actualizar pesos de la red
                self.actualizar_pesos(patron, neurona_vencedora, iteracion)
                self.graficar_pesos(iteracion)  # Graficar pesos

            # Actualizar la tasa de aprendizaje
            self.tasa_aprendizaje = self.tasa_aprendizaje_inicial / (1 + iteracion * 0.005)

            # Calcular y graficar el DM
            dm = np.mean(distancias_total)
            self.dm_values.append(dm)
            self.graficar_dm()

            # Evaluar condiciones de parada
            if self.verificar_condiciones_parada(dm, iteracion):
                print(f"Entrenamiento completado en la iteración {iteracion}")
                self.guardar_configuracion()  # Guardar pesos y configuración
                break

    def calcular_distancias(self, patron):
        # Cálculo de distancias entre el patrón y cada neurona
        return np.sqrt(np.sum((self.pesos - patron[:, np.newaxis]) ** 2, axis=0))

    def actualizar_pesos(self, patron, neurona_vencedora, iteracion):
        # Actualización de pesos basada en el tipo de competencia
        if self.tipo_competencia == 'blanda':
            for i in range(self.num_neuronas):
                distancia_neurona = np.linalg.norm(self.pesos[:, neurona_vencedora] - self.pesos[:, i])
                coef_vecindad = 0.2 / np.sqrt(iteracion)  # Coeficiente de vecindad dinámico
                if distancia_neurona < coef_vecindad:
                    self.pesos[:, i] += self.tasa_aprendizaje * (patron - self.pesos[:, i])
        elif self.tipo_competencia == 'dura':
            self.pesos[:, neurona_vencedora] += self.tasa_aprendizaje * (patron - self.pesos[:, neurona_vencedora])

        # Añadir ruido a los pesos en las últimas iteraciones
        if iteracion > self.num_iteraciones * 0.8:
            self.pesos += np.random.normal(0, 0.01, self.pesos.shape)

    def graficar_pesos(self, iteracion):
        # Graficar pesos actuales de la red
        self.ax[0].cla()  # Limpiar el gráfico anterior
        self.ax[0].imshow(self.pesos, aspect='auto', cmap='viridis')
        self.ax[0].set_title(f'Pesos de la red en la iteración {iteracion}')
        self.ax[0].set_xlabel('Neuronas')
        self.ax[0].set_ylabel('Entradas')
        plt.pause(0.01)  # Actualización rápida del gráfico

    def graficar_dm(self):
        # Graficar Distancia Media (DM) frente a iteraciones
        self.ax[1].cla()  # Limpiar gráfico anterior
        self.ax[1].plot(self.dm_values, label="DM", color='blue')
        self.ax[1].set_title("Distancia Media (DM) vs Iteraciones")
        self.ax[1].set_xlabel("Iteraciones")
        self.ax[1].set_ylabel("DM")
        self.ax[1].legend()
        self.ax[1].grid(True)
        plt.pause(0.01)  # Actualización rápida del gráfico

    def verificar_condiciones_parada(self, dm, iteracion):
        # Parar si el DM es menor que 0.1 o si se alcanzan las iteraciones máximas
        return dm < 0.1 or iteracion >= self.num_iteraciones

    def guardar_configuracion(self):
        # Guardar pesos y configuración en archivos
        np.save('pesos_red_kohonen.npy', self.pesos)
        with open('configuracion_red.txt', 'w') as f:
            f.write(f'Número de entradas: {self.num_entradas}\n')
            f.write(f'Tasa de aprendizaje: {self.tasa_aprendizaje}\n')
            f.write(f'Número de iteraciones: {self.num_iteraciones}\n')
        print("Pesos y configuración guardados con éxito.")
