import numpy as np
import matplotlib.pyplot as plt

class RedKohonen:
    def __init__(self, num_entradas, tipo_competencia, tasa_aprendizaje, num_iteraciones):
        self.num_entradas = num_entradas
        self.num_neuronas = max(4, num_entradas * 8)  # Aumentamos el número de neuronas
        self.tasa_aprendizaje_inicial = tasa_aprendizaje
        self.tasa_aprendizaje = tasa_aprendizaje
        self.num_iteraciones = num_iteraciones
        self.pesos = np.random.uniform(-0.5, 0.5, (self.num_entradas, self.num_neuronas))  # Inicialización más acotada
        self.tipo_competencia = tipo_competencia
        self.dm_values = []
        self.mejor_dm = float('inf')
        self.callback = None  # Para actualizar la interfaz

    def set_callback(self, callback_fn):
        self.callback = callback_fn

    def entrenar(self, dataset):
        dataset = np.array(dataset)
        # Normalizar dataset
        dataset = (dataset - dataset.mean()) / dataset.std()
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        plt.ion()

        for iteracion in range(1, self.num_iteraciones + 1):
            distancias_total = []
            np.random.shuffle(dataset)  # Mezclar datos en cada iteración

            for patron in dataset:
                distancias = self.calcular_distancias(patron)
                neurona_vencedora = np.argmin(distancias)
                distancias_total.append(np.min(distancias))  # Solo la menor distancia

                self.actualizar_pesos(patron, neurona_vencedora, iteracion)

            # Actualización adaptativa de la tasa de aprendizaje
            self.tasa_aprendizaje = self.tasa_aprendizaje_inicial * np.exp(-0.001 * iteracion)

            dm = np.mean(distancias_total)
            self.dm_values.append(dm)
            
            if dm < self.mejor_dm:
                self.mejor_dm = dm

            self.actualizar_graficos(axs, iteracion, dm)
            
            # Actualizar interfaz si hay callback
            if self.callback:
                self.callback(f"DM actual: {dm:.6f}\nMejor DM: {self.mejor_dm:.6f}")

            if self.verificar_condiciones_parada(dm, iteracion):
                print(f"Entrenamiento completado en iteración {iteracion}")
                print(f"DM final: {dm:.6f}")
                break

        plt.ioff()
        plt.show()

    def calcular_distancias(self, patron):
        return np.sqrt(np.sum((self.pesos - patron[:, np.newaxis]) ** 2, axis=0))

    def actualizar_pesos(self, patron, neurona_vencedora, iteracion):
        if self.tipo_competencia == 'blanda':
            radio = self.num_neuronas * np.exp(-iteracion / self.num_iteraciones)
            for i in range(self.num_neuronas):
                distancia = abs(i - neurona_vencedora)
                if distancia <= radio:
                    influencia = np.exp(-distancia**2 / (2 * radio**2))
                    self.pesos[:, i] += self.tasa_aprendizaje * influencia * (patron - self.pesos[:, i])
        else:
            self.pesos[:, neurona_vencedora] += self.tasa_aprendizaje * (patron - self.pesos[:, neurona_vencedora])

    def actualizar_graficos(self, axs, iteracion, dm):
        axs[0].cla()
        axs[0].imshow(self.pesos, aspect='auto', cmap='viridis')
        axs[0].set_title(f'Pesos en la iteración {iteracion} (DM: {dm:.6f})')
        axs[0].set_xlabel('Neuronas')
        axs[0].set_ylabel('Entradas')

        axs[1].cla()
        axs[1].plot(self.dm_values, label="DM")
        axs[1].set_title("Distancia Media (DM) vs Iteraciones")
        axs[1].set_xlabel("Iteraciones")
        axs[1].set_ylabel("DM")
        axs[1].legend()
        axs[1].grid(True)

        plt.pause(0.1)  # Actualización más rápida de gráficos

    def verificar_condiciones_parada(self, dm, iteracion):
        if len(self.dm_values) > 10:
            ultimos_dm = self.dm_values[-10:]
            variacion = np.std(ultimos_dm)
            return dm < 0.001 or (variacion < 0.0001 and dm < 0.01) or iteracion >= self.num_iteraciones
        return False
    
    def cargar_pesos(self, pesos):
        self.pesos = pesos  # Cargar pesos óptimos desde un archivo o variable
    
    def simular(self, patron):
        distancias = self.calcular_distancias(patron)  # Calcular distancias
        neurona_vencedora = np.argmin(distancias)  # Neurona con menor distancia
        return self.pesos[:, neurona_vencedora]  # Retornar los pesos de la neurona vencedora
    
    def comparar_pesos(self, dataset):
        dataset = np.array(dataset)
        for patron in dataset:
            distancias = self.calcular_distancias(patron)  # Calcular distancias
            neurona_vencedora = np.argmin(distancias)  # Neurona con menor distancia
            distancia_minima = distancias[neurona_vencedora]
            
            print(f"Patrón: {patron}, Neurona Vencedora: {neurona_vencedora}, Distancia: {distancia_minima:.6f}")