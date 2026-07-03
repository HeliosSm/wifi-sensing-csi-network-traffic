import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# Configuración de variables
PATRON_ARCHIVOS = "linea_base_iter_*.csv"
SUBCARRIER_TEST = 15 # Subportadora aleatoria para graficar el filtrado

def cargar_y_procesar_archivo(ruta_archivo):
    # Leer archivo ignorando líneas con exceso de columnas
    try:
        df = pd.read_csv(ruta_archivo, header=None, on_bad_lines='skip')
    except Exception as e:
        print(f"Error leyendo {ruta_archivo}: {e}")
        return None

    # CORRECCIÓN: Validar que el archivo completo tenga exactamente 131 columnas
    if df.shape[1] != 131:
        print(f" -> {os.path.basename(ruta_archivo)} ignorado: Tiene {df.shape[1]} columnas en lugar de 131.")
        return None

    # Eliminar filas incompletas (NaN) causadas por cortes en el puerto serial
    df = df.dropna()
    
    if df.empty:
        return None

    # Extraer la matriz I/Q (Columnas 3 a 130)
    datos_crudos = df.iloc[:, 3:131].to_numpy(dtype=float)
    
    num_tramas = datos_crudos.shape[0]
    num_subportadoras = 64
    
    amplitud = np.zeros((num_tramas, num_subportadoras))
    amplitud_filtrada = np.zeros((num_tramas, num_subportadoras))
    amplitud_normalizada = np.zeros((num_tramas, num_subportadoras))
    
    # 1. Extracción de Magnitud y 2. Filtrado
    for i in range(num_subportadoras):
        I = datos_crudos[:, 2 * i]
        Q = datos_crudos[:, 2 * i + 1]
        
        # Magnitud bruta
        amp_cruda = np.sqrt(I**2 + Q**2)
        amplitud[:, i] = amp_cruda
        
        # Aplicar filtro de mediana para eliminar Outliers
        amp_filt = medfilt(amp_cruda, kernel_size=7)
        amplitud_filtrada[:, i] = amp_filt
        
        # 3. Normalización Min-Max por subportadora
        amp_min = np.min(amp_filt)
        amp_max = np.max(amp_filt)
        if amp_max - amp_min > 0:
            amplitud_normalizada[:, i] = (amp_filt - amp_min) / (amp_max - amp_min)
        else:
            amplitud_normalizada[:, i] = 0

    return amplitud, amplitud_filtrada, amplitud_normalizada

def main():
    archivos = sorted(glob.glob(PATRON_ARCHIVOS))
    
    if not archivos:
        print("No se encontraron archivos CSV en el directorio actual.")
        return

    print(f"Se detectaron {len(archivos)} archivos. Iniciando procesamiento batch...")
    
    varianzas_por_iteracion = []
    primer_archivo_datos = None
    
    for idx, archivo in enumerate(archivos):
        print(f"Procesando: {os.path.basename(archivo)}...")
        resultados = cargar_y_procesar_archivo(archivo)
        
        if resultados is not None:
            amp_bruta, amp_filt, amp_norm = resultados
            
            # Guardamos el primero para graficar su espectrograma
            if idx == 0:
                primer_archivo_datos = (amp_bruta, amp_filt, amp_norm)
                
            # Calculamos la métrica de estabilidad: Varianza media de toda la matriz
            # En un escenario estático, esta varianza debe ser extremadamente baja
            varianza_media = np.var(amp_norm)
            varianzas_por_iteracion.append(varianza_media)

    print("\nProcesamiento completado. Generando métricas visuales...")
    generar_graficas(primer_archivo_datos, varianzas_por_iteracion, len(archivos))

def generar_graficas(datos_primer_archivo, varianzas, total_archivos):
    amp_bruta, amp_filt, amp_norm = datos_primer_archivo
    tiempo_segundos = np.arange(amp_bruta.shape[0]) * (1/50.0) # Eje X basado en 50Hz
    
    fig = plt.figure(figsize=(15, 10))
    
    # --- Gráfica 1: Efecto del Filtro de Mediana ---
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(tiempo_segundos, amp_bruta[:, SUBCARRIER_TEST], color='red', alpha=0.5, label='Señal Cruda (Con Outliers)')
    ax1.plot(tiempo_segundos, amp_filt[:, SUBCARRIER_TEST], color='blue', linewidth=2, label='Señal Filtrada')
    ax1.set_title(f'Efecto del Filtrado - Subportadora {SUBCARRIER_TEST}')
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Amplitud')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Gráfica 2: Varianza Global (Estabilidad del Experimento) ---
    ax2 = plt.subplot(2, 2, 2)
    iteraciones = np.arange(1, len(varianzas) + 1)
    ax2.bar(iteraciones, varianzas, color='purple', alpha=0.7)
    ax2.set_title('Estabilidad de la Línea Base: Varianza por Iteración')
    ax2.set_xlabel('Iteración (Minuto)')
    ax2.set_ylabel('Varianza Media Normalizada')
    ax2.set_xticks(iteraciones)
    ax2.grid(axis='y', alpha=0.3)
    
    # --- Gráfica 3: Mapa de Calor (Waterfall 2D) ---
    ax3 = plt.subplot(2, 1, 2)
    # Transponemos la matriz para que el tiempo sea el eje X y las subportadoras el eje Y
    im = ax3.imshow(amp_norm.T, aspect='auto', cmap='viridis', 
                    extent=[0, max(tiempo_segundos), 64, 1], interpolation='nearest')
    ax3.set_title('Firma Electromagnética (Espectrograma de Amplitud Normalizada - Iteración 1)')
    ax3.set_xlabel('Tiempo (s)')
    ax3.set_ylabel('Índice de Subportadora (1-64)')
    fig.colorbar(im, ax=ax3, label='Amplitud Normalizada')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()