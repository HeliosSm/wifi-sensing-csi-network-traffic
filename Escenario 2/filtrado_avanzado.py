import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURACIÓN DEL ENTORNO Y FILTROS
# ==========================================
# Cambia este patrón según el escenario que vayas a analizar
PATRON_ARCHIVOS = 'Escenario 2/movimiento_humano_iter_*.csv' 
FS = 50.0  # Tasa de muestreo empírica (Hz)
FC = 10.0  # Frecuencia de corte Butterworth (Hz)
ORDEN = 3  # Orden del filtro Butterworth

# Diseño del filtro Butterworth digital
nyq = 0.5 * FS
frecuencia_corte_normalizada = FC / nyq
b, a = butter(ORDEN, frecuencia_corte_normalizada, btype='low', analog=False)

def cargar_y_procesar_archivo(ruta_archivo):
    try:
        df = pd.read_csv(ruta_archivo, header=None, on_bad_lines='skip')
    except Exception as e:
        print(f"Error leyendo {ruta_archivo}: {e}")
        return None

    if df.shape[1] != 131:
        print(f" -> {os.path.basename(ruta_archivo)} ignorado: Formato incorrecto.")
        return None

    df = df.dropna()
    if df.empty: return None

    datos_crudos = df.iloc[:, 3:131].to_numpy(dtype=float)
    num_tramas = datos_crudos.shape[0]
    num_subportadoras = 64

    amplitud_butter = np.zeros((num_tramas, num_subportadoras))

    # --- PASO 1 y 2: Extracción y Filtrado Individual por Subportadora ---
    for i in range(num_subportadoras):
        I = datos_crudos[:, 2 * i]
        Q = datos_crudos[:, 2 * i + 1]

        # Magnitud Bruta
        amp_raw = np.sqrt(I**2 + Q**2)

        # Filtro de Mediana (Outliers)
        amp_med = medfilt(amp_raw, kernel_size=7)

        # Filtro Butterworth (Paso bajo a 10 Hz)
        amp_bw = filtfilt(b, a, amp_med)
        amplitud_butter[:, i] = amp_bw

    # --- PASO 3: Análisis de Componentes Principales (PCA) ---
    # 3.1 Estandarización de las 64 subportadoras (Media=0, Varianza=1)
    scaler = StandardScaler()
    amplitud_estandarizada = scaler.fit_transform(amplitud_butter)

    # 3.2 Extracción del Primer Componente Principal (PC1)
    pca = PCA(n_components=1)
    # PC1 es un vector 1D que condensa la mayor varianza de las 64 subportadoras
    senal_maestra_pc1 = pca.fit_transform(amplitud_estandarizada).flatten()

    return amplitud_butter, senal_maestra_pc1

def generar_graficas_pca(datos_primer_archivo):
    amp_butter, senal_maestra_pc1 = datos_primer_archivo
    tiempo_segundos = np.arange(amp_butter.shape[0]) * (1 / FS)

    plt.figure(figsize=(16, 10))

    # --- Gráfica 1: Comportamiento de Subportadoras Individuales (Ejemplos) ---
    ax1 = plt.subplot(2, 1, 1)
    # Seleccionamos 3 subportadoras dispersas para ver su comportamiento
    ax1.plot(tiempo_segundos, amp_butter[:, 10], alpha=0.6, label='Subportadora 10 (Filtrada)')
    ax1.plot(tiempo_segundos, amp_butter[:, 32], alpha=0.6, label='Subportadora 32 (Filtrada)')
    ax1.plot(tiempo_segundos, amp_butter[:, 54], alpha=0.6, label='Subportadora 54 (Filtrada)')
    
    ax1.set_title('Comportamiento Individual de Subportadoras (Tras Filtrado Butterworth a 10 Hz)')
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Amplitud')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Gráfica 2: La Señal Maestra (Primer Componente Principal - PC1) ---
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(tiempo_segundos, senal_maestra_pc1, color='purple', linewidth=2.5, label='PC1 (Señal Maestra)')
    
    # Rellenamos el área bajo la curva para resaltar las variaciones (útil para el paper)
    ax2.fill_between(tiempo_segundos, senal_maestra_pc1, 0, color='purple', alpha=0.1)
    
    ax2.set_title('Primer Componente Principal (PC1) - Representación Global del Canal Wi-Fi')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Amplitud Adimensional (Varianza Principal)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

def main():
    archivos = sorted(glob.glob(PATRON_ARCHIVOS))
    if not archivos:
        print("No se encontraron archivos CSV en el directorio.")
        return

    print(f"Se detectaron {len(archivos)} archivos. Iniciando pipeline con PCA...")
    
    primer_archivo_procesado = None

    for idx, archivo in enumerate(archivos):
        print(f"Procesando y extrayendo PC1: {archivo}...")
        resultado = cargar_y_procesar_archivo(archivo)
        
        if resultado is not None and primer_archivo_procesado is None:
            primer_archivo_procesado = resultado

    if primer_archivo_procesado:
        print("\nExtracción completada. Generando visualización de la Señal Maestra...")
        generar_graficas_pca(primer_archivo_procesado)

if __name__ == "__main__":
    main()