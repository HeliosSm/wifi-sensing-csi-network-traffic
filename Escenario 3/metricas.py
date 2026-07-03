import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt, welch
from sklearn.decomposition import PCA

# ==========================================
# CONFIGURACIÓN DEL ENTORNO Y PARÁMETROS
# ==========================================
PATRON_ARCHIVOS = 'Escenario 3/trafico_udp_10mbps_iter_*.csv' # CAMBIA ESTO PARA CADA ESCENARIO
FS = 50.0  
FC = 10.0  
ORDEN = 3  
VENTANA_VARIANZA = 50 

# --- LÍMITES VISUALES PARA LOS EJES Y ---
# Nota: La primera vez que corras el "Movimiento", mira en la consola 
# cuál es el valor máximo. Luego, pon esos valores aquí para fijar la escala 
# y que todas las gráficas sean visualmente comparables.
LIMITE_Y_PSD = None      # Ej: Cambiar a 1500 después de ver el máximo de movimiento
LIMITE_Y_VARIANZA = None # Ej: Cambiar a 500 después de ver el máximo de movimiento

# Diseño del filtro Butterworth
nyq = 0.5 * FS
b, a = butter(ORDEN, FC / nyq, btype='low', analog=False)

def cargar_y_procesar_archivo(ruta_archivo):
    try:
        df = pd.read_csv(ruta_archivo, header=None, on_bad_lines='skip')
    except Exception as e: return None

    if df.shape[1] != 131: return None
    df = df.dropna()
    if df.empty: return None

    datos_crudos = df.iloc[:, 3:131].to_numpy(dtype=float)
    num_tramas = datos_crudos.shape[0]
    num_subportadoras = 64
    amplitud_butter = np.zeros((num_tramas, num_subportadoras))

    for i in range(num_subportadoras):
        I, Q = datos_crudos[:, 2 * i], datos_crudos[:, 2 * i + 1]
        amp_raw = np.sqrt(I**2 + Q**2)
        amp_med = medfilt(amp_raw, kernel_size=7)
        amplitud_butter[:, i] = filtfilt(b, a, amp_med)

    # --- PCA CORREGIDO (Preservando la energía real) ---
    pca = PCA(n_components=1)
    # Ya no usamos StandardScaler. Pasamos la matriz directamente.
    pc1 = pca.fit_transform(amplitud_butter).flatten()

    # --- EXTRACCIÓN DE MÉTRICAS PHY ---
    nperseg = min(len(pc1), 128)
    frecuencias, psd = welch(pc1, fs=FS, nperseg=nperseg)
    varianza_movil = pd.Series(pc1).rolling(window=VENTANA_VARIANZA).var().fillna(0).to_numpy()

    return pc1, frecuencias, psd, varianza_movil

def generar_dashboard_metricas(datos_archivo, nombre_archivo):
    pc1, frecuencias, psd, varianza_movil = datos_archivo
    tiempo = np.arange(len(pc1)) * (1 / FS)
    
    # --- IMPRESIÓN EN CONSOLA DE LAS MÉTRICAS REALES ---
    max_psd = np.max(psd)
    max_var = np.max(varianza_movil)
    print(f"\n[{nombre_archivo}] -> MÉTRICAS CLAVE PARA EL PAPER:")
    print(f" -> Energía Máxima PSD     : {max_psd:.4f} V^2/Hz")
    print(f" -> Varianza Móvil Máxima  : {max_var:.4f}")
    print("-" * 50)

    plt.figure(figsize=(15, 12))

    # Gráfica 1: PC1
    plt.subplot(3, 1, 1)
    plt.plot(tiempo, pc1, color='purple')
    plt.title('1. Dominio del Tiempo: Componente Principal 1 (Energía Real Absoluta)')
    plt.ylabel('Amplitud')
    plt.grid(True, alpha=0.3)

    # Gráfica 2: PSD
    plt.subplot(3, 1, 2)
    plt.plot(frecuencias, psd, color='darkorange', linewidth=2)
    plt.title('2. Densidad Espectral de Potencia (Método de Welch)')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('PSD [V**2/Hz]')
    plt.xlim(0, 10)
    if LIMITE_Y_PSD is not None:
        plt.ylim(0, LIMITE_Y_PSD) # Bloqueo del Eje Y
    plt.axvspan(0.5, 3.0, color='yellow', alpha=0.2, label='Banda Cinemática (0.5 - 3 Hz)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Gráfica 3: Varianza Móvil
    plt.subplot(3, 1, 3)
    plt.plot(tiempo, varianza_movil, color='green', linewidth=2)
    plt.title(f'3. Varianza Móvil (Ventana = {VENTANA_VARIANZA} muestras / 1 seg)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Varianza')
    if LIMITE_Y_VARIANZA is not None:
        plt.ylim(0, LIMITE_Y_VARIANZA) # Bloqueo del Eje Y
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    archivos = sorted(glob.glob(PATRON_ARCHIVOS))
    if not archivos:
        print("No se encontraron archivos.")
        return

    print("Calculando Métricas Físicas Absolutas...")
    # Procesamos el primer archivo y pasamos su nombre para la consola
    resultado = cargar_y_procesar_archivo(archivos[0]) 
    
    if resultado:
        generar_dashboard_metricas(resultado, os.path.basename(archivos[0]))

if __name__ == "__main__":
    main()