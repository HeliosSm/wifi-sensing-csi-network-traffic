import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt, spectrogram
from sklearn.decomposition import PCA

# ==========================================
# CONFIGURACIÓN DEL ENTORNO
# ==========================================
PATRON_ARCHIVOS = 'trafico_udp_10mbps_iter_*.csv' # Cambiar según escenario
FS = 50.0  
FC = 10.0  
ORDEN = 3  

# --- LÍMITES DE COLOR PARA EL ESPECTROGRAMA (ESCALA LOGARÍTMICA dB) ---
# Primero corre el escenario de Movimiento con VMAX_DB = None para ver el máximo.
# Luego, define ese número aquí para que todos los gráficos usen la misma paleta.
VMIN_DB = -20.0  # Piso de ruido en dB
VMAX_DB = None   # Ej: Cambiar a 30.0 después de ver el movimiento humano

# Diseño del filtro Butterworth
nyq = 0.5 * FS
b, a = butter(ORDEN, FC / nyq, btype='low', analog=False)

def cargar_y_procesar_archivo(ruta_archivo):
    try:
        df = pd.read_csv(ruta_archivo, header=None, on_bad_lines='skip')
    except Exception: return None

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

    # PCA sin escalar (Preservación de energía)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(amplitud_butter).flatten()

    return pc1

def generar_espectrograma(pc1, nombre_archivo):
    tiempo_total = np.arange(len(pc1)) * (1 / FS)

    # ==========================================
    # CÁLCULO DEL ESPECTROGRAMA (STFT)
    # ==========================================
    # nperseg = 100 (Ventana de 2 segundos a 50Hz para capturar bien la baja frecuencia)
    # noverlap = 95 (95% de solapamiento para alta resolución gráfica)
    # nfft = 512 (Para interpolar suavemente el eje Y de frecuencias)
    frecuencias, tiempos_stft, Sxx = spectrogram(
        pc1, fs=FS, window='hann', nperseg=100, noverlap=95, nfft=512
    )

    # Convertimos la potencia a escala logarítmica (Decibelios) para visualizar
    # pequeñas variaciones morfológicas sin perder los picos grandes
    Sxx_db = 10 * np.log10(Sxx + 1e-10) # 1e-10 evita log(0)
    
    max_db_real = np.max(Sxx_db)
    print(f"\n[{nombre_archivo}] -> Energía Máxima en el Espectrograma: {max_db_real:.2f} dB")

    # ==========================================
    # VISUALIZACIÓN
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 2]})

    # 1. Gráfica en el Dominio del Tiempo (Para referencia)
    ax1.plot(tiempo_total, pc1, color='purple', linewidth=1.5)
    ax1.set_title(f'Señal Maestra PC1 en el Dominio del Tiempo - {nombre_archivo}')
    ax1.set_ylabel('Amplitud Absoluta')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, tiempo_total[-1])

    # 2. Espectrograma (Análisis Morfológico)
    vmax_plot = VMAX_DB if VMAX_DB is not None else max_db_real
    
    # Usamos pcolormesh para generar el mapa de calor
    c = ax2.pcolormesh(tiempos_stft, frecuencias, Sxx_db, 
                       shading='gouraud', cmap='jet', vmin=VMIN_DB, vmax=vmax_plot)
    
    ax2.set_title('Espectrograma Temporal-Frecuencial (STFT)')
    ax2.set_ylabel('Frecuencia (Hz)')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylim(0, 10) # Solo nos interesa de 0 a 10 Hz
    ax2.set_xlim(0, tiempo_total[-1])
    
    # Barra de colores
    cbar = fig.colorbar(c, ax=ax2, label='Densidad de Potencia (dB)')

    plt.tight_layout()
    plt.show()

def main():
    archivos = sorted(glob.glob(PATRON_ARCHIVOS))
    if not archivos:
        print("No se encontraron archivos.")
        return

    print("Generando Espectrograma...")
    pc1 = cargar_y_procesar_archivo(archivos[0]) 
    
    if pc1 is not None:
        generar_espectrograma(pc1, os.path.basename(archivos[0]))

if __name__ == "__main__":
    main()