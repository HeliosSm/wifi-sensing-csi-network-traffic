import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# Configuración de variables
TASAS_MBPS = [2, 5, 10]
SUBCARRIER_TEST = 15 # Subportadora para graficar el filtrado

def cargar_y_procesar_archivo(ruta_archivo):
    try:
        df = pd.read_csv(ruta_archivo, header=None, on_bad_lines='skip')
    except Exception as e:
        print(f"Error leyendo {ruta_archivo}: {e}")
        return None

    if df.shape[1] != 131:
        return None

    df = df.dropna()
    
    if df.empty:
        return None

    datos_crudos = df.iloc[:, 3:131].to_numpy(dtype=float)
    
    num_tramas = datos_crudos.shape[0]
    num_subportadoras = 64
    
    amplitud = np.zeros((num_tramas, num_subportadoras))
    amplitud_filtrada = np.zeros((num_tramas, num_subportadoras))
    amplitud_normalizada = np.zeros((num_tramas, num_subportadoras))
    
    for i in range(num_subportadoras):
        I = datos_crudos[:, 2 * i]
        Q = datos_crudos[:, 2 * i + 1]
        
        amp_cruda = np.sqrt(I**2 + Q**2)
        amplitud[:, i] = amp_cruda
        
        amp_filt = medfilt(amp_cruda, kernel_size=7)
        amplitud_filtrada[:, i] = amp_filt
        
        amp_min = np.min(amp_filt)
        amp_max = np.max(amp_filt)
        if amp_max - amp_min > 0:
            amplitud_normalizada[:, i] = (amp_filt - amp_min) / (amp_max - amp_min)
        else:
            amplitud_normalizada[:, i] = 0

    return amplitud, amplitud_filtrada, amplitud_normalizada

def main():
    varianzas_por_tasa = {tasa: [] for tasa in TASAS_MBPS}
    primer_archivo_datos_10mbps = None # Guardaremos los datos de 10Mbps para el espectrograma
    
    for tasa in TASAS_MBPS:
        patron = f"trafico_udp_{tasa}mbps_iter_*.csv"
        archivos = sorted(glob.glob(patron))
        
        if not archivos:
            print(f"No se encontraron archivos para la tasa de {tasa} Mbps.")
            continue
            
        print(f"Procesando {len(archivos)} archivos para {tasa} Mbps...")
        
        for idx, archivo in enumerate(archivos):
            resultados = cargar_y_procesar_archivo(archivo)
            
            if resultados is not None:
                amp_bruta, amp_filt, amp_norm = resultados
                
                # Guardamos una muestra representativa (ej. la primera de 10Mbps)
                if tasa == 10 and primer_archivo_datos_10mbps is None:
                    primer_archivo_datos_10mbps = (amp_bruta, amp_filt, amp_norm)
                    
                # Calcular varianza media de la iteración
                varianza_media = np.var(amp_norm)
                varianzas_por_tasa[tasa].append(varianza_media)

    print("\nGenerando gráficas comparativas...")
    generar_graficas_comparativas(varianzas_por_tasa, primer_archivo_datos_10mbps)

def generar_graficas_comparativas(varianzas_dict, datos_espectrograma):
    fig = plt.figure(figsize=(16, 12))
    
    # --- Gráfica 1: Comparativa de Varianzas (Barras Agrupadas) ---
    ax1 = plt.subplot(2, 1, 1)
    
    tasas = list(varianzas_dict.keys())
    num_iteraciones = max(len(v) for v in varianzas_dict.values())
    x = np.arange(1, num_iteraciones + 1)
    ancho_barra = 0.25
    
    colores = {2: 'lightblue', 5: 'royalblue', 10: 'navy'}
    
    for i, tasa in enumerate(tasas):
        varianzas = varianzas_dict[tasa]
        # Rellenar con ceros si faltan iteraciones para que no falle el gráfico
        if len(varianzas) < num_iteraciones:
             varianzas.extend([0] * (num_iteraciones - len(varianzas)))
             
        posiciones_x = x + (i - 1) * ancho_barra
        ax1.bar(posiciones_x, varianzas, ancho_barra, label=f'Tráfico UDP {tasa} Mbps', color=colores[tasa], alpha=0.8)

    ax1.set_title('Impacto del Tráfico de Red en la Varianza del CSI (Escenario 3)', fontsize=14)
    ax1.set_xlabel('Iteración (Minuto)', fontsize=12)
    ax1.set_ylabel('Varianza Media Normalizada', fontsize=12)
    ax1.set_xticks(x)
    
    # Añadir línea de referencia teórica de la Línea Base (~0.05)
    ax1.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='Umbral Línea Base Aprox.')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # --- Gráfica 2: Espectrograma de la Tasa más alta (10 Mbps) ---
    if datos_espectrograma is not None:
        ax2 = plt.subplot(2, 1, 2)
        amp_bruta, amp_filt, amp_norm = datos_espectrograma
        
        # Ajustamos el eje de tiempo a los paquetes reales recibidos, no a los teóricos de 50Hz,
        # ya que la inyección de tráfico degrada la tasa de muestreo.
        tiempo_segundos = np.linspace(0, 60, amp_norm.shape[0]) 
        
        im = ax2.imshow(amp_norm.T, aspect='auto', cmap='viridis', 
                        extent=[0, 60, 64, 1], interpolation='nearest')
        ax2.set_title('Firma Electromagnética (Espectrograma de Tráfico UDP a 10 Mbps)', fontsize=14)
        ax2.set_xlabel('Tiempo (s)', fontsize=12)
        ax2.set_ylabel('Índice de Subportadora (1-64)', fontsize=12)
        fig.colorbar(im, ax=ax2, label='Amplitud Normalizada')
    else:
         print("No hay datos de 10 Mbps para generar el espectrograma.")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()