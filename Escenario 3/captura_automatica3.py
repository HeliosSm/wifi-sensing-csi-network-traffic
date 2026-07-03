import serial
import csv
import time
import socket
import threading
from datetime import datetime
import sys

# Configuración Serial
PUERTO = 'COM4' 
BAUDRATE = 921600

# Configuración del Experimento
TASAS_MBPS = [2, 5, 10]
ITERACIONES_POR_TASA = 15
DURACION_SEGUNDOS = 60

# Configuración de Red
IP_DESTINO = '192.168.4.1'
PUERTO_UDP = 9999

inyector_activo = False

def inyector_udp(tasa_mbps):
    """Genera tráfico UDP mediante ráfagas para no asfixiar el CPU de la PC."""
    global inyector_activo
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tamano_payload = 1024
    payload = b'X' * tamano_payload
    
    # Cálculos de ancho de banda
    bytes_por_segundo = (tasa_mbps * 1_000_000) / 8
    paquetes_por_segundo = bytes_por_segundo / tamano_payload
    
    # Técnica de Ráfaga: Disparamos cada 50ms para liberar el CPU
    intervalo_rafaga = 0.05 
    paquetes_por_rafaga = int(paquetes_por_segundo * intervalo_rafaga)
    
    print(f"   [RECLAMO DE CANAL] Inyectando {tasa_mbps} Mbps (Ráfagas de {paquetes_por_rafaga} pkts cada 50ms)...")
    
    while inyector_activo:
        t_inicio = time.time()
        
        # Enviar la ráfaga completa de golpe
        for _ in range(paquetes_por_rafaga):
            try:
                sock.sendto(payload, (IP_DESTINO, PUERTO_UDP))
            except Exception:
                pass 
                
        # Calcular cuánto tardó la inyección y dormir el resto del tiempo
        tiempo_transcurrido = time.time() - t_inicio
        tiempo_a_dormir = intervalo_rafaga - tiempo_transcurrido
        
        # time.sleep libera el procesador para que el puerto serial pueda leer el CSI
        if tiempo_a_dormir > 0:
            time.sleep(tiempo_a_dormir)
            
    sock.close()

def ejecutar_escenario_3():
    global inyector_activo
    print("==================================================")
    print("  INICIANDO AUTOMATIZACIÓN: ESCENARIO 3 (TRÁFICO UDP) ")
    print("==================================================")
    print("Verifica que el Wi-Fi de tu PC esté conectado a CSI_TEST_NET.")
    print(f"Tiempo estimado total: {len(TASAS_MBPS) * ITERACIONES_POR_TASA} minutos.")
    print("Tienes 20 segundos para salir del cuarto y cerrar la puerta...")
    
    for i in range(20, 0, -1):
        sys.stdout.write(f"\rIniciando en {i} segundos...   ")
        sys.stdout.flush()
        time.sleep(1)
        
    print("\n\n¡Iniciando captura desatendida! Entorno estático requerido.\n")
    
    try:
        ser = serial.Serial(PUERTO, BAUDRATE)
        ser.reset_input_buffer() 
        
        for tasa in TASAS_MBPS:
            print("--------------------------------------------------")
            print(f" PREPARANDO ESCENARIO DE CARGA: {tasa} Mbps ")
            print("--------------------------------------------------")
            
            inyector_activo = True
            hilo_trafico = threading.Thread(target=inyector_udp, args=(tasa,))
            hilo_trafico.start()
            
            time.sleep(2) # Estabilizar canal
            
            for iteracion in range(1, ITERACIONES_POR_TASA + 1):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"trafico_udp_{tasa}mbps_iter_{iteracion}_{timestamp}.csv"
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {tasa}Mbps - Iter {iteracion}/{ITERACIONES_POR_TASA} -> Guardando...")
                
                with open(nombre_archivo, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    inicio_tiempo = time.time()
                    paquetes_guardados = 0
                    
                    while (time.time() - inicio_tiempo) < DURACION_SEGUNDOS:
                        if ser.in_waiting > 0:
                            linea = ser.readline().decode('utf-8', errors='ignore').strip()
                            if linea.startswith("CSI_DATA"):
                                datos = linea.split(',')
                                writer.writerow(datos)
                                paquetes_guardados += 1
                                
                print(f"   -> Completado: {paquetes_guardados} tramas CSI capturadas.\n")
            
            inyector_activo = False
            hilo_trafico.join()
            print(f">>> Fase de {tasa} Mbps finalizada. Descansando canal 5 segundos...\n")
            time.sleep(5)
            ser.reset_input_buffer()
            
        print("==================================================")
        print(" ¡EXPERIMENTO FINALIZADO CON ÉXITO!               ")
        print(" Ya puedes entrar al cuarto.                      ")
        print("==================================================")
        ser.close()
        
    except serial.SerialException:
        print(f"\n[ERROR CRÍTICO] No se pudo abrir {PUERTO}.")
    except KeyboardInterrupt:
        inyector_activo = False
        print("\n[CANCELADO] Captura detenida manualmente.")
    except Exception as e:
        inyector_activo = False
        print(f"\n[ERROR INESPERADO] {e}")

if __name__ == "__main__":
    ejecutar_escenario_3()