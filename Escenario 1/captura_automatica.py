import serial
import csv
import time
from datetime import datetime
import sys

PUERTO = 'COM4' 
BAUDRATE = 921600
ITERACIONES = 15
DURACION_SEGUNDOS = 60

def capturar_automatizado():
    print("==================================================")
    print("  INICIANDO AUTOMATIZACIÓN: ESCENARIO LÍNEA BASE  ")
    print("==================================================")
    print("Tienes 20 segundos para salir del cuarto y cerrar la puerta...")
    
    for i in range(20, 0, -1):
        sys.stdout.write(f"\rIniciando captura en {i} segundos...   ")
        sys.stdout.flush()
        time.sleep(1)
        
    print("\n\n¡Iniciando capturas! El entorno debe permanecer estático.\n")
    
    try:
        ser = serial.Serial(PUERTO, BAUDRATE)
        ser.reset_input_buffer() 
        
        for iteracion in range(1, ITERACIONES + 1):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo = f"linea_base_iter_{iteracion}_{timestamp}.csv"
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Iteración {iteracion}/{ITERACIONES} -> Guardando en {nombre_archivo}")
            
            with open(nombre_archivo, mode='w', newline='') as file:
                writer = csv.writer(file)
                inicio_tiempo = time.time()
                paquetes_guardados = 0
                
                while (time.time() - inicio_tiempo) < DURACION_SEGUNDOS:
                    if ser.in_waiting > 0:
                        linea = ser.readline().decode('utf-8', errors='ignore').strip()
                        
                        # Monitoreo de estado de red
                        if linea.startswith("WIFI_STATUS"):
                            estado = linea.split(',')[1]
                            print(f"\n   >>> [ALERTA DE RED] El ESP32 RX está: {estado} <<<\n")
                        
                        # Captura de datos
                        elif linea.startswith("CSI_DATA"):
                            datos = linea.split(',')
                            writer.writerow(datos)
                            paquetes_guardados += 1
                            
            print(f"   -> Completado: {paquetes_guardados} tramas CSI capturadas.\n")
            
        print("==================================================")
        print(" ¡EXPERIMENTO FINALIZADO CON ÉXITO!               ")
        print(" Ya puedes entrar al cuarto.                      ")
        print("==================================================")
        ser.close()
        
    except serial.SerialException:
        print(f"\n[ERROR CRÍTICO] No se pudo abrir {PUERTO}.")
        print("Asegúrate de que el monitor serial de ESP-IDF esté CERRADO.")
    except KeyboardInterrupt:
        print("\n[CANCELADO] Captura detenida manualmente.")
    except Exception as e:
        print(f"\n[ERROR INESPERADO] {e}")

if __name__ == "__main__":
    capturar_automatizado()