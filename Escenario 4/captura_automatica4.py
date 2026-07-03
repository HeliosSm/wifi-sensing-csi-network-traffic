import serial
import csv
import time
import socket
import threading
import random
from datetime import datetime
import sys

# ==========================================================
#  ESCENARIO 4: MOVIMIENTO HUMANO + TRAFICO DE RED ALEATORIO
#  (fusion del Escenario 2 y el Escenario 3)
# ==========================================================
# La persona camina de forma continua (como en el Escenario 2) MIENTRAS
# un hilo inyecta trafico UDP a una tasa que se re-sortea aleatoriamente
# cada INTERVALO_RESORTEO segundos (a diferencia del Escenario 3 que usaba
# tasas fijas 2/5/10 Mbps). La tasa instantanea se registra en un log
# paralelo con marca de tiempo para poder etiquetar el CSI a posteriori.

# --- Configuracion Serial (igual que escenarios previos) ---
PUERTO = 'COM4'
BAUDRATE = 921600

# --- Configuracion del Experimento ---
ITERACIONES = 20              # Igual que el Escenario 2 (movimiento)
DURACION_SEGUNDOS = 60        # Ventana de captura por iteracion

# --- Configuracion de Red (igual topologia que el Escenario 3) ---
IP_DESTINO = '192.168.4.1'
PUERTO_UDP = 9999

# --- Aleatorizacion del trafico (realismo) ---
MBPS_MIN = 2.0                # Limite inferior del rango sondeado en E3
MBPS_MAX = 10.0               # Limite superior del rango sondeado en E3
INTERVALO_RESORTEO = 5.0      # Cada cuantos segundos se re-sortea la tasa
PROB_RAFAGA = 0.15            # Prob. de una rafaga puntual (pico de congestion)
MBPS_RAFAGA = 12.0            # Tasa de la rafaga (estresa el canal)

inyector_activo = False
log_tasas = []                # [(timestamp_relativo, mbps)] para etiquetado


def inyector_udp_aleatorio(t0_ref):
    """Inyecta UDP a una tasa que cambia aleatoriamente en el tiempo.

    Reproduce el realismo de una red domestica: la carga no es constante,
    sino que fluctua. Se registra cada cambio de tasa para el etiquetado.
    """
    global inyector_activo, log_tasas
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tamano_payload = 1024
    payload = b'X' * tamano_payload
    intervalo_rafaga = 0.05  # 50 ms, identico al Escenario 3

    t_ultimo_sorteo = 0.0
    tasa_mbps = random.uniform(MBPS_MIN, MBPS_MAX)

    while inyector_activo:
        ahora = time.time() - t0_ref

        # --- Re-sorteo periodico de la tasa ---
        if (ahora - t_ultimo_sorteo) >= INTERVALO_RESORTEO:
            if random.random() < PROB_RAFAGA:
                tasa_mbps = MBPS_RAFAGA            # pico de congestion
            else:
                tasa_mbps = random.uniform(MBPS_MIN, MBPS_MAX)
            log_tasas.append((round(ahora, 3), round(tasa_mbps, 3)))
            t_ultimo_sorteo = ahora

        # --- Inyeccion por rafagas (no asfixia el CPU -> deja leer CSI) ---
        bytes_por_segundo = (tasa_mbps * 1_000_000) / 8
        paquetes_por_segundo = bytes_por_segundo / tamano_payload
        paquetes_por_rafaga = int(paquetes_por_segundo * intervalo_rafaga)

        t_inicio = time.time()
        for _ in range(paquetes_por_rafaga):
            try:
                sock.sendto(payload, (IP_DESTINO, PUERTO_UDP))
            except Exception:
                pass

        tiempo_a_dormir = intervalo_rafaga - (time.time() - t_inicio)
        if tiempo_a_dormir > 0:
            time.sleep(tiempo_a_dormir)

    sock.close()


def ejecutar_escenario_4():
    global inyector_activo, log_tasas
    print("=" * 60)
    print("  ESCENARIO 4: MOVIMIENTO HUMANO + TRAFICO UDP ALEATORIO ")
    print("=" * 60)
    print("Conecta el Wi-Fi de tu PC a CSI_TEST_NET.")
    print(f"Tasa de trafico aleatoria en [{MBPS_MIN}, {MBPS_MAX}] Mbps "
          f"(re-sorteo cada {INTERVALO_RESORTEO}s, rafagas a {MBPS_RAFAGA} Mbps).")
    print("DEBES CAMINAR de forma continua y a paso constante durante toda")
    print("la captura (igual que en el Escenario 2).")
    print("Tienes 15 segundos para ubicarte en el punto de inicio...")

    for i in range(15, 0, -1):
        sys.stdout.write(f"\rIniciando en {i} segundos...   ")
        sys.stdout.flush()
        time.sleep(1)
    print("\n\n¡Iniciando captura! Camina sin parar.\n")

    try:
        ser = serial.Serial(PUERTO, BAUDRATE)
        ser.reset_input_buffer()

        for iteracion in range(1, ITERACIONES + 1):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_csv = f"mov_trafico_iter_{iteracion}_{timestamp}.csv"
            nombre_log = f"tasas_iter_{iteracion}_{timestamp}.csv"

            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Iteracion {iteracion}/{ITERACIONES} -> {nombre_csv}")

            # Lanzar el inyector aleatorio sincronizado con la captura
            log_tasas = []
            inyector_activo = True
            t0_ref = time.time()
            hilo = threading.Thread(target=inyector_udp_aleatorio, args=(t0_ref,))
            hilo.start()

            with open(nombre_csv, mode='w', newline='') as f:
                writer = csv.writer(f)
                inicio = time.time()
                guardados = 0
                while (time.time() - inicio) < DURACION_SEGUNDOS:
                    if ser.in_waiting > 0:
                        linea = ser.readline().decode('utf-8', errors='ignore').strip()
                        if linea.startswith("CSI_DATA"):
                            datos = linea.split(',')
                            # t_rel permite alinear el CSI con la tasa de red
                            t_rel = round(time.time() - t0_ref, 3)
                            writer.writerow(datos + [t_rel])
                            guardados += 1

            inyector_activo = False
            hilo.join()

            # Guardar el log de tasas para el etiquetado posterior
            with open(nombre_log, mode='w', newline='') as fl:
                wl = csv.writer(fl)
                wl.writerow(["t_rel_seg", "mbps"])
                wl.writerows(log_tasas)

            print(f"   -> {guardados} tramas CSI | "
                  f"{len(log_tasas)} cambios de tasa registrados.\n")
            ser.reset_input_buffer()

        print("=" * 60)
        print(" ESCENARIO 4 FINALIZADO. Ya puedes detenerte.")
        print("=" * 60)
        ser.close()

    except serial.SerialException:
        inyector_activo = False
        print(f"\n[ERROR CRITICO] No se pudo abrir {PUERTO}.")
    except KeyboardInterrupt:
        inyector_activo = False
        print("\n[CANCELADO] Captura detenida manualmente.")
    except Exception as e:
        inyector_activo = False
        print(f"\n[ERROR INESPERADO] {e}")


if __name__ == "__main__":
    ejecutar_escenario_4()
