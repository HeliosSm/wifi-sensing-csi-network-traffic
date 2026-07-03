# Firmware ESP32-C6

Esta carpeta está reservada para el firmware de los nodos **ESP32-C6** utilizados
en el banco de pruebas (basado en **ESP-IDF** y **FreeRTOS**, en C/C++).

El sistema emplea dos nodos:

- **Nodo Transmisor (TX):** configurado en modo Punto de Acceso (AP). Inyecta
  tramas Wi-Fi crudas (*Raw 802.11 Frames*) mediante la API de bajo nivel
  `esp_wifi_80211_tx()`, evitando la pila TCP/IP, con una tarea de alta prioridad
  en FreeRTOS que fija una frecuencia de muestreo efectiva de **50 Hz**.
- **Nodo Receptor (RX):** configurado en modo Estación (STA) y modo promiscuo
  (*sniffer*). Filtra por dirección MAC, extrae el CSI de cada trama válida
  evaluando el *Long Training Field* (LTF) y lo envía por el puerto serie.

> **Pendiente:** coloca aquí el código fuente del firmware (proyecto ESP-IDF)
> antes de publicar la versión final del repositorio.
