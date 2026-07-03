# Análisis Experimental del Impacto del Tráfico de Red en Mediciones CSI para Wi-Fi Sensing

Repositorio de código y datos que acompaña al artículo del mismo título. El trabajo
evalúa, a nivel de **capa física**, si las variaciones de la Información del Estado
del Canal (**CSI**) provocadas por el **tráfico de red** pueden distinguirse de las
generadas por el **movimiento humano**, empleando un banco de pruebas de bajo costo
con dos microcontroladores **ESP32-C6**.

<!-- Cuando publiques el conjunto de datos completo en Zenodo, reemplaza el
     siguiente marcador por la insignia real del DOI:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
-->
**DOI del conjunto de datos (Zenodo):** _pendiente de asignar_

---

## Descripción

A partir de la captura sincronizada de CSI a una frecuencia de muestreo efectiva de
**50 Hz** en un entorno interior controlado, se registran cuatro escenarios y se
aplica un *pipeline* determinista de procesamiento en la capa física:

1. Cálculo de la amplitud por subportadora a partir de las componentes I/Q.
2. Filtro de mediana (`N = 7`) para eliminar ruido impulsivo.
3. Filtro Butterworth pasabajo de tercer orden (`fc = 10 Hz`).
4. Análisis de Componentes Principales (**PCA**) **sin estandarización previa**,
   preservando la energía absoluta del canal.
5. Extracción de métricas físicas de divergencia: densidad espectral de potencia
   (método de Welch), varianza móvil y dimensionalidad efectiva `d95`.

## Escenarios experimentales

| Escenario | Descripción | Capturas CSI |
|-----------|-------------|:------------:|
| **E1 — Línea base** | Entorno estático, sin personas ni tráfico. | 15 |
| **E2 — Movimiento humano** | Un sujeto camina cruzando la línea de vista (LOS). | 20 |
| **E3 — Tráfico de red** | Canal saturado con tráfico UDP (2, 5 y 10 Mbps), entorno estático. | 45 |
| **E4 — Coexistencia** | Movimiento humano + tráfico de tasa aleatoria (2–10 Mbps, ráfagas de 12 Mbps). | 20 |

Total: **100 capturas** de ventanas continuas de **60 s**. En el Escenario 4, además
de la señal CSI (`mov_trafico_iter_*.csv`) se registra la **tasa de tráfico
instantánea** por trama (`tasas_iter_*.csv`) para el análisis de independencia.

## Estructura del repositorio

```
wifi-sensing-csi-network-traffic/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── Escenario 1/          # Línea base: scripts + 15 CSV
├── Escenario 2/          # Movimiento humano: scripts + 20 CSV
├── Escenario 3/          # Tráfico de red: scripts + 45 CSV
├── Escenario 4/          # Coexistencia: scripts + 20 CSI + 20 tasas + resultados.json
├── firmware/             # (Reservado) firmware ESP32-C6 sobre ESP-IDF
└── figures/              # Figuras generadas por los scripts (salida)
```

Cada carpeta `Escenario N` contiene tanto el script de captura como los scripts de
procesamiento y sus datos CSV, de modo que el análisis es reproducible sin mover
archivos. Scripts principales:

- `captura_automatica*.py` — captura automatizada del CSI desde el ESP32-C6 (requiere `pyserial` y el hardware).
- `filtrado_avanzado.py` — acondicionamiento de la señal (mediana + Butterworth).
- `metricas.py` — cálculo de métricas físicas por captura.
- `actividad7_espectrogramas.py` — espectrogramas (STFT).
- `pca_multicomponente.py` — análisis multi-componente y dimensionalidad efectiva (Escenario 4).
- `analisis_validacion_e4.py` — validación estadística e independencia (Escenario 4).
- `figuras_paper.py` — genera todas las figuras del artículo en `figures/` (Escenario 4).

## Formato de los datos CSV

Cada archivo es un volcado crudo del CSI entregado por el ESP32-C6. Cada línea
comienza con la etiqueta `CSI_DATA` seguida de metadatos (p. ej. RSSI) y de los
pares **I/Q** de las **64 subportadoras OFDM** del canal de 20 MHz (canal 6,
banda de 2.4 GHz). Las subportadoras nulas aparecen como pares `0,0`.

## Requisitos e instalación

```bash
python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Probado con Python 3.10+.

## Reproducción del análisis

```bash
# Ejemplo: regenerar todas las figuras del artículo
cd "Escenario 4"
python figuras_paper.py        # las figuras se guardan en ../figures/

# Ejemplo: métricas de un escenario
cd "Escenario 1"
python metricas.py
```

> Los scripts de captura (`captura_automatica*.py`) requieren los nodos ESP32-C6
> conectados por puerto serie; el resto de scripts funcionan directamente sobre los
> CSV incluidos.

## Material complementario

- **Video del Escenario 4 (~20 min):** grabación del sujeto caminando en el entorno
  de prueba durante la captura de coexistencia. Por su tamaño (~1.1 GB) **no se
  incluye en GitHub**; se publica junto al conjunto de datos en **Zenodo** (ver DOI
  arriba).
- **Firmware ESP32-C6:** ver `firmware/README.md`.

## Cómo citar

Si utilizas este material, por favor cita el artículo (y el conjunto de datos de
Zenodo una vez disponible):

```bibtex
@article{ramon_wifisensing_csi,
  author  = {Ram{\'o}n, Erick and Vinces, Mart{\'i}n and Gonz{\'a}lez Mart{\'i}nez, Santiago Ren{\'a}n},
  title   = {An{\'a}lisis Experimental del Impacto del Tr{\'a}fico de Red en Mediciones CSI para Wi-Fi Sensing},
  journal = {IEEE Access},
  year    = {2026},
  note    = {C{\'o}digo y datos: este repositorio}
}
```

## Autores

- **Erick Ramón** — Universidad de Cuenca
- **Martín Vinces** — Universidad de Cuenca
- **Santiago Renán González Martínez, Ph.D.** — Universidad de Cuenca (director)

## Licencia

El código se distribuye bajo la licencia **MIT** (ver `LICENSE`). Los datos CSI se
comparten con fines de investigación y educativos; si publicas el conjunto en Zenodo,
considera una licencia **CC BY 4.0** para los datos.
