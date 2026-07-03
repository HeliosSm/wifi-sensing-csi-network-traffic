import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt
from sklearn.decomposition import PCA

# ==========================================================
#  ANALISIS MULTI-COMPONENTE (PC1, PC2, PC3, PC4 ...)
#  para los 4 escenarios. Extiende el pipeline de Avance 4/5
#  (Butterworth + PCA SIN StandardScaler, energia preservada)
#  mas alla de PC1 para validar el estudio con varias componentes.
# ==========================================================

FS = 50.0
FC = 10.0
ORDEN = 3
N_COMPONENTES = 6          # cuantas PCs analizar/graficar
UMBRAL_VAR_ACUM = 0.95     # criterio de varianza acumulada (95%)

nyq = 0.5 * FS
b, a = butter(ORDEN, FC / nyq, btype='low', analog=False)

# Mapa escenario -> patron de archivos (ajusta las rutas a tu carpeta)
ESCENARIOS = {
    "E1_Linea_Base":  "Escenario 1/linea_base_iter_*.csv",
    "E2_Movimiento":  "Escenario 2/movimiento_humano_iter_*.csv",
    "E3_Trafico":     "Escenario 3/trafico_udp_*mbps_iter_*.csv",
    "E4_Mov+Trafico": "Escenario 4/mov_trafico_iter_*.csv",
}


def matriz_amplitud_filtrada(ruta):
    """Devuelve la matriz [tramas x 64] filtrada (igual que Avance 4/5)."""
    try:
        df = pd.read_csv(ruta, header=None, on_bad_lines='skip')
    except Exception:
        return None
    # E1-E3 tienen 131 columnas; E4 agrega t_rel -> 132. Toleramos >=131.
    if df.shape[1] < 131:
        return None
    df = df.dropna(subset=df.columns[:131])
    if df.empty:
        return None

    crudo = df.iloc[:, 3:131].to_numpy(dtype=float)
    n = crudo.shape[0]
    amp = np.zeros((n, 64))
    for i in range(64):
        I, Q = crudo[:, 2 * i], crudo[:, 2 * i + 1]
        amp_raw = np.sqrt(I**2 + Q**2)
        amp_med = medfilt(amp_raw, kernel_size=7)
        amp[:, i] = filtfilt(b, a, amp_med)
    return amp


def pca_de_archivo(ruta):
    """PCA completo (sin escalado) sobre un archivo. Devuelve scores y ratios."""
    amp = matriz_amplitud_filtrada(ruta)
    if amp is None or amp.shape[0] < N_COMPONENTES:
        return None
    pca = PCA(n_components=N_COMPONENTES)
    scores = pca.fit_transform(amp)              # [tramas x N_COMPONENTES]
    ratios = pca.explained_variance_ratio_       # varianza explicada por PC
    return scores, ratios


def resumen_escenario(nombre, patron):
    """Promedia ratios y reune metricas discriminantes por escenario."""
    archivos = sorted(glob.glob(patron))
    if not archivos:
        print(f"[!] {nombre}: sin archivos para '{patron}'")
        return None

    ratios_acum, comps_95, var_pc1 = [], [], []
    scores_repr = None
    for ar in archivos:
        res = pca_de_archivo(ar)
        if res is None:
            continue
        scores, ratios = res
        ratios_acum.append(ratios)
        # nº de componentes para alcanzar el 95% de varianza
        acum = np.cumsum(ratios)
        comps_95.append(int(np.argmax(acum >= UMBRAL_VAR_ACUM) + 1))
        var_pc1.append(ratios[0])
        if scores_repr is None:
            scores_repr = scores                 # guarda el 1er archivo para scatter

    if not ratios_acum:
        return None

    ratios_medio = np.mean(np.vstack(ratios_acum), axis=0)
    print(f"\n=== {nombre} ({len(ratios_acum)} archivos) ===")
    for k, r in enumerate(ratios_medio, 1):
        print(f"  PC{k}: {r*100:5.1f}% var")
    print(f"  -> Varianza media en PC1 : {np.mean(var_pc1)*100:.1f}%")
    print(f"  -> Componentes para 95%  : {np.mean(comps_95):.2f} "
          f"(min {min(comps_95)}, max {max(comps_95)})")
    return {
        "nombre": nombre,
        "ratios_medio": ratios_medio,
        "comps_95": np.mean(comps_95),
        "var_pc1": np.mean(var_pc1),
        "scores_repr": scores_repr,
    }


def graficar(resumenes):
    colores = {"E1_Linea_Base": "gray", "E2_Movimiento": "crimson",
               "E3_Trafico": "darkorange", "E4_Mov+Trafico": "purple"}

    # --- 1) Scree plot + varianza acumulada ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(1, N_COMPONENTES + 1)
    for r in resumenes:
        c = colores.get(r["nombre"], None)
        axL.plot(x, r["ratios_medio"] * 100, marker='o', color=c, label=r["nombre"])
        axR.plot(x, np.cumsum(r["ratios_medio"]) * 100, marker='s', color=c,
                 label=r["nombre"])
    axL.set_title("Scree plot (varianza explicada por componente)")
    axL.set_xlabel("Componente principal"); axL.set_ylabel("% varianza")
    axL.grid(alpha=0.3); axL.legend()
    axR.axhline(95, ls='--', color='k', alpha=0.6, label="95%")
    axR.set_title("Varianza acumulada")
    axR.set_xlabel("Nº de componentes"); axR.set_ylabel("% varianza acumulada")
    axR.grid(alpha=0.3); axR.legend()
    plt.tight_layout(); plt.show()

    # --- 2) Dispersion en el espacio PC1-PC2 (separabilidad de clases) ---
    plt.figure(figsize=(8, 7))
    for r in resumenes:
        s = r["scores_repr"]
        if s is None:
            continue
        plt.scatter(s[:, 0], s[:, 1], s=6, alpha=0.4,
                    color=colores.get(r["nombre"], None), label=r["nombre"])
    plt.title("Proyeccion en el espacio PC1-PC2 por escenario")
    plt.xlabel("PC1 (energia dominante)"); plt.ylabel("PC2")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.show()


def main():
    resumenes = []
    for nombre, patron in ESCENARIOS.items():
        r = resumen_escenario(nombre, patron)
        if r is not None:
            resumenes.append(r)

    if resumenes:
        print("\n--- TABLA RESUMEN (feature discriminante propuesto) ---")
        print(f"{'Escenario':<18}{'%PC1':>8}{'Comps 95%':>12}")
        for r in resumenes:
            print(f"{r['nombre']:<18}{r['var_pc1']*100:>7.1f}%{r['comps_95']:>12.2f}")
        graficar(resumenes)


if __name__ == "__main__":
    main()
