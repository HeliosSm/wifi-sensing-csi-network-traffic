"""
ANALISIS DE VALIDACION FINAL - Articulo CSI Wi-Fi Sensing
Procesa los 4 escenarios con el pipeline de Avance 4/5 (Butterworth 10 Hz + PCA
SIN estandarizacion) y extiende el analisis a MULTIPLES componentes principales.

Genera, de forma reproducible y cuantitativa:
  1. Metricas PHY por escenario (PSD max, energia banda cinematica, varianza movil).
  2. Analisis multi-componente: %varianza PC1..PC6 y nº de comps para 95%.
  3. Validacion del umbral de divergencia tau sobre el Escenario 4.
  4. Separabilidad: centroides y silueta en el espacio PC1-PC2.
  5. MANOVA / ANOVA de la varianza movil entre escenarios.
  6. Correlacion varianza-movil vs Mbps instantaneos en E4 (no-contaminacion).
  7. Figuras: scree plot, varianza acumulada, scatter PC1-PC2, dashboard E4.

Ejecutar desde la carpeta raiz del proyecto:  python "Escenario 4/analisis_validacion_e4.py"
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # guardar figuras sin display
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt, welch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ----------------- Configuracion (identica a Avance 4/5) -----------------
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS, FC, ORDEN = 50.0, 10.0, 3
VENTANA_VAR = 50            # 1 s a 50 Hz
N_COMP = 6
UMBRAL_VAR_ACUM = 0.95
BANDA_CIN = (0.5, 3.0)      # banda cinematica Hz
TAU = (20, 30)             # umbral de divergencia propuesto (Avance 5)
SALIDA = os.path.join(RAIZ, "Escenario 4")

nyq = 0.5 * FS
b, a = butter(ORDEN, FC / nyq, btype="low", analog=False)

ESCENARIOS = {
    "E1_LineaBase":   "Escenario 1/linea_base_iter_*.csv",
    "E2_Movimiento":  "Escenario 2/movimiento_humano_iter_*.csv",
    "E3_Trafico":     "Escenario 3/trafico_udp_*mbps_iter_*.csv",
    "E4_Mov+Trafico": "Escenario 4/mov_trafico_iter_*.csv",
}
COLOR = {"E1_LineaBase": "gray", "E2_Movimiento": "crimson",
         "E3_Trafico": "darkorange", "E4_Mov+Trafico": "purple"}


def amplitud_filtrada(ruta):
    try:
        df = pd.read_csv(ruta, header=None, on_bad_lines="skip")
    except Exception:
        return None
    if df.shape[1] < 131:
        return None
    df = df.iloc[:, :131].dropna()
    if df.shape[0] < N_COMP + 1:
        return None
    crudo = df.iloc[:, 3:131].to_numpy(dtype=float)
    amp = np.zeros((crudo.shape[0], 64))
    for i in range(64):
        I, Q = crudo[:, 2 * i], crudo[:, 2 * i + 1]
        amp[:, i] = filtfilt(b, a, medfilt(np.sqrt(I**2 + Q**2), kernel_size=7))
    return amp


def procesar_archivo(ruta):
    amp = amplitud_filtrada(ruta)
    if amp is None:
        return None
    # PCA COMPLETO (todas las componentes) para un comps95 correcto.
    pca = PCA(n_components=min(64, amp.shape[0] - 1))
    scores_full = pca.fit_transform(amp)
    ratios_full = pca.explained_variance_ratio_
    ratios = ratios_full[:N_COMP]          # solo las primeras N para el scree
    scores = scores_full[:, :N_COMP]
    pc1 = scores_full[:, 0]

    nperseg = min(len(pc1), 128)
    f, psd = welch(pc1, fs=FS, nperseg=nperseg)
    mask = (f >= BANDA_CIN[0]) & (f <= BANDA_CIN[1])
    var_movil = pd.Series(pc1).rolling(VENTANA_VAR).var().fillna(0).to_numpy()

    # nº de componentes para 95%: si nunca se alcanza, = total disponible
    acum = np.cumsum(ratios_full)
    idx = np.where(acum >= UMBRAL_VAR_ACUM)[0]
    comps95 = int(idx[0] + 1) if idx.size else int(len(ratios_full))
    return {
        "ratios": ratios, "comps95": comps95,
        "psd_max": float(np.max(psd)),
        "energia_cin": float(np.max(psd[mask]) if mask.any() else 0.0),
        "var_movil_max": float(np.max(var_movil)),
        "var_movil_med": float(np.mean(var_movil)),
        "pc1": pc1, "f": f, "psd": psd, "var_movil": var_movil,
        "scores": scores,
    }


def resumen():
    agregado, repr_scores, repr_dash = {}, {}, {}
    feat_X, feat_y = [], []   # features por iteracion para separabilidad
    for nombre, patron in ESCENARIOS.items():
        archivos = sorted(glob.glob(os.path.join(RAIZ, patron)))
        filas = []
        for ar in archivos:
            r = procesar_archivo(ar)
            if r is None:
                continue
            filas.append(r)
            # vector de caracteristicas discriminantes (log para comprimir rangos)
            feat_X.append([np.log1p(r["var_movil_max"]), np.log1p(r["energia_cin"]),
                           r["ratios"][0], r["comps95"]])
            feat_y.append(nombre)
            if nombre not in repr_scores:
                repr_scores[nombre] = r["scores"]
                repr_dash[nombre] = r
        if not filas:
            print(f"[!] {nombre}: sin datos validos")
            continue
        agregado[nombre] = {
            "n": len(filas),
            "ratios": np.mean([f["ratios"] for f in filas], axis=0),
            "comps95": float(np.mean([f["comps95"] for f in filas])),
            "psd_max": float(np.mean([f["psd_max"] for f in filas])),
            "energia_cin": float(np.mean([f["energia_cin"] for f in filas])),
            "var_movil_max": float(np.mean([f["var_movil_max"] for f in filas])),
            "var_movil_max_list": [f["var_movil_max"] for f in filas],
        }
    return agregado, repr_scores, repr_dash, np.array(feat_X), np.array(feat_y)


def correlacion_trafico_e4():
    """Correlaciona la varianza movil de E4 con la tasa Mbps instantanea."""
    pares = []
    for csv in sorted(glob.glob(os.path.join(RAIZ, "Escenario 4/mov_trafico_iter_*.csv"))):
        base = os.path.basename(csv).replace("mov_trafico_", "tasas_")
        log = os.path.join(RAIZ, "Escenario 4", base)
        if not os.path.exists(log):
            continue
        amp = amplitud_filtrada(csv)
        if amp is None:
            continue
        pc1 = PCA(n_components=1).fit_transform(amp).flatten()
        vm = pd.Series(pc1).rolling(VENTANA_VAR).var().fillna(0).to_numpy()
        # t_rel por trama (ultima col del CSV original)
        try:
            t_rel = pd.read_csv(csv, header=None, on_bad_lines="skip").iloc[:len(vm), 131].to_numpy(float)
        except Exception:
            continue
        tasas = pd.read_csv(log)
        # tasa instantanea por trama (escalon previo)
        idx = np.searchsorted(tasas["t_rel_seg"].to_numpy(), t_rel, side="right") - 1
        idx = np.clip(idx, 0, len(tasas) - 1)
        mbps = tasas["mbps"].to_numpy()[idx]
        m = vm > 0
        pares.append(np.column_stack([mbps[m], vm[m]]))
    if not pares:
        return None
    D = np.vstack(pares)
    r, p = stats.pearsonr(D[:, 0], D[:, 1])
    return {"pearson_r": float(r), "p_value": float(p), "n": int(D.shape[0])}


def figuras(agg, scores, dash):
    # 1) scree + acumulada
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(1, N_COMP + 1)
    for k, v in agg.items():
        axL.plot(x, v["ratios"] * 100, marker="o", color=COLOR[k], label=k)
        axR.plot(x, np.cumsum(v["ratios"]) * 100, marker="s", color=COLOR[k], label=k)
    axL.set_title("Scree plot (varianza explicada por componente)")
    axL.set_xlabel("Componente principal"); axL.set_ylabel("% varianza")
    axL.grid(alpha=0.3); axL.legend()
    axR.axhline(95, ls="--", color="k", alpha=0.6)
    axR.set_title("Varianza acumulada"); axR.set_xlabel("Nº componentes")
    axR.set_ylabel("% acumulado"); axR.grid(alpha=0.3); axR.legend()
    plt.tight_layout(); plt.savefig(os.path.join(SALIDA, "fig_scree_acumulada.png"), dpi=150)
    plt.close()

    # 2) scatter PC1-PC2
    plt.figure(figsize=(8, 7))
    for k, s in scores.items():
        plt.scatter(s[:, 0], s[:, 1], s=6, alpha=0.35, color=COLOR[k], label=k)
    plt.title("Espacio PC1-PC2 por escenario")
    plt.xlabel("PC1 (energia dominante)"); plt.ylabel("PC2")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(SALIDA, "fig_scatter_pc1_pc2.png"), dpi=150)
    plt.close()

    # 3) dashboard E4 (estilo Avance 5)
    d = dash.get("E4_Mov+Trafico")
    if d:
        t = np.arange(len(d["pc1"])) / FS
        plt.figure(figsize=(15, 12))
        plt.subplot(3, 1, 1); plt.plot(t, d["pc1"], color="purple")
        plt.title("E4 - PC1 (Energia Real Absoluta)"); plt.ylabel("Amplitud"); plt.grid(alpha=0.3)
        plt.subplot(3, 1, 2); plt.plot(d["f"], d["psd"], color="darkorange", lw=2)
        plt.axvspan(0.5, 3.0, color="yellow", alpha=0.2, label="Banda Cinematica")
        plt.xlim(0, 10); plt.title("E4 - PSD (Welch)"); plt.xlabel("Hz"); plt.ylabel("PSD V^2/Hz")
        plt.legend(); plt.grid(alpha=0.3)
        plt.subplot(3, 1, 3); plt.plot(t, d["var_movil"], color="green", lw=2)
        plt.axhspan(TAU[0], TAU[1], color="red", alpha=0.15, label=f"Umbral tau {TAU}")
        plt.title("E4 - Varianza Movil"); plt.xlabel("Tiempo (s)"); plt.ylabel("Varianza")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(SALIDA, "fig_dashboard_e4.png"), dpi=150)
        plt.close()


def main():
    agg, scores, dash, feat_X, feat_y = resumen()

    print("\n" + "=" * 70)
    print(" TABLA MAESTRA DE RESULTADOS (promedios por escenario)")
    print("=" * 70)
    print(f"{'Escenario':<16}{'n':>4}{'%PC1':>8}{'Comp95':>8}"
          f"{'PSD_max':>10}{'E_cin':>9}{'VarMov_max':>12}")
    for k, v in agg.items():
        print(f"{k:<16}{v['n']:>4}{v['ratios'][0]*100:>7.1f}%{v['comps95']:>8.2f}"
              f"{v['psd_max']:>10.3f}{v['energia_cin']:>9.3f}{v['var_movil_max']:>12.2f}")

    # Validacion del umbral tau sobre E4
    if "E4_Mov+Trafico" in agg and "E2_Movimiento" in agg:
        e4 = np.array(agg["E4_Mov+Trafico"]["var_movil_max_list"])
        deteccion = float(np.mean(e4 > TAU[1]))
        print("\n--- VALIDACION DEL UMBRAL (Objetivo 3 / Actividad 8-9) ---")
        print(f" Var.movil E2 (mov)      : {agg['E2_Movimiento']['var_movil_max']:.2f}")
        print(f" Var.movil E3 (trafico)  : {agg['E3_Trafico']['var_movil_max']:.2f}")
        print(f" Var.movil E4 (mov+traf) : {agg['E4_Mov+Trafico']['var_movil_max']:.2f}")
        print(f" Deteccion de presencia en E4 con tau>{TAU[1]}: {deteccion*100:.1f}% de iteraciones")

    # Separabilidad sobre features por iteracion (var_movil, E_cin, %PC1, comps95)
    try:
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(feat_X)
        sil = silhouette_score(Xs, feat_y)
        print(f"\n Coef. de silueta (features discriminantes, 4 clases): {sil:.3f}")
        print("  (features: log varianza movil, log energia cinematica, %PC1, comps95)")
    except Exception as e:
        print(f" [silueta] {e}")

    # ANOVA de la varianza movil entre escenarios
    grupos = [np.array(v["var_movil_max_list"]) for v in agg.values() if "var_movil_max_list" in v]
    if len(grupos) >= 2:
        F, p = stats.f_oneway(*grupos)
        print(f"\n ANOVA var.movil entre escenarios: F={F:.2f}, p={p:.3e}")

    # Correlacion trafico vs varianza en E4
    corr = correlacion_trafico_e4()
    if corr:
        print(f"\n--- NO-CONTAMINACION (E4): correlacion Mbps vs varianza movil ---")
        print(f" Pearson r = {corr['pearson_r']:.4f}  (p={corr['p_value']:.3e}, n={corr['n']})")
        print(" => r cercano a 0 indica que la tasa de red NO modula la firma cinematica.")

    figuras(agg, scores, dash)
    print(f"\nFiguras guardadas en: {SALIDA}")

    # Volcado JSON para el manuscrito
    dump = {k: {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist())
                for kk, vv in v.items() if kk != "var_movil_max_list"}
            for k, v in agg.items()}
    with open(os.path.join(SALIDA, "resultados.json"), "w") as fp:
        json.dump(dump, fp, indent=2)
    print("Metricas numericas guardadas en resultados.json")


if __name__ == "__main__":
    main()
