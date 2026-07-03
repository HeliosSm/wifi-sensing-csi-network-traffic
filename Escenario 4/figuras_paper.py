"""Genera las figuras del articulo (IEEE Access) directamente en la carpeta PRI.
Nomenclatura IEEE: primeras 5 letras del autor -> ramonN.png
Orden de aparicion en el texto:
  ramon2 = acondicionamiento/limpieza (Actividad 4)
  ramon3 = morfologia PC1 (4 escenarios)
  ramon4 = scree + varianza acumulada (multi-componente)
  ramon5 = dashboard Escenario 4
  ramon6 = no-contaminacion (Mbps vs varianza)
Reutiliza el pipeline de validacion (Butterworth 10 Hz + PCA sin escalar)."""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import medfilt, butter, filtfilt, welch, spectrogram
from scipy import stats
from sklearn.decomposition import PCA

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(RAIZ, "figures")
os.makedirs(DEST, exist_ok=True)
FS, FC, ORDEN, VENTANA_VAR, N_COMP = 50.0, 10.0, 3, 50, 6
nyq = 0.5 * FS
b, a = butter(ORDEN, FC / nyq, btype="low", analog=False)

ESC = {
    "E1 (Baseline)":          ("Escenario 1/linea_base_iter_*.csv", "gray"),
    "E2 (Motion)":            ("Escenario 2/movimiento_humano_iter_*.csv", "crimson"),
    "E3 (Traffic)":           ("Escenario 3/trafico_udp_*mbps_iter_*.csv", "darkorange"),
    "E4 (Motion + Traffic)":  ("Escenario 4/mov_trafico_iter_*.csv", "purple"),
}
plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    # Tipografia Times New Roman en todo el texto de las figuras
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


def amp_filt(ruta):
    """Amplitud filtrada (mediana + Butterworth) -> matriz [tramas x 64]."""
    df = pd.read_csv(ruta, header=None, on_bad_lines="skip")
    if df.shape[1] < 131:
        return None
    df = df.iloc[:, :131].dropna()
    if df.shape[0] < N_COMP + 1:
        return None
    crudo = df.iloc[:, 3:131].to_numpy(float)
    amp = np.zeros((crudo.shape[0], 64))
    for i in range(64):
        I, Q = crudo[:, 2 * i], crudo[:, 2 * i + 1]
        amp[:, i] = filtfilt(b, a, medfilt(np.sqrt(I**2 + Q**2), 7))
    return amp


def pc1_de(ruta):
    amp = amp_filt(ruta)
    return None if amp is None else PCA(n_components=1).fit_transform(amp).flatten()


def amp_cruda_y_mediana(ruta):
    """Amplitud bruta y amplitud tras filtro de mediana (sin Butterworth)."""
    df = pd.read_csv(ruta, header=None, on_bad_lines="skip")
    if df.shape[1] < 131:
        return None
    df = df.iloc[:, :131].dropna()
    crudo = df.iloc[:, 3:131].to_numpy(float)
    n = crudo.shape[0]
    bruta = np.zeros((n, 64))
    med = np.zeros((n, 64))
    for i in range(64):
        I, Q = crudo[:, 2 * i], crudo[:, 2 * i + 1]
        ar = np.sqrt(I**2 + Q**2)
        bruta[:, i] = ar
        med[:, i] = medfilt(ar, 7)
    return bruta, med


def _norm_minmax(M):
    out = np.zeros_like(M)
    for i in range(M.shape[1]):
        lo, hi = M[:, i].min(), M[:, i].max()
        out[:, i] = (M[:, i] - lo) / (hi - lo) if hi > lo else 0.0
    return out


# ---------- FIG 2 (ramon2): plano del entorno fisico (Metodologia) ----------
def fig_escenario_fisico():
    W, H = 5.00, 3.40
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    # Paredes de la habitacion
    ax.add_patch(mpatches.Rectangle((0, 0), W, H, fill=False, lw=2.5, edgecolor="black"))
    # Mobiliario estatico (generadores de multitrayecto)
    ax.add_patch(mpatches.Rectangle((0.20, 0.20), 1.30, 0.55, facecolor="#d2b48c", edgecolor="k"))
    ax.text(0.85, 0.475, "Armario", ha="center", va="center", fontsize=8)
    ax.add_patch(mpatches.Rectangle((3.15, 1.95), 1.65, 1.25, facecolor="#f4c2c2", edgecolor="k"))
    ax.text(3.975, 2.575, "Cama", ha="center", va="center", fontsize=8)
    # Ventanas en la pared izquierda (segmentos 2.05 m + 1.35 m)
    ax.plot([0, 0], [0.10, 2.05], color="deepskyblue", lw=5, solid_capstyle="butt")
    ax.plot([0, 0], [2.05, 3.30], color="deepskyblue", lw=5, solid_capstyle="butt")
    ax.text(-0.45, 1.05, "2.05 m", rotation=90, ha="center", va="center", fontsize=7)
    ax.text(-0.45, 2.70, "1.35 m", rotation=90, ha="center", va="center", fontsize=7)
    # Nodos TX y RX
    tx, rx = (0.45, 2.85), (4.70, 0.45)
    ax.plot(*tx, "o", color="blue", ms=11, zorder=5)
    ax.text(tx[0] + 0.18, tx[1], "TX", color="blue", fontsize=9, fontweight="bold", va="center")
    ax.plot(*rx, "o", color="green", ms=11, zorder=5)
    ax.text(rx[0] - 0.18, rx[1], "RX", color="green", fontsize=9, fontweight="bold", va="center", ha="right")
    # Linea de vista (LOS)
    ax.annotate("", xy=rx, xytext=tx, arrowprops=dict(arrowstyle="<->", ls="--", color="red", lw=1.3))
    ax.text((tx[0] + rx[0]) / 2 + 0.1, (tx[1] + rx[1]) / 2 + 0.18,
            r"LOS $\approx$ 4.99 m", color="red", fontsize=8, rotation=-27, ha="center")
    # Cotas exteriores
    ax.annotate("", xy=(0, -0.30), xytext=(W, -0.30), arrowprops=dict(arrowstyle="<->", lw=1))
    ax.text(W / 2, -0.55, "5.00 m", ha="center", fontsize=8)
    ax.annotate("", xy=(W + 0.30, 0), xytext=(W + 0.30, H), arrowprops=dict(arrowstyle="<->", lw=1))
    ax.text(W + 0.55, H / 2, "3.40 m", va="center", rotation=90, fontsize=8)
    # Nota de altura
    ax.text(W / 2, H + 0.28, "Nodos ESP32-C6 a 1.50 m de altura",
            ha="center", fontsize=8, style="italic")
    ax.set_xlim(-0.9, 6.1); ax.set_ylim(-0.9, 3.95)
    ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(DEST, "ramon2.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ---------- FIG 3 (ramon3): acondicionamiento / limpieza (Actividad 4) ----------
def fig_preprocesamiento():
    archivos = sorted(glob.glob(os.path.join(RAIZ, "Escenario 1/linea_base_iter_*.csv")))
    res = next((amp_cruda_y_mediana(f) for f in archivos
                if amp_cruda_y_mediana(f) is not None), None)
    bruta, med = res
    sc = int(np.argmax(np.max(np.abs(bruta - med), axis=0)))  # subportadora con mas outliers
    t = np.arange(bruta.shape[0]) / FS
    norm = _norm_minmax(med)

    varis = []
    for f in archivos:
        r = amp_cruda_y_mediana(f)
        if r is not None:
            varis.append(np.var(_norm_minmax(r[1])))
    print(f"FIG2 -> var. normalizada media E1 = {np.mean(varis):.4f} "
          f"(sd {np.std(varis):.4f}, n={len(varis)})")

    fig, ax = plt.subplots(2, 1, figsize=(7.0, 6.2))
    ax[0].plot(t, bruta[:, sc], color="red", alpha=0.5, lw=0.8, label="Raw signal (with outliers)")
    ax[0].plot(t, med[:, sc], color="blue", lw=1.4, label="Filtered signal (median, N=7)")
    ax[0].set_ylabel("Amplitude"); ax[0].set_xlabel("Time (s)")
    ax[0].text(0.01, 0.97, "(a)", transform=ax[0].transAxes, fontsize=11, va="top", fontweight="bold")
    ax[0].legend(fontsize=8, loc="upper right")
    im = ax[1].imshow(norm.T, aspect="auto", cmap="viridis", origin="lower",
                      extent=[0, t[-1], 1, 64], interpolation="nearest")
    ax[1].set_xlabel("Time (s)"); ax[1].set_ylabel("Subcarrier")
    ax[1].text(0.01, 0.97, "(b)", transform=ax[1].transAxes, fontsize=11, va="top",
               fontweight="bold", color="white")
    fig.colorbar(im, ax=ax[1], label="Normalized amplitude")
    plt.tight_layout()
    plt.savefig(os.path.join(DEST, "ramon3.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ---------- FIG 3 (ramon3): morfologia PC1 de los 4 escenarios ----------
def fig_morfologia():
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 8.0), sharex=True)
    for ax, (nombre, (patron, color)) in zip(axes, ESC.items()):
        archivos = sorted(glob.glob(os.path.join(RAIZ, patron)))
        pc1 = next((p for p in (pc1_de(f) for f in archivos) if p is not None), None)
        if pc1 is None:
            continue
        t = np.arange(len(pc1)) / FS
        ax.plot(t, pc1, color=color, lw=0.9)
        ax.set_ylabel("PC1")
        ax.text(0.01, 0.92, nombre, transform=ax.transAxes, fontsize=10,
                va="top", fontweight="bold")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(DEST, "ramon4.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ---------- FIG 4 (ramon4): espectrogramas tiempo-frecuencia (Actividad 7) ----------
def fig_espectrograma():
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.2))
    for ax, (nombre, (patron, _)) in zip(axes.ravel(), ESC.items()):
        archivos = sorted(glob.glob(os.path.join(RAIZ, patron)))
        pc1 = next((p for p in (pc1_de(f) for f in archivos) if p is not None), None)
        if pc1 is None:
            continue
        fr, tt, Sxx = spectrogram(pc1, fs=FS, nperseg=128, noverlap=112)
        Sdb = 10 * np.log10(Sxx + 1e-12)
        m = fr <= 10
        pcm = ax.pcolormesh(tt, fr[m], Sdb[m], shading="gouraud",
                            cmap="inferno", vmin=-30, vmax=20)
        ax.axhline(0.5, color="cyan", ls="--", lw=0.8, alpha=0.7)
        ax.axhline(3.0, color="cyan", ls="--", lw=0.8, alpha=0.7)
        ax.text(0.02, 0.94, nombre, transform=ax.transAxes, fontsize=9,
                va="top", fontweight="bold", color="white")
        ax.set_ylim(0, 10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Frequency (Hz)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    fig.colorbar(pcm, ax=axes, label="PSD (dB)", shrink=0.85, pad=0.02)
    plt.savefig(os.path.join(DEST, "ramon5.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ---------- FIG 5 (ramon5): scree + varianza acumulada ----------
def fig_scree():
    ratios = {}
    for nombre, (patron, color) in ESC.items():
        rr = []
        for f in sorted(glob.glob(os.path.join(RAIZ, patron))):
            amp = amp_filt(f)
            if amp is None:
                continue
            rr.append(PCA(n_components=N_COMP).fit(amp).explained_variance_ratio_)
        if rr:
            ratios[nombre] = (np.mean(rr, axis=0), color)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.16, 3.0))
    x = np.arange(1, N_COMP + 1)
    for nombre, (r, c) in ratios.items():
        axL.plot(x, r * 100, marker="o", ms=4, color=c, label=nombre)
        axR.plot(x, np.cumsum(r) * 100, marker="s", ms=4, color=c, label=nombre)
    axL.set_xlabel("Principal component"); axL.set_ylabel("Explained variance (%)")
    axL.legend(fontsize=7)
    axL.text(0.03, 0.05, "(a)", transform=axL.transAxes, fontsize=11, va="bottom",
             ha="left", fontweight="bold")
    axR.axhline(95, ls="--", color="k", alpha=0.6)
    axR.set_xlabel("Number of components"); axR.set_ylabel("Cumulative variance (%)")
    axR.legend(fontsize=7)
    axR.text(0.97, 0.05, "(b)", transform=axR.transAxes, fontsize=11, va="bottom",
             ha="right", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(DEST, "ramon6.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ---------- FIG 6 (ramon6): dashboard E4 (PC1 / PSD / varianza movil) ----------
def fig_dashboard_e4():
    f0 = sorted(glob.glob(os.path.join(RAIZ, "Escenario 4/mov_trafico_iter_*.csv")))[0]
    pc1 = pc1_de(f0)
    t = np.arange(len(pc1)) / FS
    fr, psd = welch(pc1, fs=FS, nperseg=min(len(pc1), 128))
    vm = pd.Series(pc1).rolling(VENTANA_VAR).var().fillna(0).to_numpy()

    fig, ax = plt.subplots(3, 1, figsize=(7.16, 7.0))
    ax[0].plot(t, pc1, color="purple"); ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("Time (s)")
    ax[0].text(0.01, 0.95, "(a)", transform=ax[0].transAxes, fontsize=11, va="top", fontweight="bold")
    ax[1].plot(fr, psd, color="darkorange", lw=1.8)
    ax[1].axvspan(0.5, 3.0, color="gold", alpha=0.25, label="Kinematic band (0.5-3 Hz)")
    ax[1].set_xlim(0, 10); ax[1].set_ylabel("PSD (V$^2$/Hz)"); ax[1].set_xlabel("Frequency (Hz)")
    ax[1].text(0.01, 0.95, "(b)", transform=ax[1].transAxes, fontsize=11, va="top", fontweight="bold")
    ax[1].legend(fontsize=8)
    ax[2].plot(t, vm, color="green", lw=1.5)
    ax[2].axhspan(20, 30, color="red", alpha=0.15, label="Divergence threshold (20-30)")
    ax[2].set_ylabel("Variance"); ax[2].set_xlabel("Time (s)")
    ax[2].text(0.01, 0.95, "(c)", transform=ax[2].transAxes, fontsize=11, va="top", fontweight="bold")
    ax[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(DEST, "ramon7.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ---------- FIG 7 (ramon7): no-contaminacion (Mbps vs varianza movil en E4) ----------
def fig_no_contaminacion():
    xs, ys = [], []
    for csv in sorted(glob.glob(os.path.join(RAIZ, "Escenario 4/mov_trafico_iter_*.csv"))):
        log = os.path.join(RAIZ, "Escenario 4",
                           os.path.basename(csv).replace("mov_trafico_", "tasas_"))
        if not os.path.exists(log):
            continue
        amp = amp_filt(csv)
        if amp is None:
            continue
        pc1 = PCA(n_components=1).fit_transform(amp).flatten()
        vm = pd.Series(pc1).rolling(VENTANA_VAR).var().fillna(0).to_numpy()
        try:
            t_rel = pd.read_csv(csv, header=None, on_bad_lines="skip").iloc[:len(vm), 131].to_numpy(float)
        except Exception:
            continue
        tasas = pd.read_csv(log)
        idx = np.clip(np.searchsorted(tasas["t_rel_seg"].to_numpy(), t_rel, "right") - 1,
                      0, len(tasas) - 1)
        mbps = tasas["mbps"].to_numpy()[idx]
        m = vm > 0
        xs.append(mbps[m]); ys.append(vm[m])
    X, Y = np.concatenate(xs), np.concatenate(ys)
    r, p = stats.pearsonr(X, Y)

    plt.figure(figsize=(3.5, 3.2))
    plt.hexbin(X, Y, gridsize=30, cmap="Purples", mincnt=1)
    cb = plt.colorbar(); cb.set_label("Frame density", fontsize=8)
    s, b0, *_ = stats.linregress(X, Y)
    xx = np.linspace(X.min(), X.max(), 50)
    plt.plot(xx, s * xx + b0, "r--", lw=1.5)
    plt.text(0.05, 0.92, f"$r = {r:.3f}$", transform=plt.gca().transAxes, fontsize=10, color="red")
    plt.xlabel("Instantaneous traffic rate (Mbps)")
    plt.ylabel("PC1 moving variance")
    plt.tight_layout()
    plt.savefig(os.path.join(DEST, "ramon8.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"FIG7 -> r={r:.4f}, p={p:.3e}, n={len(X)}")


if __name__ == "__main__":
    # ramon2.png (plano del entorno) se genera con plano.tex (TikZ standalone)
    fig_preprocesamiento();  print("ramon3.png OK")
    fig_morfologia();        print("ramon4.png OK")
    fig_espectrograma();     print("ramon5.png OK")
    fig_scree();             print("ramon6.png OK")
    fig_dashboard_e4();      print("ramon7.png OK")
    fig_no_contaminacion();  print("ramon8.png OK")
    print("Figuras en:", DEST)
