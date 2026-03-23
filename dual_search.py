"""
dual_search.py
==============
Cerca segnali exp-drop simultanei in **due** serie temporali con lo stesso
campionamento e stesso numero di punti.

Modello
-------
Le due serie condividono t0 e b (stesso evento fisico, stessa scala di
recupero) ma possono avere ampiezze e fondi diversi:

    mu_1(t) = k1 - a1 * exp(-(t-t0)/b)   per t > t0   (k1 altrimenti)
    mu_2(t) = k2 - a2 * exp(-(t-t0)/b)   per t > t0   (k2 altrimenti)

Le due serie sono statisticamente indipendenti: il profilo LLR congiunto
per b condiviso è la somma dei profili individuali:

    LLR_joint(t0, b) = max_a1 LLR1(t0,a1,b) + max_a2 LLR2(t0,a2,b)
    LLR_joint(t0)    = max_b  LLR_joint(t0, b)

Categorie di rilevamento
------------------------
    "both"        : entrambe le serie individualmente sopra soglia
    "joint_only"  : solo LLR congiunto sopra soglia (segnale debole in ciascuna)
    "only1"       : solo serie 1 sopra soglia
    "only2"       : solo serie 2 sopra soglia

Funzioni pubbliche
------------------
    calibrate_thresholds_dual(k1, k2, N, ...)  -> (thr1, thr2, thr_joint)
    find_signals_dual(data1, data2, ...)        -> dict
    plot_dual_results(data1, data2, results, ..., plot_path)
    write_dual_report(results, ..., report_path)
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from datetime import datetime, timezone
from pathlib import Path
from scipy.stats import beta as _beta_dist

from main import (
    _mu, _anticausal, _llr_exact, _refine,
    _default_grids, _TAYLOR_ORDER,
)


# ── profilo LLR per-b (nucleo della ricerca congiunta) ────────────────────────

def _llr_profile_perbval(
    data:     np.ndarray,
    k:        float,
    a_values: np.ndarray,
    b_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Profilo LLR massimizzato su ``a`` per ogni coppia (t0, b).

    Ritorna
    -------
    profiles : ndarray (N, Nb) — max_a LLR(t0, a, b) per ogni (t0, b)
    best_a   : ndarray (N, Nb) — valore di a ottimo
    """
    N  = len(data)
    Nb = len(b_values)
    profiles = np.full((N, Nb), -np.inf)
    best_a   = np.zeros((N, Nb))
    ones     = np.ones(N, dtype=np.float64)

    for j, b_val in enumerate(b_values):
        alpha = float(np.exp(-1.0 / b_val))

        # termini di Taylor: C_n[t] = Σ_{s≥1} data[t+s] · α^{n·s}
        taylor: list[np.ndarray] = []
        alp_n = alpha
        for _ in range(_TAYLOR_ORDER):
            taylor.append(_anticausal(data, alp_n))
            alp_n *= alpha
        s1 = _anticausal(ones, alpha)   # S1[t] = Σ_{s≥1} α^s

        for a_val in a_values:
            if a_val >= k:
                continue
            r   = a_val / k
            llr = a_val * s1
            rn  = r
            for n, cn in enumerate(taylor, start=1):
                llr -= (rn / n) * cn
                rn  *= r

            better              = llr > profiles[:, j]
            profiles[better, j] = llr[better]
            best_a[better, j]   = a_val

    return profiles, best_a


# ── calibrazione soglie ───────────────────────────────────────────────────────

def calibrate_thresholds_dual(
    k1:     float,
    k2:     float,
    N:      int,
    n_cal:  int   = 50_000,
    fpr:    float = 0.01,
    n_sim:  int   = 200,
    seed:   int | None = None,
    a_values1: np.ndarray | None = None,
    a_values2: np.ndarray | None = None,
    b_values:  np.ndarray | None = None,
) -> tuple[float, float, float]:
    """
    Calibra via Monte Carlo tre soglie LLR:
      - ``thr1``   : soglia per la serie 1 da sola
      - ``thr2``   : soglia per la serie 2 da sola
      - ``thr_joint``: soglia per il test congiunto (b condiviso)

    Applica la correzione di Bonferroni per serie di lunghezza N >> n_cal.

    Ritorna
    -------
    (thr1, thr2, thr_joint) : tuple di float
    """
    rng = np.random.default_rng(seed)

    if a_values1 is None: a_values1, _bv = _default_grids(k1)
    if a_values2 is None: a_values2, _   = _default_grids(k2)
    if b_values  is None: _, b_values    = _default_grids(k1)

    n_windows = max(1.0, N / n_cal)
    fpr_loc   = fpr / n_windows
    q         = 1.0 - fpr_loc

    max1, max2, maxj = [], [], []

    for _ in range(n_sim):
        null1 = rng.poisson(k1, n_cal).astype(np.float64)
        null2 = rng.poisson(k2, n_cal).astype(np.float64)

        p1, _ = _llr_profile_perbval(null1, float(null1.mean()), a_values1, b_values)
        p2, _ = _llr_profile_perbval(null2, float(null2.mean()), a_values2, b_values)

        max1.append(float(p1.max()))
        max2.append(float(p2.max()))
        maxj.append(float((p1 + p2).max()))   # joint: stessa b, stesso t0

    thr1  = float(np.quantile(max1,  q))
    thr2  = float(np.quantile(max2,  q))
    thrj  = float(np.quantile(maxj,  q))
    return thr1, thr2, thrj


# ── ricerca principale ────────────────────────────────────────────────────────

def find_signals_dual(
    data1: np.ndarray,
    data2: np.ndarray,
    threshold1:     float,
    threshold2:     float,
    threshold_joint: float,
    k1:    float | None = None,
    k2:    float | None = None,
    a_values1: np.ndarray | None = None,
    a_values2: np.ndarray | None = None,
    b_values:  np.ndarray | None = None,
    refine:      bool = True,
    max_signals: int  = 50,
) -> dict:
    """
    Ricerca greedy iterativa di segnali exp-drop simultanei in due serie.

    Strategia
    ---------
    1. Calcola profili LLR per-b per entrambe le serie sui residui correnti.
    2. Costruisce profilo congiunto: joint(t0) = max_b [LLR1_b(t0) + LLR2_b(t0)].
    3. Sceglie t0* che massimizza max(joint/thr_j, LLR1/thr1, LLR2/thr2).
    4. Classifica il segnale in base alle soglie individuali e congiunta.
    5. Affina (a, b) con Nelder-Mead per i canali coinvolti.
    6. Sottrae il contributo anomalo dai residui; ripete dal passo 1.

    Parametri
    ---------
    data1, data2     : array 1D, stessa lunghezza e campionamento
    threshold1/2     : soglie LLR individuali (da calibrate_thresholds_dual)
    threshold_joint  : soglia LLR congiunta
    k1, k2           : fondi (default: media dei dati)
    a_values1/2      : griglie di a per ciascuna serie
    b_values         : griglia di b condivisa tra le due serie
    refine           : abilita raffinamento Nelder-Mead
    max_signals      : limite iterazioni greedy

    Ritorna
    -------
    dict con chiavi:
        "both"       : segnali trovati individualmente in entrambe le serie
        "joint_only" : segnali significativi solo per il test congiunto
        "only1"      : segnali trovati solo nella serie 1
        "only2"      : segnali trovati solo nella serie 2

    Ogni elemento è un dict con:
        t0, k1, k2,
        a1, b1, llr1, a1_err, b1_err   (NaN se canale non rilevato)
        a2, b2, llr2, a2_err, b2_err   (NaN se canale non rilevato)
        category
    """
    data1 = np.asarray(data1, dtype=np.float64)
    data2 = np.asarray(data2, dtype=np.float64)
    N = len(data1)
    if len(data2) != N:
        raise ValueError(f"Le serie devono avere la stessa lunghezza ({N} ≠ {len(data2)})")

    k1 = float(data1.mean()) if k1 is None else float(k1)
    k2 = float(data2.mean()) if k2 is None else float(k2)

    if a_values1 is None: a_values1, _bv = _default_grids(k1)
    if a_values2 is None: a_values2, _   = _default_grids(k2)
    if b_values  is None: _, b_values    = _default_grids(k1)

    a_values1 = np.asarray(a_values1, dtype=np.float64)
    a_values2 = np.asarray(a_values2, dtype=np.float64)
    b_values  = np.asarray(b_values,  dtype=np.float64)

    t   = np.arange(N, dtype=np.float64)
    res1 = data1.copy()
    res2 = data2.copy()

    found: list[dict] = []

    for _ in range(max_signals):
        # profili per-b
        prof1_b, ba1 = _llr_profile_perbval(res1, k1, a_values1, b_values)
        prof2_b, ba2 = _llr_profile_perbval(res2, k2, a_values2, b_values)

        # profili marginali (max su b)
        prof1 = prof1_b.max(axis=1)   # (N,)
        prof2 = prof2_b.max(axis=1)   # (N,)

        # profilo congiunto con b condiviso
        joint_b = prof1_b + prof2_b   # (N, Nb)
        joint   = joint_b.max(axis=1) # (N,)

        # score normalizzato: rileva chi supera prima la propria soglia
        score = np.maximum(
            joint / threshold_joint,
            np.maximum(prof1 / threshold1, prof2 / threshold2),
        )
        t0_best = int(np.argmax(score))

        in1      = bool(prof1[t0_best]  >= threshold1)
        in2      = bool(prof2[t0_best]  >= threshold2)
        joint_ok = bool(joint[t0_best]  >= threshold_joint)

        if not (in1 or in2 or joint_ok):
            break   # nessun segnale significativo

        # b ottimo per il test congiunto
        b_idx = int(np.argmax(joint_b[t0_best]))
        b0    = float(b_values[b_idx])

        # a0 iniziali per il raffinamento
        a0_1 = float(ba1[t0_best, b_idx])
        a0_2 = float(ba2[t0_best, b_idx])

        # classifica
        if   in1 and in2:
            category = "both"
        elif joint_ok:
            category = "joint_only"
        elif in1:
            category = "only1"
        else:
            category = "only2"

        # raffinamento e sottrazione per i canali coinvolti
        fit1 = fit2 = None
        fit_ch1 = in1 or joint_ok
        fit_ch2 = in2 or joint_ok

        if fit_ch1 and refine:
            fit1 = _refine(res1, t0_best, k1, max(a0_1, 0.1), b0)
        elif fit_ch1:
            fit1 = {"a": a0_1, "b": b0, "llr": float(prof1[t0_best]),
                    "a_err": np.nan, "b_err": np.nan}

        if fit_ch2 and refine:
            fit2 = _refine(res2, t0_best, k2, max(a0_2, 0.1), b0)
        elif fit_ch2:
            fit2 = {"a": a0_2, "b": b0, "llr": float(prof2[t0_best]),
                    "a_err": np.nan, "b_err": np.nan}

        _nan = {"a": np.nan, "b": np.nan, "llr": np.nan, "a_err": np.nan, "b_err": np.nan}
        if fit1 is None: fit1 = _nan
        if fit2 is None: fit2 = _nan

        found.append({
            "t0":      t0_best,
            "k1":      k1,       "k2":      k2,
            "a1":      fit1["a"], "b1":      fit1["b"], "llr1":   fit1["llr"],
            "a1_err":  fit1["a_err"], "b1_err": fit1["b_err"],
            "a2":      fit2["a"], "b2":      fit2["b"], "llr2":   fit2["llr"],
            "a2_err":  fit2["a_err"], "b2_err": fit2["b_err"],
            "category": category,
        })

        # sottrai contributo anomalo dai residui
        if fit_ch1 and np.isfinite(fit1["a"]):
            mu1   = _mu(t, float(t0_best), k1, fit1["a"], fit1["b"])
            res1 -= mu1 - k1
        if fit_ch2 and np.isfinite(fit2["a"]):
            mu2   = _mu(t, float(t0_best), k2, fit2["a"], fit2["b"])
            res2 -= mu2 - k2

    found.sort(key=lambda r: r["t0"])

    return {
        "both":       [r for r in found if r["category"] == "both"],
        "joint_only": [r for r in found if r["category"] == "joint_only"],
        "only1":      [r for r in found if r["category"] == "only1"],
        "only2":      [r for r in found if r["category"] == "only2"],
    }


# ── stile matplotlib ──────────────────────────────────────────────────────────

def _publication_style() -> None:
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "Georgia"],
        "font.size":          11,
        "axes.labelsize":     12,
        "axes.titlesize":     11,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    9,
        "lines.linewidth":    1.3,
        "axes.linewidth":     0.8,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.top":          True,
        "ytick.right":        True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.grid":          True,
        "grid.color":         "0.82",
        "grid.linewidth":     0.5,
        "grid.linestyle":     "--",
        "figure.constrained_layout.use": True,
        "savefig.bbox":       "tight",
    })


# palette colori per categoria
_CAT_COLOR = {
    "both":       "C0",   # blu
    "joint_only": "C4",   # viola
    "only1":      "C2",   # verde
    "only2":      "C3",   # rosso
}
_CAT_LABEL = {
    "both":       "Both channels",
    "joint_only": "Joint only",
    "only1":      "Series 1 only",
    "only2":      "Series 2 only",
}


# ── grafici ───────────────────────────────────────────────────────────────────

def plot_dual_results(
    data1:    np.ndarray,
    data2:    np.ndarray,
    results:  dict,
    plot_path: str | Path,
    dpi:      int   = 300,
    label1:   str   = "Series 1",
    label2:   str   = "Series 2",
    k1:       float | None = None,
    k2:       float | None = None,
) -> None:
    """
    Produce figura da pubblicazione con 5 pannelli e la salva in ``plot_path``.

    Pannelli
    --------
    ax1 : serie 1 con segnali annotati per categoria
    ax2 : serie 2 con segnali annotati per categoria
    ax3 : scatter a1 vs a2 per segnali "both" e "joint_only"
    ax4 : scatter b1 vs b2 per segnali "both" e "joint_only"
    ax5 : conteggio segnali per categoria (barre)
    """
    _publication_style()

    data1 = np.asarray(data1, dtype=np.float64)
    data2 = np.asarray(data2, dtype=np.float64)
    N     = len(data1)
    t     = np.arange(N)

    k1 = float(data1.mean()) if k1 is None else float(k1)
    k2 = float(data2.mean()) if k2 is None else float(k2)

    all_sigs = (results.get("both", []) + results.get("joint_only", []) +
                results.get("only1", []) + results.get("only2", []))

    # ── layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(3, 3, figure=fig,
                   height_ratios=[1, 1, 1.1],
                   hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])   # serie 1 (top, full width)
    ax2 = fig.add_subplot(gs[1, :])   # serie 2 (middle, full width)
    ax3 = fig.add_subplot(gs[2, 0])   # scatter a1 vs a2
    ax4 = fig.add_subplot(gs[2, 1])   # scatter b1 vs b2
    ax5 = fig.add_subplot(gs[2, 2])   # conteggi per categoria

    def _draw_timeseries(ax, data, k_val, label, chan_key_own, chan_key_both):
        """Disegna una serie con i segnali colorati per categoria."""
        ax.step(t, data, color="0.65", lw=0.5, where="mid", zorder=1)
        ax.axhline(k_val, color="k", ls="--", lw=0.9, zorder=2,
                   label=f"Background $k={k_val:.0f}$")
        for cat, sigs in results.items():
            col = _CAT_COLOR[cat]
            for s in sigs:
                # canale 1 vede tutti tranne only2, e viceversa
                if chan_key_own == "1" and cat == "only2":
                    continue
                if chan_key_own == "2" and cat == "only1":
                    continue
                ax.axvline(s["t0"], color=col, lw=1.6, alpha=0.75, zorder=3)
        ax.set_xlabel("Sample index $t$")
        ax.set_ylabel("Counts")
        ax.set_xlim(0, N - 1)
        ax.set_title(label)
        # legenda con patch per categoria
        handles = [Line2D([0], [0], color="0.65",  lw=1,   label="Data"),
                   Line2D([0], [0], color="k",     lw=0.9, ls="--", label=f"$k={k_val:.0f}$")]
        for cat in ["both", "joint_only", "only1" if chan_key_own == "1" else "only2"]:
            if results.get(cat):
                handles.append(Line2D([0], [0], color=_CAT_COLOR[cat], lw=1.6,
                                      label=_CAT_LABEL[cat]))
        ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=8.5)

    _draw_timeseries(ax1, data1, k1, label1, "1", "both")
    _draw_timeseries(ax2, data2, k2, label2, "2", "both")

    # ── scatter a1 vs a2 ──────────────────────────────────────────────────────
    for cat in ("both", "joint_only"):
        sigs = results.get(cat, [])
        if not sigs:
            continue
        a1v = [s["a1"] for s in sigs if np.isfinite(s["a1"])]
        a2v = [s["a2"] for s in sigs if np.isfinite(s["a2"])]
        if a1v and a2v:
            ax3.scatter(a1v, a2v, s=30, color=_CAT_COLOR[cat],
                        alpha=0.7, edgecolors="none", label=_CAT_LABEL[cat])

    if ax3.collections:
        all_a = [s["a1"] for s in all_sigs if np.isfinite(s.get("a1", np.nan))] + \
                [s["a2"] for s in all_sigs if np.isfinite(s.get("a2", np.nan))]
        if all_a:
            lim = [min(all_a) * 0.7, max(all_a) * 1.4]
            ax3.plot(lim, lim, "k--", lw=0.8)
            ax3.set_xlim(lim); ax3.set_ylim(lim)
    ax3.set_xlabel(r"Amplitude $a_1$ (series 1)")
    ax3.set_ylabel(r"Amplitude $a_2$ (series 2)")
    ax3.set_title(r"Parameter recovery: $a$")
    if ax3.collections:
        ax3.legend(fontsize=8, framealpha=0.9)

    # ── scatter b1 vs b2 ──────────────────────────────────────────────────────
    for cat in ("both", "joint_only"):
        sigs = results.get(cat, [])
        if not sigs:
            continue
        b1v = [s["b1"] for s in sigs if np.isfinite(s.get("b1", np.nan))]
        b2v = [s["b2"] for s in sigs if np.isfinite(s.get("b2", np.nan))]
        if b1v and b2v:
            ax4.scatter(b1v, b2v, s=30, color=_CAT_COLOR[cat],
                        alpha=0.7, edgecolors="none", label=_CAT_LABEL[cat])

    if ax4.collections:
        all_b = [s["b1"] for s in all_sigs if np.isfinite(s.get("b1", np.nan))] + \
                [s["b2"] for s in all_sigs if np.isfinite(s.get("b2", np.nan))]
        if all_b:
            lim = [min(all_b) * 0.7, max(all_b) * 1.4]
            ax4.plot(lim, lim, "k--", lw=0.8)
            ax4.set_xlim(lim); ax4.set_ylim(lim)
            ax4.set_xscale("log"); ax4.set_yscale("log")
    ax4.set_xlabel(r"Timescale $b_1$ (series 1)")
    ax4.set_ylabel(r"Timescale $b_2$ (series 2)")
    ax4.set_title(r"Parameter recovery: $b$")
    if ax4.collections:
        ax4.legend(fontsize=8, framealpha=0.9)

    # ── barre per categoria ───────────────────────────────────────────────────
    cats   = ["both", "joint_only", "only1", "only2"]
    counts = [len(results.get(c, [])) for c in cats]
    colors = [_CAT_COLOR[c] for c in cats]
    xlbls  = ["Both", "Joint\nonly", "Only 1", "Only 2"]
    bars   = ax5.bar(xlbls, counts, color=colors, edgecolor="k", linewidth=0.6)
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(cnt), ha="center", va="bottom", fontsize=10)
    ax5.set_ylabel("Number of detections")
    ax5.set_title("Detections by category")
    ax5.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── titolo generale ───────────────────────────────────────────────────────
    n_tot = sum(counts)
    n_sim = len(results.get("both", [])) + len(results.get("joint_only", []))
    fig.suptitle(
        f"Dual-channel signal search  |  {n_tot} detections total  |  "
        f"{n_sim} simultaneous  ({label1} / {label2})",
        fontsize=12,
    )

    fig.savefig(plot_path, dpi=dpi)
    plt.close(fig)


# ── report ASCII ──────────────────────────────────────────────────────────────

def write_dual_report(
    results:     dict,
    report_path: str | Path,
    thresholds:  tuple[float, float, float] | None = None,
    label1:      str   = "Series 1",
    label2:      str   = "Series 2",
    extra_info:  dict  | None = None,
) -> None:
    """
    Scrive report ASCII dettagliato in ``report_path``.

    Parametri
    ---------
    results     : dict restituito da find_signals_dual()
    thresholds  : (thr1, thr2, thr_joint) — se None non vengono stampate
    label1/2    : etichette delle serie
    extra_info  : dict con info aggiuntive opzionali (es. noise type, k, N)
    """
    W   = 82
    SEP = "=" * W
    sep = "-" * W
    ts  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    cats   = ["both", "joint_only", "only1", "only2"]
    counts = {c: len(results.get(c, [])) for c in cats}
    n_tot  = sum(counts.values())

    lines: list[str] = [
        SEP,
        "  DUAL-CHANNEL SIGNAL DETECTION REPORT",
        f"  Generated : {ts}",
        SEP, "",
    ]

    # --- info opzionali ---
    if extra_info:
        lines += ["DATASET INFO", sep]
        for k, v in extra_info.items():
            lines.append(f"  {k:<26}: {v}")
        lines.append("")

    if thresholds:
        lines += [
            "THRESHOLDS",
            sep,
            f"  LLR threshold {label1:<18}: {thresholds[0]:.4f}",
            f"  LLR threshold {label2:<18}: {thresholds[1]:.4f}",
            f"  LLR threshold joint         : {thresholds[2]:.4f}",
            "",
        ]

    # --- riepilogo ---
    lines += [
        "SUMMARY",
        sep,
        f"  Total detections            : {n_tot}",
        f"  Both channels (individual)  : {counts['both']}",
        f"  Both channels (joint only)  : {counts['joint_only']}",
        f"  {label1} only{'':<14}: {counts['only1']}",
        f"  {label2} only{'':<14}: {counts['only2']}",
        "",
    ]

    # --- tabella per categoria ---
    def _fmt_table(cat: str, sigs: list[dict]) -> list[str]:
        if not sigs:
            return [f"  (none)", ""]

        both_chans = cat in ("both", "joint_only")

        if both_chans:
            hdr = (f"  {'#':>4}  {'t0':>8}  "
                   f"{'a1':>8}  {'±':>6}  {'b1':>7}  {'±':>6}  {'llr1':>7}  "
                   f"{'a2':>8}  {'±':>6}  {'b2':>7}  {'±':>6}  {'llr2':>7}")
            row_sep = "  " + "-" * (len(hdr) - 2)
            rows = [hdr, row_sep]
            for i, s in enumerate(sigs):
                def _f(v): return f"{v:8.3f}" if np.isfinite(v) else "     nan"
                def _fe(v): return f"{v:6.2f}" if np.isfinite(v) else "   nan"
                rows.append(
                    f"  {i+1:>4}  {s['t0']:>8}  "
                    f"{_f(s['a1'])}  {_fe(s['a1_err'])}  "
                    f"{_f(s['b1']):>7}  {_fe(s['b1_err'])}  "
                    f"{_f(s['llr1']):>7}  "
                    f"{_f(s['a2'])}  {_fe(s['a2_err'])}  "
                    f"{_f(s['b2']):>7}  {_fe(s['b2_err'])}  "
                    f"{_f(s['llr2']):>7}"
                )
        else:
            ak  = "a1" if cat == "only1" else "a2"
            bk  = "b1" if cat == "only1" else "b2"
            lk  = "llr1" if cat == "only1" else "llr2"
            aek = "a1_err" if cat == "only1" else "a2_err"
            bek = "b1_err" if cat == "only1" else "b2_err"
            hdr = (f"  {'#':>4}  {'t0':>8}  {'a':>8}  {'±':>6}  "
                   f"{'b':>8}  {'±':>6}  {'llr':>8}")
            row_sep = "  " + "-" * (len(hdr) - 2)
            rows = [hdr, row_sep]
            for i, s in enumerate(sigs):
                def _f(v): return f"{v:8.3f}" if np.isfinite(v) else "     nan"
                def _fe(v): return f"{v:6.2f}" if np.isfinite(v) else "   nan"
                rows.append(
                    f"  {i+1:>4}  {s['t0']:>8}  "
                    f"{_f(s[ak])}  {_fe(s[aek])}  "
                    f"{_f(s[bk]):>8}  {_fe(s[bek])}  "
                    f"{_f(s[lk]):>8}"
                )
        return rows + [""]

    section_titles = {
        "both":       f"SIMULTANEOUS SIGNALS — both channels individually (N={counts['both']})",
        "joint_only": f"SIMULTANEOUS SIGNALS — joint test only (N={counts['joint_only']})",
        "only1":      f"SIGNALS ONLY IN {label1.upper()} (N={counts['only1']})",
        "only2":      f"SIGNALS ONLY IN {label2.upper()} (N={counts['only2']})",
    }

    for cat in cats:
        lines += [section_titles[cat], sep]
        lines += _fmt_table(cat, results.get(cat, []))

    lines += [SEP, "  END OF REPORT", SEP, ""]

    Path(report_path).write_text("\n".join(lines), encoding="utf-8")
