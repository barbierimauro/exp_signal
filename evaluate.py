"""
evaluate.py
===========
Valuta le prestazioni del detector ``find_signals`` su serie sintetiche
generate da ``simulate.generate_data``.

Per ogni realizzazione:
  1. genera dati con rumore e segnali iniettati (``generate_data``)
  2. esegue ``find_signals`` con soglia calibrata
  3. abbina segnali trovati a quelli veri (matching per t0)
  4. accumula statistiche

Output prodotti
---------------
  - grafici publication-quality (PDF/PNG configurabile)
  - report ASCII con statistiche globali e lista detections / missed

Funzioni pubbliche
------------------
    run_evaluation(config_path)  -> dict  (statistiche riepilogative)

Uso da riga di comando
----------------------
    python evaluate.py [simulation_config.yaml]
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # non serve display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import beta as _beta_dist

from main import calibrate_threshold, find_signals
from simulate import load_config, generate_data


# ── stile matplotlib ──────────────────────────────────────────────────────────

def _publication_style() -> None:
    """Configura rcParams per grafici da pubblicazione scientifica."""
    plt.rcParams.update({
        # font
        "font.family":          "serif",
        "font.serif":           ["Times New Roman", "DejaVu Serif", "Georgia"],
        "font.size":            11,
        "axes.labelsize":       12,
        "axes.titlesize":       12,
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        "legend.fontsize":      9.5,
        # linee e assi
        "lines.linewidth":      1.3,
        "axes.linewidth":       0.8,
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "xtick.top":            True,
        "ytick.right":          True,
        "xtick.minor.visible":  True,
        "ytick.minor.visible":  True,
        # griglia
        "axes.grid":            True,
        "grid.color":           "0.82",
        "grid.linewidth":       0.5,
        "grid.linestyle":       "--",
        # layout
        "figure.constrained_layout.use": True,
        "savefig.bbox":         "tight",
    })


# ── matching segnali ──────────────────────────────────────────────────────────

def _match_signals(
    true_signals: list[dict],
    detected_signals: list[dict],
    tol_t0: int | None = None,
) -> tuple[list[tuple], list[dict], list[dict]]:
    """
    Abbina segnali veri a detectati per prossimità in t0.

    Un segnale vero è "trovato" se esiste un detected con
        |t0_det − t0_true| <= max(10, 3 * b_true)   (o tol_t0 se fornito)
    Ogni detected viene usato al più una volta (matching greedy per dt minimo).

    Ritorna
    -------
    matches       : lista di (true_dict, det_dict)
    missed_sigs   : segnali veri non trovati
    false_pos     : detections senza segnale vero corrispondente
    """
    matched_true = set()
    matched_det  = set()
    matches: list[tuple] = []

    for i, sig in enumerate(true_signals):
        t0_true = sig["t0"]
        b_true  = sig["b"]
        tol     = tol_t0 if tol_t0 is not None else max(10, int(3 * b_true))

        best_j  = None
        best_dt = np.inf
        for j, det in enumerate(detected_signals):
            if j in matched_det:
                continue
            dt = abs(det["t0"] - t0_true)
            if dt <= tol and dt < best_dt:
                best_j  = j
                best_dt = dt

        if best_j is not None:
            matches.append((sig, detected_signals[best_j]))
            matched_true.add(i)
            matched_det.add(best_j)

    missed_sigs = [true_signals[i]    for i in range(len(true_signals))    if i not in matched_true]
    false_pos   = [detected_signals[j] for j in range(len(detected_signals)) if j not in matched_det]
    return matches, missed_sigs, false_pos


# ── intervallo di Clopper-Pearson al 68% ─────────────────────────────────────

def _cp68(k: int, n: int) -> tuple[float, float]:
    """Intervallo di confidenza binomiale 68% (Clopper-Pearson)."""
    alpha = 0.32
    lo = float(_beta_dist.ppf(alpha / 2, k,     n - k + 1)) if k > 0 else 0.0
    hi = float(_beta_dist.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def _efficiency_with_errors(
    true_vals:  list[float],
    found_vals: list[float],
    bins:       np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcola efficienza e IC68% per bin logaritmici.

    Ritorna
    -------
    centers, eff, err_lo, err_hi
    """
    n_true,  _ = np.histogram(true_vals,  bins=bins)
    n_found, _ = np.histogram(found_vals, bins=bins)
    centers = np.sqrt(bins[:-1] * bins[1:])

    eff    = np.full(len(centers), np.nan)
    err_lo = np.full(len(centers), 0.0)
    err_hi = np.full(len(centers), 0.0)

    for i, (nt, nf) in enumerate(zip(n_true, n_found)):
        if nt == 0:
            continue
        nf_clip = min(int(nf), int(nt))      # per robustezza numerica
        eff[i] = nf_clip / nt
        lo, hi = _cp68(nf_clip, int(nt))
        err_lo[i] = eff[i] - lo
        err_hi[i] = hi - eff[i]

    return centers, eff, err_lo, err_hi


# ── grafici ───────────────────────────────────────────────────────────────────

def _make_plots(
    results: list[dict],
    all_matches: list[tuple],
    config: dict,
    plot_path: Path,
    dpi: int,
) -> None:
    """Produce figura da pubblicazione e la salva in ``plot_path``."""
    _publication_style()

    # --- raccogli dati aggregati ---
    true_a  = [s["a"]    for r in results for s in r["true"]]
    true_b  = [s["b"]    for r in results for s in r["true"]]
    found_a = [m[0]["a"] for m in all_matches]
    found_b = [m[0]["b"] for m in all_matches]
    det_a   = [m[1]["a"] for m in all_matches]
    det_b   = [m[1]["b"] for m in all_matches]

    # --- layout ---
    fig = plt.figure(figsize=(13, 9))
    gs  = GridSpec(2, 3, figure=fig, height_ratios=[1.4, 1],
                   hspace=0.40, wspace=0.35)

    k_val      = config["background"]["k"]
    noise_type = config["noise"]["type"]
    n_real     = len(results)
    n_true_tot = sum(len(r["true"]) for r in results)
    n_found_tot = sum(r["n_found"]  for r in results)
    global_eff  = n_found_tot / n_true_tot if n_true_tot > 0 else 0.0

    # ── pannello 1: esempio di realizzazione ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ex   = results[0]
    data_ex = ex["data"]
    t_ex    = np.arange(len(data_ex))

    ax1.step(t_ex, data_ex, color="0.65", lw=0.6, where="mid", label="Data", zorder=1)
    ax1.axhline(k_val, color="k", ls="--", lw=1.0, label=f"Background $k = {k_val:.0f}$", zorder=2)

    # segnali veri
    for s in ex["true"]:
        ax1.axvline(s["t0"], color="C0", lw=1.8, ls="-", alpha=0.75, zorder=3)

    # segnali detectati
    for d in ex["detected"]:
        ax1.axvline(d["t0"], color="C3", lw=1.6, ls="--", alpha=0.85, zorder=4)

    ax1.set_xlabel("Sample index $t$")
    ax1.set_ylabel("Counts")
    ax1.set_xlim(0, len(data_ex) - 1)
    ax1.set_title(
        f"Example realization — noise: {noise_type} — "
        f"{len(ex['true'])} injected / {len(ex['detected'])} detected",
        pad=6,
    )
    legend_handles = [
        Line2D([0], [0], color="0.65", lw=1.2, label="Data"),
        Line2D([0], [0], color="k",   lw=1.0, ls="--", label="Background"),
        Line2D([0], [0], color="C0",  lw=1.8, label="True $t_0$"),
        Line2D([0], [0], color="C3",  lw=1.6, ls="--", label="Detected $t_0$"),
    ]
    ax1.legend(handles=legend_handles, loc="upper right", framealpha=0.92)

    # ── pannello 2: efficienza vs ampiezza a ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    if true_a and found_a:
        a_bins = np.geomspace(min(true_a) * 0.85, max(true_a) * 1.15, 10)
        ctrs, eff, elo, ehi = _efficiency_with_errors(true_a, found_a, a_bins)
        valid = ~np.isnan(eff)
        ax2.errorbar(
            ctrs[valid], eff[valid],
            yerr=[elo[valid], ehi[valid]],
            fmt="o-", color="C0", ms=5, capsize=3, lw=1.3,
            label="Efficiency (68% CI)",
        )
        ax2.fill_between(ctrs[valid], eff[valid] - elo[valid], eff[valid] + ehi[valid],
                         alpha=0.15, color="C0")
    ax2.axhline(global_eff, color="C3", ls=":", lw=1.1, label=f"Global eff. {global_eff:.2f}")
    ax2.set_xscale("log")
    ax2.set_xlabel("Signal amplitude $a$ (counts)")
    ax2.set_ylabel("Detection efficiency")
    ax2.set_ylim(-0.05, 1.10)
    ax2.set_title("Efficiency vs amplitude")
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    # ── pannello 3: efficienza vs scala temporale b ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    if true_b and found_b:
        b_bins = np.geomspace(min(true_b) * 0.85, max(true_b) * 1.15, 10)
        ctrs, eff, elo, ehi = _efficiency_with_errors(true_b, found_b, b_bins)
        valid = ~np.isnan(eff)
        ax3.errorbar(
            ctrs[valid], eff[valid],
            yerr=[elo[valid], ehi[valid]],
            fmt="s-", color="C1", ms=5, capsize=3, lw=1.3,
        )
        ax3.fill_between(ctrs[valid], eff[valid] - elo[valid], eff[valid] + ehi[valid],
                         alpha=0.15, color="C1")
    ax3.axhline(global_eff, color="C3", ls=":", lw=1.1)
    ax3.set_xscale("log")
    ax3.set_xlabel("Signal timescale $b$ (samples)")
    ax3.set_ylabel("Detection efficiency")
    ax3.set_ylim(-0.05, 1.10)
    ax3.set_title("Efficiency vs timescale")
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    # ── pannello 4: recupero parametri ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    if found_a and det_a:
        sc = ax4.scatter(
            found_a, det_a,
            c=np.log10(np.array(found_b) + 1),
            cmap="viridis", s=20, alpha=0.65, edgecolors="none",
            label="Detections",
        )
        cb = fig.colorbar(sc, ax=ax4, pad=0.02)
        cb.set_label(r"$\log_{10}(b_\mathrm{true}+1)$", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        lims = [
            min(min(found_a), min(det_a)) * 0.75,
            max(max(found_a), max(det_a)) * 1.30,
        ]
        ax4.plot(lims, lims, "k--", lw=0.9, label="1:1")
        ax4.set_xlim(lims)
        ax4.set_ylim(lims)
        ax4.set_xscale("log")
        ax4.set_yscale("log")
    ax4.set_xlabel(r"True amplitude $a$")
    ax4.set_ylabel(r"Recovered amplitude $\hat{a}$")
    ax4.set_title("Parameter recovery")
    ax4.legend(loc="upper left", framealpha=0.9)

    # ── titolo generale ───────────────────────────────────────────────────────
    fig.suptitle(
        f"Signal detection evaluation  |  noise: {noise_type}  |  "
        f"$k = {k_val:.0f}$  |  "
        f"{n_real} realizations  |  "
        f"global efficiency: {global_eff:.3f}",
        fontsize=12,
    )

    fig.savefig(plot_path, dpi=dpi)
    plt.close(fig)


# ── report ASCII ──────────────────────────────────────────────────────────────

def _write_report(
    results: list[dict],
    all_matches: list[tuple],
    config: dict,
    report_path: Path,
    threshold: float,
) -> None:
    """Scrive il report ASCII di valutazione in ``report_path``."""
    n_real      = len(results)
    n_true_tot  = sum(len(r["true"])     for r in results)
    n_found_tot = sum(r["n_found"]       for r in results)
    n_miss_tot  = sum(r["n_missed"]      for r in results)
    n_fp_tot    = sum(r["n_false_pos"]   for r in results)
    eff = n_found_tot / n_true_tot if n_true_tot > 0 else 0.0
    fp_per_real = n_fp_tot / n_real if n_real > 0 else 0.0

    W   = 80
    SEP = "=" * W
    sep = "-" * W

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = [
        SEP,
        "  SIGNAL DETECTION EVALUATION REPORT",
        f"  Generated : {ts}",
        SEP,
        "",
        "CONFIGURATION",
        sep,
        f"  Noise type            : {config['noise']['type']}",
        f"  Series length         : {config['series']['length']} samples",
        f"  Background k          : {config['background']['k']}",
        f"  Realizations          : {n_real}",
        f"  LLR threshold         : {threshold:.4f}",
        f"  FPR target (global)   : {config['detection']['fpr']}",
        f"  Signal a range        : [{config['signals']['a']['min']}, {config['signals']['a']['max']}]",
        f"  Signal b range        : [{config['signals']['b']['min']}, {config['signals']['b']['max']}]",
        "",
        "GLOBAL STATISTICS",
        sep,
        f"  Total injected signals : {n_true_tot}",
        f"  Total detected         : {n_found_tot}",
        f"  Total missed           : {n_miss_tot}",
        f"  False positives (total): {n_fp_tot}",
        f"  Detection efficiency   : {eff:.4f}  ({100*eff:.1f}%)",
        f"  False pos / realization: {fp_per_real:.3f}",
        "",
        "PER-REALIZATION SUMMARY",
        sep,
        f"  {'Real':>5}  {'N_inj':>6}  {'N_det':>5}  {'N_miss':>6}  "
        f"{'N_fp':>4}  {'Eff':>7}",
        sep,
    ]

    for i, r in enumerate(results):
        nt  = len(r["true"])
        nd  = r["n_found"]
        nm  = r["n_missed"]
        nfp = r["n_false_pos"]
        ef  = nd / nt if nt > 0 else 0.0
        lines.append(
            f"  {i+1:>5}  {nt:>6}  {nd:>5}  {nm:>6}  {nfp:>4}  {ef:>7.3f}"
        )

    lines += [
        sep,
        "",
        f"MATCHED DETECTIONS  (showing first {min(100, len(all_matches))} of {len(all_matches)})",
        sep,
        f"  {'#':>5}  {'t0_true':>8}  {'a_true':>8}  {'b_true':>8}  "
        f"{'t0_det':>7}  {'a_det':>8}  {'b_det':>8}  {'LLR':>8}  {'dt0':>5}",
        sep,
    ]
    for i, (tr, dt) in enumerate(all_matches[:100]):
        dt0 = dt["t0"] - tr["t0"]
        lines.append(
            f"  {i+1:>5}  {tr['t0']:>8}  {tr['a']:>8.3f}  {tr['b']:>8.3f}  "
            f"{dt['t0']:>7}  {dt['a']:>8.3f}  {dt['b']:>8.3f}  "
            f"{dt['llr']:>8.3f}  {dt0:>+5}"
        )
    if len(all_matches) > 100:
        lines.append(f"  ... ({len(all_matches) - 100} additional detections omitted)")

    all_missed = [(s, ) for r in results for s in r["missed_sigs"]]
    lines += [
        sep,
        "",
        f"MISSED SIGNALS  (showing first {min(100, len(all_missed))} of {len(all_missed)})",
        sep,
        f"  {'#':>5}  {'t0_true':>8}  {'a_true':>8}  {'b_true':>8}  "
        f"{'a/sqrt(k)':>10}  {'a/k':>6}",
        sep,
    ]
    k_val = config["background"]["k"]
    for i, (s,) in enumerate(all_missed[:100]):
        snr = s["a"] / np.sqrt(k_val)
        rel = s["a"] / k_val
        lines.append(
            f"  {i+1:>5}  {s['t0']:>8}  {s['a']:>8.3f}  {s['b']:>8.3f}  "
            f"{snr:>10.3f}  {rel:>6.3f}"
        )
    if len(all_missed) > 100:
        lines.append(f"  ... ({len(all_missed) - 100} additional missed signals omitted)")

    lines += ["", SEP, "  END OF REPORT", SEP, ""]

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ── funzione principale ───────────────────────────────────────────────────────

def run_evaluation(config_path: str | Path) -> dict:
    """
    Esegue la valutazione completa del detector su dati sintetici.

    Parametri
    ---------
    config_path : percorso del file YAML di configurazione

    Ritorna
    -------
    dict con chiavi:
        threshold, n_realizations, n_true, n_found, n_missed, efficiency
    """
    config_path = Path(config_path)
    config      = load_config(config_path)

    N          = int(config["series"]["length"])
    k          = float(config["background"]["k"])
    n_real     = int(config["series"]["n_realizations"])
    det_cfg    = config["detection"]
    out_cfg    = config["output"]
    seed0      = int(det_cfg.get("seed", 42))
    dt_seconds = float(config["series"].get("dt_seconds", 3600.0))
    tau_rise_max = float(det_cfg.get("tau_rise_max_samples", 0.0))
    k_method   = det_cfg.get("k_method", "sigma_clip")

    # ── calibrazione soglia ───────────────────────────────────────────────────
    print(f"[evaluate] Calibrazione soglia  k={k}  N={N}  "
          f"n_sim={det_cfg['n_sim']}  n_cal={det_cfg['n_cal']} ...")
    threshold = calibrate_threshold(
        k          = k,
        N          = N,
        n_cal      = int(det_cfg["n_cal"]),
        fpr        = float(det_cfg["fpr"]),
        n_sim      = int(det_cfg["n_sim"]),
        seed       = seed0,
        dt_seconds = dt_seconds,
    )
    print(f"[evaluate] Soglia LLR = {threshold:.4f}")

    # ── loop sulle realizzazioni ──────────────────────────────────────────────
    results: list[dict] = []
    all_matches: list[tuple] = []

    for i in range(n_real):
        seed_i = seed0 + i + 1
        data, true_signals = generate_data(config, seed=seed_i)
        detected           = find_signals(
            data, threshold=threshold, k=k,
            dt_seconds=dt_seconds, tau_rise_max=tau_rise_max, k_method=k_method,
        )

        matches, missed_sigs, false_pos = _match_signals(true_signals, detected)

        results.append({
            "data":        data if i == 0 else None,   # salva dati solo per l'esempio
            "true":        true_signals,
            "detected":    detected,
            "n_found":     len(matches),
            "n_missed":    len(missed_sigs),
            "missed_sigs": missed_sigs,
            "n_false_pos": len(false_pos),
        })
        all_matches.extend(matches)

        if (i + 1) % max(1, n_real // 10) == 0 or i == 0:
            cum_eff = (sum(r["n_found"] for r in results) /
                       max(1, sum(len(r["true"]) for r in results)))
            print(f"[evaluate]  {i+1:>4}/{n_real}  "
                  f"inj={len(true_signals)}  det={len(detected)}  "
                  f"found={len(matches)}  miss={len(missed_sigs)}  "
                  f"cum_eff={cum_eff:.3f}")

    # ── statistiche globali ───────────────────────────────────────────────────
    n_true_tot  = sum(len(r["true"])   for r in results)
    n_found_tot = sum(r["n_found"]     for r in results)
    n_miss_tot  = sum(r["n_missed"]    for r in results)
    eff = n_found_tot / n_true_tot if n_true_tot > 0 else 0.0

    print(f"\n[evaluate] ── RISULTATI ──────────────────────────────")
    print(f"           Segnali iniettati : {n_true_tot}")
    print(f"           Segnali trovati   : {n_found_tot}")
    print(f"           Segnali mancati   : {n_miss_tot}")
    print(f"           Efficienza globale: {eff:.4f}  ({100*eff:.1f}%)")

    # ── grafici ───────────────────────────────────────────────────────────────
    plot_path  = Path(out_cfg["plot_file"])
    dpi        = int(out_cfg.get("dpi", 300))
    print(f"\n[evaluate] Salvo grafici → {plot_path}")
    _make_plots(results, all_matches, config, plot_path, dpi)

    # ── report ────────────────────────────────────────────────────────────────
    report_path = Path(out_cfg["report_file"])
    print(f"[evaluate] Scrivo report  → {report_path}")
    _write_report(results, all_matches, config, report_path, threshold)

    stats = {
        "threshold":      threshold,
        "n_realizations": n_real,
        "n_true":         n_true_tot,
        "n_found":        n_found_tot,
        "n_missed":       n_miss_tot,
        "efficiency":     eff,
    }
    return stats


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "simulation_config.yaml"
    stats = run_evaluation(cfg)
    print("\nStats:", stats)
