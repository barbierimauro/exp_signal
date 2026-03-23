"""
pipeline.py
===========
Entry point unificato. Tre modalità selezionabili via campo ``mode`` nel
file di configurazione YAML:

  simulate       : genera serie singola sintetica, valuta detector
                   (delega a evaluate.run_evaluation)
  evaluate_dual  : genera coppie di serie correlate, valuta detector duale,
                   produce matrice di confusione, grafici ed efficienza
  search_file    : legge CSV (datetime ISO + 1-2 colonne), cerca segnali,
                   produce grafici con asse datetime e report ASCII

Uso
---
    python pipeline.py [pipeline_config.yaml]

oppure via API:
    from pipeline import run_pipeline
    stats = run_pipeline("pipeline_config.yaml")
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime, timezone

from simulate    import load_config, generate_data, generate_dual_data
from evaluate    import run_evaluation, _cp68, _efficiency_with_errors, _publication_style
from dual_search import (calibrate_thresholds_dual, find_signals_dual,
                          plot_dual_results, write_dual_report, _CAT_COLOR, _CAT_LABEL)
from main        import calibrate_threshold, find_signals
from io_utils    import load_csv, datetime_axis


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATE_DUAL — matching, plot, report
# ═══════════════════════════════════════════════════════════════════════════════

def _match_dual_eval(true_both, true_only1, true_only2, results):
    """
    Abbina segnali veri a detectati (greedy min |Δt0|, tolleranza = 3·b).
    Ritorna dict con confusion matrix e liste dei segnali abbinati.
    """
    all_true = ([("both",  s) for s in true_both] +
                [("only1", s) for s in true_only1] +
                [("only2", s) for s in true_only2])
    all_det  = [(cat, s) for cat, sigs in results.items() for s in sigs]

    matched_det = set()
    conf        = {}
    found_both, found_only1, found_only2 = [], [], []

    for i, (tc, ts) in enumerate(all_true):
        tol = max(10, int(3 * ts.get("b", 20.0)))
        best_j, best_dt = None, np.inf
        for j, (_, ds) in enumerate(all_det):
            if j in matched_det:
                continue
            dt = abs(ds["t0"] - ts["t0"])
            if dt <= tol and dt < best_dt:
                best_j, best_dt = j, dt

        if best_j is not None:
            dc = all_det[best_j][0]
            matched_det.add(best_j)
            if   tc == "both":  found_both.append( (ts, all_det[best_j][1]))
            elif tc == "only1": found_only1.append((ts, all_det[best_j][1]))
            else:               found_only2.append((ts, all_det[best_j][1]))
        else:
            dc = "missed"

        conf[(tc, dc)] = conf.get((tc, dc), 0) + 1

    return {"conf": conf,
            "n_false_pos":  len(all_det) - len(matched_det),
            "n_true_both":  len(true_both),
            "n_true_only1": len(true_only1),
            "n_true_only2": len(true_only2),
            "found_both":   found_both,
            "found_only1":  found_only1,
            "found_only2":  found_only2}


def _plot_evaluate_dual(ex, acc, conf_total, config, plot_path, dpi):
    """
    Figura 3-riga per evaluate_dual.
    ex  = dict con data1, data2, true_both, true_only1, true_only2, results
    acc = dict con *_a_true / *_a_found per le curve di efficienza
    """
    _publication_style()
    fig = plt.figure(figsize=(14, 11))
    gs  = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.1],
                   hspace=0.42, wspace=0.34)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])
    ax5 = fig.add_subplot(gs[2, 2])

    # mappa colori per segnali veri nell'esempio
    _TC = {"both": "C0", "only1": "C2", "only2": "C3"}

    def _ts_panel(ax, data, k_val, true_sigs_map, results, ch, label):
        ax.step(np.arange(len(data)), data, color="0.65", lw=0.5, where="mid")
        ax.axhline(k_val, color="k", ls="--", lw=0.9)
        for tc, sigs in true_sigs_map.items():
            for s in sigs:
                ax.axvline(s["t0"], color=_TC[tc], lw=1.5, alpha=0.6, ls="-")
        for cat, sigs in results.items():
            for s in sigs:
                if ch == "1" and cat == "only2": continue
                if ch == "2" and cat == "only1": continue
                ax.axvline(s["t0"], color=_CAT_COLOR[cat], lw=1.2,
                           alpha=0.85, ls="--")
        ax.set_ylabel("Counts"); ax.set_title(label)
        ax.set_xlim(0, len(data) - 1)
        handles = [
            Line2D([0],[0], color="0.65", lw=1, label="Data"),
            Line2D([0],[0], color="k",    lw=0.9, ls="--", label=f"k={k_val:.0f}"),
            Line2D([0],[0], color="C0",   lw=1.5, label="True both"),
            Line2D([0],[0], color="C2",   lw=1.5, label="True only1"),
            Line2D([0],[0], color="C3",   lw=1.5, label="True only2"),
            Line2D([0],[0], color="grey", lw=1.2, ls="--", label="Detected"),
        ]
        ax.legend(handles=handles, ncol=3, fontsize=8, loc="upper right",
                  framealpha=0.9)

    k1 = float(config["background"]["k"])
    k2 = float(config["dual"]["k2"])
    _ts_panel(ax1, ex["data1"], k1,
              {"both": ex["true_both"], "only1": ex["true_only1"]},
              ex["results"], "1", "Series 1 — example realization")
    _ts_panel(ax2, ex["data2"], k2,
              {"both": ex["true_both"], "only2": ex["true_only2"]},
              ex["results"], "2", "Series 2 — example realization")

    # efficienza vs a — segnali "both"
    def _eff_panel(ax, a_true, a_found, color, title):
        if not a_true:
            ax.set_title(title); return
        bins = np.geomspace(min(a_true)*0.85, max(a_true)*1.15, 9)
        c, eff, elo, ehi = _efficiency_with_errors(a_true, a_found, bins)
        ok = ~np.isnan(eff)
        ax.errorbar(c[ok], eff[ok], yerr=[elo[ok], ehi[ok]],
                    fmt="o-", color=color, ms=5, capsize=3, lw=1.3)
        ax.fill_between(c[ok], eff[ok]-elo[ok], eff[ok]+ehi[ok],
                        alpha=0.15, color=color)
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.10)
        ax.set_xlabel("Signal amplitude $a$")
        ax.set_ylabel("Detection efficiency")
        ax.set_title(title)
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    _eff_panel(ax3, acc["both_a_true"],  acc["both_a_found"],  "C0", "Efficiency — both")
    _eff_panel(ax4, acc["o1_a_true"],    acc["o1_a_found"],    "C2", "Efficiency — only 1")

    # recall per categoria (barre con CP68)
    cats   = ["both", "only1", "only2"]
    labels = ["Both", "Only 1", "Only 2"]
    colors = ["C0", "C2", "C3"]
    n_true = [sum(conf_total.get((c, d), 0) for d in
               ["both","joint_only","only1","only2","missed"]) for c in cats]
    n_ok   = [conf_total.get(("both","both"),0) + conf_total.get(("both","joint_only"),0),
               conf_total.get(("only1","only1"), 0),
               conf_total.get(("only2","only2"), 0)]
    for i, (lbl, nt, nf, col) in enumerate(zip(labels, n_true, n_ok, colors)):
        if nt == 0: continue
        eff = nf / nt
        lo, hi = _cp68(nf, nt)
        ax5.bar(i, eff, color=col, edgecolor="k", lw=0.6, alpha=0.85)
        ax5.errorbar(i, eff, yerr=[[eff-lo],[hi-eff]], fmt="none",
                     color="k", capsize=4, lw=1.2)
        ax5.text(i, min(hi + 0.04, 1.0), f"{eff:.2f}", ha="center",
                 va="bottom", fontsize=9)
    ax5.set_xticks(range(len(labels))); ax5.set_xticklabels(labels)
    ax5.set_ylim(0, 1.15); ax5.set_ylabel("Recall (68% CI)")
    ax5.set_title("Recall by true category")

    n_real = config["series"]["n_realizations"]
    fig.suptitle(
        f"Dual-channel evaluation  |  {n_real} realizations  |  "
        f"noise: {config['noise']['type']}  |  "
        f"$k_1={k1:.0f}$, $k_2={k2:.0f}$",
        fontsize=12)
    fig.savefig(plot_path, dpi=dpi)
    plt.close(fig)


def _write_eval_dual_report(all_stats, conf_total, thresholds, config, path):
    W, SEP, sep = 80, "="*80, "-"*80
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    k1, k2 = config["background"]["k"], config["dual"]["k2"]

    def _recall(tc, ok_cats):
        n  = sum(conf_total.get((tc, d), 0)
                 for d in ["both","joint_only","only1","only2","missed"])
        nf = sum(conf_total.get((tc, d), 0) for d in ok_cats)
        return nf, n, (nf/n if n else 0.0)

    nf_b,  nt_b,  rec_b  = _recall("both",  ["both","joint_only"])
    nf_o1, nt_o1, rec_o1 = _recall("only1", ["only1"])
    nf_o2, nt_o2, rec_o2 = _recall("only2", ["only2"])
    n_fp = sum(s["n_false_pos"] for s in all_stats)
    n_r  = len(all_stats)

    lines = [SEP, "  DUAL-CHANNEL EVALUATION REPORT", f"  {ts}", SEP, "",
             "CONFIGURATION", sep,
             f"  mode          : evaluate_dual",
             f"  noise         : {config['noise']['type']}",
             f"  k1 / k2       : {k1} / {k2}",
             f"  series length : {config['series']['length']}",
             f"  realizations  : {n_r}",
             f"  thr1 / thr2 / thr_joint : "
             f"{thresholds[0]:.3f} / {thresholds[1]:.3f} / {thresholds[2]:.3f}",
             "",
             "CONFUSION MATRIX  (rows=true, cols=detected)",
             sep,
             "  {:<12}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}".format(
                 "True \\ Det", "both", "joint", "only1", "only2", "missed", "total"),
             sep]
    for tc, lbl in [("both","Both"),("only1","Only 1"),("only2","Only 2")]:
        row = [conf_total.get((tc, d), 0)
               for d in ["both","joint_only","only1","only2","missed"]]
        tot = sum(row)
        lines.append(f"  {lbl:12}  " + "  ".join(f"{v:>6}" for v in row)
                     + f"  {tot:>6}")

    lines += [sep, "",
              "DETECTION EFFICIENCY  (Clopper-Pearson 68% CI)",
              sep,
              f"  {'Category':14}  {'found':>6}  {'total':>6}  "
              f"{'recall':>8}  {'lo68':>8}  {'hi68':>8}",
              sep]
    for lbl, nf, nt, ok_cats in [
            ("Both (simult.)", nf_b,  nt_b,  ["both","joint_only"]),
            ("Only series 1",  nf_o1, nt_o1, ["only1"]),
            ("Only series 2",  nf_o2, nt_o2, ["only2"])]:
        if nt == 0:
            lines.append(f"  {lbl:14}  {'—':>6}  {'0':>6}")
            continue
        rec = nf / nt
        lo, hi = _cp68(nf, nt)
        lines.append(f"  {lbl:14}  {nf:>6}  {nt:>6}  "
                     f"{rec:>8.3f}  {lo:>8.3f}  {hi:>8.3f}")
    lines += [f"  False positives (total)  : {n_fp}",
              f"  False positives / realiz : {n_fp/n_r:.2f}",
              "",
              "PER-REALIZATION SUMMARY",
              sep,
              f"  {'#':>4}  {'n_both':>7}  {'n_o1':>5}  {'n_o2':>5}  "
              f"{'found_b':>7}  {'found1':>6}  {'found2':>6}  {'fp':>4}",
              sep]
    for i, s in enumerate(all_stats):
        lines.append(
            f"  {i+1:>4}  {s['n_true_both']:>7}  {s['n_true_only1']:>5}  "
            f"{s['n_true_only2']:>5}  "
            f"{len(s['found_both']):>7}  {len(s['found_only1']):>6}  "
            f"{len(s['found_only2']):>6}  {s['n_false_pos']:>4}")
    lines += [sep, "", SEP, "  END OF REPORT", SEP, ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODALITÀ
# ═══════════════════════════════════════════════════════════════════════════════

def _run_simulate(config, config_path):
    """Delega a evaluate.run_evaluation (serie singola)."""
    print("[pipeline] mode=simulate → evaluate.run_evaluation")
    return run_evaluation(config_path)


def _run_evaluate_dual(config):
    N, k1 = int(config["series"]["length"]), float(config["background"]["k"])
    k2     = float(config["dual"]["k2"])
    n_r    = int(config["series"]["n_realizations"])
    det    = config["detection"]
    out    = config["output"]
    seed0  = int(det.get("seed", 42))

    print(f"[pipeline] Calibrazione soglie dual  k1={k1} k2={k2} N={N} ...")
    thr1, thr2, thrj = calibrate_thresholds_dual(
        k1, k2, N, n_cal=det["n_cal"], fpr=det["fpr"],
        n_sim=det["n_sim"], seed=seed0)
    print(f"[pipeline] thr1={thr1:.3f}  thr2={thr2:.3f}  thrj={thrj:.3f}")

    all_stats = []
    acc = {k: [] for k in ["both_a_true","both_a_found",
                            "o1_a_true","o1_a_found",
                            "o2_a_true","o2_a_found"]}
    ex = {}

    for i in range(n_r):
        d1, d2, tb, to1, to2 = generate_dual_data(config, seed=seed0 + i + 1)
        res   = find_signals_dual(d1, d2, thr1, thr2, thrj, k1=k1, k2=k2)
        stats = _match_dual_eval(tb, to1, to2, res)
        all_stats.append(stats)

        if i == 0:
            ex = dict(data1=d1, data2=d2, true_both=tb,
                      true_only1=to1, true_only2=to2, results=res)

        for s in tb:   acc["both_a_true"].append(s["a1"])
        for s,_ in stats["found_both"]:  acc["both_a_found"].append(s["a1"])
        for s in to1:  acc["o1_a_true"].append(s["a1"])
        for s,_ in stats["found_only1"]: acc["o1_a_found"].append(s["a1"])
        for s in to2:  acc["o2_a_true"].append(s["a2"])
        for s,_ in stats["found_only2"]: acc["o2_a_found"].append(s["a2"])

        if (i+1) % max(1, n_r//5) == 0 or i == 0:
            nb  = len(stats["found_both"])
            n1  = len(stats["found_only1"])
            n2  = len(stats["found_only2"])
            print(f"[pipeline]  {i+1}/{n_r}  "
                  f"inj_b={stats['n_true_both']} "
                  f"inj_1={stats['n_true_only1']} "
                  f"inj_2={stats['n_true_only2']}  "
                  f"found b/1/2={nb}/{n1}/{n2}  fp={stats['n_false_pos']}")

    conf_total = {}
    for s in all_stats:
        for k, v in s["conf"].items():
            conf_total[k] = conf_total.get(k, 0) + v

    plot_path   = Path(out["plot_file"])
    report_path = Path(out["report_file"])
    print(f"\n[pipeline] Salvo grafici → {plot_path}")
    _plot_evaluate_dual(ex, acc, conf_total, config, plot_path, int(out.get("dpi", 300)))
    print(f"[pipeline] Scrivo report → {report_path}")
    _write_eval_dual_report(all_stats, conf_total, (thr1, thr2, thrj), config, report_path)

    def _rec(tc, ok):
        nf = sum(conf_total.get((tc, d), 0) for d in ok)
        nt = sum(conf_total.get((tc, d), 0)
                 for d in ["both","joint_only","only1","only2","missed"])
        return nf/nt if nt else 0.0

    stats_out = dict(
        recall_both  = _rec("both",  ["both","joint_only"]),
        recall_only1 = _rec("only1", ["only1"]),
        recall_only2 = _rec("only2", ["only2"]),
        n_false_pos  = sum(s["n_false_pos"] for s in all_stats),
        n_realizations = n_r,
    )
    print(f"\n[pipeline] recall_both={stats_out['recall_both']:.3f}  "
          f"recall_only1={stats_out['recall_only1']:.3f}  "
          f"recall_only2={stats_out['recall_only2']:.3f}  "
          f"fp_tot={stats_out['n_false_pos']}")
    return stats_out


def _run_search_file(config):
    inp   = config["input"]
    det   = config["detection"]
    out   = config["output"]
    seed0 = int(det.get("seed", 42))

    s2col    = inp.get("series2_col") or None
    k1_user  = inp.get("k1") or None
    k2_user  = inp.get("k2") or None
    lbl1     = inp.get("series1_label", inp["series1_col"])
    lbl2     = inp.get("series2_label", inp.get("series2_col", "Series 2"))

    print(f"[pipeline] Lettura {inp['file']} ...")
    ts, d1, d2, info = load_csv(
        inp["file"],
        datetime_col = inp["datetime_col"],
        series1_col  = inp["series1_col"],
        series2_col  = s2col,
        delimiter    = inp.get("delimiter", ","),
        k1 = k1_user, k2 = k2_user,
    )
    N  = info["N"]
    k1 = info["k1"]
    print(f"[pipeline] N={N}  dt={info['sampling_seconds']:.2f}s  "
          f"{info['start']} → {info['end']}")

    plot_path   = Path(out["plot_file"])
    report_path = Path(out["report_file"])
    dpi         = int(out.get("dpi", 300))

    extra = {"File": inp["file"], "N": N,
             "Sampling": f"{info['sampling_seconds']:.3f} s",
             "Period": f"{info['start']} – {info['end']}"}

    if d2 is not None:
        k2 = info["k2"]
        print(f"[pipeline] Dual-channel  k1={k1:.2f}  k2={k2:.2f}")
        print("[pipeline] Calibrazione soglie dual ...")
        thr1, thr2, thrj = calibrate_thresholds_dual(
            k1, k2, N, n_cal=det["n_cal"], fpr=det["fpr"],
            n_sim=det["n_sim"], seed=seed0)
        print(f"[pipeline] thr1={thr1:.3f}  thr2={thr2:.3f}  thrj={thrj:.3f}")

        results = find_signals_dual(d1, d2, thr1, thr2, thrj, k1=k1, k2=k2)
        n_tot   = sum(len(v) for v in results.values())
        print(f"[pipeline] Detections: "
              f"both={len(results['both'])}  joint={len(results['joint_only'])}  "
              f"only1={len(results['only1'])}  only2={len(results['only2'])}")

        print(f"[pipeline] Salvo grafici → {plot_path}")
        plot_dual_results(d1, d2, results, plot_path, dpi=dpi,
                          label1=lbl1, label2=lbl2,
                          k1=k1, k2=k2, timestamps=ts)

        extra.update({"k1": f"{k1:.2f}", "k2": f"{k2:.2f}",
                      lbl1: inp["series1_col"], lbl2: inp.get("series2_col","")})
        print(f"[pipeline] Scrivo report → {report_path}")
        write_dual_report(results, report_path,
                          thresholds=(thr1, thr2, thrj),
                          label1=lbl1, label2=lbl2, extra_info=extra)
        return results

    else:
        print(f"[pipeline] Single-channel  k1={k1:.2f}")
        print("[pipeline] Calibrazione soglia ...")
        thr = calibrate_threshold(k1, N, n_cal=det["n_cal"],
                                  fpr=det["fpr"], n_sim=det["n_sim"], seed=seed0)
        print(f"[pipeline] threshold={thr:.3f}")

        sigs = find_signals(d1, threshold=thr, k=k1)
        print(f"[pipeline] Segnali trovati: {len(sigs)}")

        # plot single-channel
        _publication_style()
        fig, axes = plt.subplots(1, 2, figsize=(13, 4),
                                 gridspec_kw={"width_ratios": [3, 1]})
        ax, ax2 = axes
        ax.step(np.arange(N), d1, color="0.65", lw=0.5, where="mid")
        ax.axhline(k1, color="k", ls="--", lw=0.9, label=f"k={k1:.0f}")
        for s in sigs:
            ax.axvline(s["t0"], color="C0", lw=1.5, alpha=0.8)
        datetime_axis(ax, ts, max_ticks=6)
        ax.set_ylabel("Counts"); ax.set_title(f"{lbl1} — {len(sigs)} signal(s) detected")
        ax.legend(fontsize=9)
        if sigs:
            av = [s["a"] for s in sigs]; bv = [s["b"] for s in sigs]
            ax2.scatter(av, bv, c=[s["llr"] for s in sigs],
                        cmap="viridis", s=40, edgecolors="k", lw=0.5)
            ax2.set_xlabel("Amplitude $a$"); ax2.set_ylabel("Timescale $b$")
            ax2.set_title("Detected parameters")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=dpi); plt.close(fig)

        # report ASCII
        W = 78
        lines = ["="*W, "  SINGLE-CHANNEL SEARCH REPORT",
                 f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
                 "="*W, ""]
        for k, v in extra.items():
            lines.append(f"  {k:<24}: {v}")
        lines += [f"  {'LLR threshold':<24}: {thr:.4f}",
                  f"  {'Signals found':<24}: {len(sigs)}", "",
                  "-"*W,
                  f"  {'#':>4}  {'t0':>8}  {'a':>8}  {'±':>6}  "
                  f"{'b':>8}  {'±':>6}  {'llr':>8}",
                  "-"*W]
        for i, s in enumerate(sigs):
            def _f(v): return f"{v:.3f}" if np.isfinite(v) else "nan"
            lines.append(f"  {i+1:>4}  {s['t0']:>8}  {_f(s['a']):>8}  "
                         f"{_f(s['a_err']):>6}  {_f(s['b']):>8}  "
                         f"{_f(s['b_err']):>6}  {_f(s['llr']):>8}")
        lines += ["-"*W, "", "="*W, "  END OF REPORT", "="*W, ""]
        Path(report_path).write_text("\n".join(lines), encoding="utf-8")
        return sigs


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(config_path: str | Path) -> dict:
    """
    Esegue la pipeline nella modalità specificata dal campo ``mode`` del config.

    Parametri
    ---------
    config_path : percorso del file YAML (es. pipeline_config.yaml)

    Ritorna
    -------
    dict con statistiche riepilogative della modalità eseguita
    """
    config_path = Path(config_path)
    config      = load_config(config_path)
    mode        = config.get("mode", "simulate").strip().lower()

    print(f"[pipeline] ── mode={mode}  config={config_path.name} ──")

    if   mode == "simulate":
        return _run_simulate(config, config_path)
    elif mode == "evaluate_dual":
        return _run_evaluate_dual(config)
    elif mode == "search_file":
        return _run_search_file(config)
    else:
        raise ValueError(
            f"mode '{mode}' non riconosciuto. "
            "Valori validi: simulate | evaluate_dual | search_file")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "pipeline_config.yaml"
    result = run_pipeline(cfg)
    print("\n[pipeline] Done:", result)
