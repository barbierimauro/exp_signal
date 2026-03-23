"""
simulate.py
===========
Genera serie temporali sintetiche con rumore e segnali iniettati per la
valutazione del detector di segnali exp-drop.

Tipi di rumore supportati
--------------------------
  poisson : fondo piatto k campionato con Poisson (caso standard del detector)
  colored : rumore 1/f^beta (da ``colorednoise``) sommato al fondo prima del
            campionamento Poisson; rende il fondo non stazionario
  shot    : picchi impulsivi casuali (glitch, raggi cosmici) sovrapposti al
            fondo Poisson; rappresenta un fondo contaminato da outlier positivi

Nota: il detector assume fondo piatto k; rumore colorato e shot noise
degradano le prestazioni in modo fisicamente realistico.

Funzioni pubbliche
------------------
    load_config(path)                         -> dict
    generate_data(config, seed)               -> (data, true_signals)
    generate_dual_data(config, seed)          -> (data1, data2,
                                                   true_both,
                                                   true_only1,
                                                   true_only2)
"""

import numpy as np
import yaml
from pathlib import Path

import colorednoise as cn


# ── caricamento configurazione ────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    """Carica il file YAML di configurazione e ritorna il dizionario."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ── modello segnale (coerente con main.py) ────────────────────────────────────

def _signal_profile(t: np.ndarray, t0: int, a: float, b: float) -> np.ndarray:
    """
    Contributo negativo di un segnale exp-drop al profilo mu:
        delta_mu[t] = -a * exp(-(t - t0) / b)   per t > t0
                    = 0                           per t <= t0
    """
    exponent = np.where(t <= t0, 0.0, -(t - t0) / b)
    return np.where(t <= t0, 0.0, -a * np.exp(np.clip(exponent, -500, 0)))


# ── iniezione segnali ─────────────────────────────────────────────────────────

def _inject_signals(
    mu: np.ndarray,
    k: float,
    cfg_signals: dict,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Inietta segnali exp-drop casuali in ``mu`` in-place.

    Ritorna lista di dict {t0, a, b, k} che descrive i segnali iniettati.
    """
    N = len(mu)
    t = np.arange(N, dtype=np.float64)

    n_min = int(cfg_signals["n_per_realization"]["min"])
    n_max = int(cfg_signals["n_per_realization"]["max"])
    n_sig = int(rng.integers(n_min, n_max + 1))

    a_min    = float(cfg_signals["a"]["min"])
    a_max    = float(cfg_signals["a"]["max"])
    log_a    = bool(cfg_signals["a"].get("log_scale", True))
    b_min    = float(cfg_signals["b"]["min"])
    b_max    = float(cfg_signals["b"]["max"])
    log_b    = bool(cfg_signals["b"].get("log_scale", True))
    min_sep  = int(cfg_signals.get("min_separation", 200))

    true_signals: list[dict] = []
    occupied:     list[int]  = []

    for _ in range(n_sig):
        # campiona a
        if log_a:
            a = float(np.exp(rng.uniform(np.log(a_min), np.log(a_max))))
        else:
            a = float(rng.uniform(a_min, a_max))
        a = min(a, k * 0.93)  # garantisce a/k < 0.93 (validità espansione Taylor)

        # campiona b
        if log_b:
            b = float(np.exp(rng.uniform(np.log(b_min), np.log(b_max))))
        else:
            b = float(rng.uniform(b_min, b_max))

        # cerca t0 libero (lontano dagli altri segnali e dai bordi)
        margin = max(min_sep, int(10 * b))
        if margin * 2 >= N:
            continue
        for _ in range(300):
            t0 = int(rng.integers(min_sep, N - margin))
            if all(abs(t0 - o) >= min_sep for o in occupied):
                break
        else:
            continue  # spazio esaurito, segnale saltato

        occupied.append(t0)
        mu += _signal_profile(t, t0, a, b)
        true_signals.append({"t0": t0, "a": a, "b": b, "k": k})

    return true_signals


# ── generazione rumore ────────────────────────────────────────────────────────

def _add_colored_noise(mu: np.ndarray, cfg: dict, rng: np.random.Generator) -> None:
    """Aggiunge rumore colorato 1/f^beta a mu in-place."""
    exponent  = float(cfg["exponent"])
    amplitude = float(cfg["amplitude"])
    seed_cn   = int(rng.integers(0, 2**31))
    noise     = cn.powerlaw_psd_gaussian(exponent, len(mu), random_state=seed_cn)
    noise     = noise / (float(np.std(noise)) + 1e-12) * amplitude
    mu       += noise


def _add_shot_noise(mu: np.ndarray, cfg: dict, rng: np.random.Generator) -> None:
    """Aggiunge picchi impulsivi casuali (shot noise) a mu in-place."""
    N         = len(mu)
    rate      = float(cfg["rate"])
    amp_mean  = float(cfg["amplitude_mean"])
    amp_std   = float(cfg.get("amplitude_std", amp_mean * 0.3))
    n_shots   = int(rng.poisson(rate * N))
    if n_shots == 0:
        return
    times = rng.integers(0, N, size=n_shots)
    amps  = np.clip(rng.normal(amp_mean, amp_std, size=n_shots), 0.0, None)
    np.add.at(mu, times, amps)


# ── API pubblica ──────────────────────────────────────────────────────────────

def generate_data(
    config: dict,
    seed: int | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Genera una serie temporale sintetica con rumore e segnali iniettati.

    Parametri
    ---------
    config : dict prodotto da load_config()
    seed   : seme RNG per riproducibilità

    Ritorna
    -------
    data         : array 1D float64 di conteggi (campionati da Poisson)
    true_signals : lista di dict {t0, a, b, k} — parametri veri dei segnali
    """
    rng = np.random.default_rng(seed)

    N         = int(config["series"]["length"])
    k         = float(config["background"]["k"])
    noise_cfg = config["noise"]
    noise_type = noise_cfg["type"].lower()

    # profilo di valor medio (fondo piatto + rumore deterministico + segnali)
    mu = np.full(N, k, dtype=np.float64)

    if noise_type == "colored":
        _add_colored_noise(mu, noise_cfg["colored"], rng)
        np.clip(mu, 1.0, None, out=mu)          # evita mu negativi
    elif noise_type == "shot":
        _add_shot_noise(mu, noise_cfg["shot"], rng)
    elif noise_type != "poisson":
        raise ValueError(
            f"noise.type non riconosciuto: '{noise_type}'. "
            "Valori validi: poisson, colored, shot."
        )

    # inietta segnali exp-drop
    true_signals = _inject_signals(mu, k, config["signals"], rng)

    # garantisce mu > 0 prima del campionamento
    np.clip(mu, 1e-9, None, out=mu)

    # campionamento poissoniano
    data = rng.poisson(mu).astype(np.float64)

    return data, true_signals


# ── generatore dual-channel ───────────────────────────────────────────────────

def generate_dual_data(
    config: dict,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict], list[dict]]:
    """
    Genera due serie temporali correlate per la valutazione del detector duale.

    I segnali vengono classificati casualmente come:
      - **both**  : stesso t0 e b in entrambe le serie, a1 ≠ a2
      - **only1** : segnale solo nella serie 1
      - **only2** : segnale solo nella serie 2

    La proporzione è controllata da ``dual.fraction_both`` nel config.
    Il rapporto a2/a1 per i segnali "both" è campionato dall'intervallo
    ``dual.a2_factor``.

    Richiede nel config le sezioni ``background``, ``noise``, ``signals``
    (per la serie 1) e ``dual`` (per la serie 2).

    Parametri
    ---------
    config : dict da load_config()
    seed   : seme RNG

    Ritorna
    -------
    data1      : array float64 — serie 1
    data2      : array float64 — serie 2
    true_both  : lista di dict {t0, a1, b, a2, k1, k2}
    true_only1 : lista di dict {t0, a1, b, k1}
    true_only2 : lista di dict {t0, a2, b, k2}
    """
    rng = np.random.default_rng(seed)

    N          = int(config["series"]["length"])
    k1         = float(config["background"]["k"])
    dual_cfg   = config["dual"]
    k2         = float(dual_cfg["k2"])
    sig_cfg    = config["signals"]
    noise1_cfg = config["noise"]
    noise2_cfg = dual_cfg["noise2"]

    frac_both = float(dual_cfg.get("fraction_both", 0.5))

    a2f_min  = float(dual_cfg["a2_factor"]["min"])
    a2f_max  = float(dual_cfg["a2_factor"]["max"])
    log_a2f  = bool(dual_cfg["a2_factor"].get("log_scale", True))

    # --- fondi con rumore ---
    mu1 = np.full(N, k1, dtype=np.float64)
    mu2 = np.full(N, k2, dtype=np.float64)

    if noise1_cfg["type"] == "colored":
        _add_colored_noise(mu1, noise1_cfg["colored"], rng)
        np.clip(mu1, 1.0, None, out=mu1)
    elif noise1_cfg["type"] == "shot":
        _add_shot_noise(mu1, noise1_cfg["shot"], rng)

    if noise2_cfg["type"] == "colored":
        _add_colored_noise(mu2, noise2_cfg["colored"], rng)
        np.clip(mu2, 1.0, None, out=mu2)
    elif noise2_cfg["type"] == "shot":
        _add_shot_noise(mu2, noise2_cfg["shot"], rng)

    # --- parametri griglia segnali ---
    t       = np.arange(N, dtype=np.float64)
    n_min   = int(sig_cfg["n_per_realization"]["min"])
    n_max   = int(sig_cfg["n_per_realization"]["max"])
    n_sig   = int(rng.integers(n_min, n_max + 1))
    a_min   = float(sig_cfg["a"]["min"])
    a_max   = float(sig_cfg["a"]["max"])
    log_a   = bool(sig_cfg["a"].get("log_scale", True))
    b_min   = float(sig_cfg["b"]["min"])
    b_max   = float(sig_cfg["b"]["max"])
    log_b   = bool(sig_cfg["b"].get("log_scale", True))
    min_sep = int(sig_cfg.get("min_separation", 200))

    true_both:  list[dict] = []
    true_only1: list[dict] = []
    true_only2: list[dict] = []
    occupied:   list[int]  = []

    for _ in range(n_sig):
        # campiona a1, b
        a1 = float(np.exp(rng.uniform(np.log(a_min), np.log(a_max)))) if log_a \
             else float(rng.uniform(a_min, a_max))
        a1 = min(a1, k1 * 0.93)

        b = float(np.exp(rng.uniform(np.log(b_min), np.log(b_max)))) if log_b \
            else float(rng.uniform(b_min, b_max))

        # cerca t0 libero
        margin = max(min_sep, int(10 * b))
        if margin * 2 >= N:
            continue
        for _ in range(300):
            t0 = int(rng.integers(min_sep, N - margin))
            if all(abs(t0 - o) >= min_sep for o in occupied):
                break
        else:
            continue
        occupied.append(t0)

        p = float(rng.uniform())

        if p < frac_both:
            # segnale in entrambe le serie
            fac = float(np.exp(rng.uniform(np.log(a2f_min), np.log(a2f_max)))) if log_a2f \
                  else float(rng.uniform(a2f_min, a2f_max))
            a2 = min(max(a1 * fac, 0.1), k2 * 0.93)
            mu1 += _signal_profile(t, t0, a1, b)
            mu2 += _signal_profile(t, t0, a2, b)
            true_both.append({"t0": t0, "a1": a1, "b": b, "a2": a2, "k1": k1, "k2": k2})

        elif p < frac_both + (1.0 - frac_both) / 2.0:
            # solo serie 1
            mu1 += _signal_profile(t, t0, a1, b)
            true_only1.append({"t0": t0, "a1": a1, "b": b, "k1": k1})

        else:
            # solo serie 2 (usa a1 come ampiezza per serie 2, riscalato)
            a2 = min(a1, k2 * 0.93)
            mu2 += _signal_profile(t, t0, a2, b)
            true_only2.append({"t0": t0, "a2": a2, "b": b, "k2": k2})

    np.clip(mu1, 1e-9, None, out=mu1)
    np.clip(mu2, 1e-9, None, out=mu2)

    data1 = rng.poisson(mu1).astype(np.float64)
    data2 = rng.poisson(mu2).astype(np.float64)

    return data1, data2, true_both, true_only1, true_only2
