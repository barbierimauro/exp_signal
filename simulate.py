"""
simulate.py
===========
Genera serie temporali sintetiche con rumore e segnali iniettati per la
valutazione del detector di Forbush Decrease.

Tipi di rumore supportati
--------------------------
  poisson : fondo piatto k campionato con Poisson (caso standard del detector)
  colored : rumore 1/f^beta (da ``colorednoise``) sommato al fondo prima del
            campionamento Poisson; rende il fondo non stazionario
  shot    : picchi impulsivi casuali (glitch, raggi cosmici) sovrapposti al
            fondo Poisson; rappresenta un fondo contaminato da outlier positivi

Modello del segnale
-------------------
    mu(t) = k - a * (1 - exp(-(t-t0)/tau_rise)) * exp(-(t-t0)/b)  per t > t0
    mu(t) = k                                                        per t <= t0

Per tau_rise = 0: onset istantaneo (drop immediato a k-a).

Configurazione temporale
-------------------------
Il campo ``series.dt_seconds`` specifica la durata di ogni campione in secondi.
I parametri b e tau_rise nel file di configurazione sono espressi in campioni.
Per convertire da ore: b_campioni = b_ore * 3600 / dt_seconds.

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

def _signal_profile(
    t: np.ndarray,
    t0: int,
    a: float,
    b: float,
    tau_rise: float = 0.0,
) -> np.ndarray:
    """
    Contributo negativo di un segnale exp-drop al profilo mu:

        delta_mu[t] = -a * (1 - exp(-(t-t0)/tau_rise)) * exp(-(t-t0)/b)
                      per t > t0, 0 altrimenti.

    Per tau_rise = 0: onset istantaneo.
    """
    after = t > t0
    s = np.where(after, t - t0, 0.0)

    exponent_b = np.where(after, -s / b, 0.0)
    decay = np.where(after, a * np.exp(np.clip(exponent_b, -500, 0)), 0.0)

    if tau_rise > 0.0:
        rise_factor = np.where(
            after,
            1.0 - np.exp(np.clip(-s / tau_rise, -500, 0)),
            0.0,
        )
        decay = decay * rise_factor

    return -decay


# ── iniezione segnali ─────────────────────────────────────────────────────────

def _inject_signals(
    mu: np.ndarray,
    k: float,
    cfg_signals: dict,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Inietta segnali exp-drop casuali in ``mu`` in-place.

    I parametri b e tau_rise sono in campioni (non in ore).

    Ritorna lista di dict {t0, a, b, tau_rise, k} che descrive i segnali iniettati.
    """
    N = len(mu)
    t = np.arange(N, dtype=np.float64)

    n_min = int(cfg_signals["n_per_realization"]["min"])
    n_max = int(cfg_signals["n_per_realization"]["max"])
    n_sig = int(rng.integers(n_min, n_max + 1))

    a_min   = float(cfg_signals["a"]["min"])
    a_max   = float(cfg_signals["a"]["max"])
    log_a   = bool(cfg_signals["a"].get("log_scale", True))
    b_min   = float(cfg_signals["b"]["min"])
    b_max   = float(cfg_signals["b"]["max"])
    log_b   = bool(cfg_signals["b"].get("log_scale", True))
    min_sep = int(cfg_signals.get("min_separation", 200))

    # tau_rise opzionale
    tau_cfg = cfg_signals.get("tau_rise", None)
    tau_min = float(tau_cfg["min"]) if tau_cfg else 0.0
    tau_max = float(tau_cfg["max"]) if tau_cfg else 0.0
    log_tau = bool(tau_cfg.get("log_scale", False)) if tau_cfg else False

    true_signals: list[dict] = []
    occupied:     list[int]  = []

    for _ in range(n_sig):
        # campiona a
        if log_a:
            a = float(np.exp(rng.uniform(np.log(a_min), np.log(a_max))))
        else:
            a = float(rng.uniform(a_min, a_max))
        a = min(a, k * 0.30)   # FD: massimo 30% del fondo

        # campiona b
        if log_b:
            b = float(np.exp(rng.uniform(np.log(b_min), np.log(b_max))))
        else:
            b = float(rng.uniform(b_min, b_max))

        # campiona tau_rise (0 se non configurato)
        if tau_cfg and tau_max > 0:
            if log_tau and tau_min > 0:
                tau_rise = float(np.exp(rng.uniform(np.log(tau_min), np.log(tau_max))))
            else:
                tau_rise = float(rng.uniform(tau_min, tau_max))
        else:
            tau_rise = 0.0

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
        mu += _signal_profile(t, t0, a, b, tau_rise)
        true_signals.append({"t0": t0, "a": a, "b": b, "tau_rise": tau_rise, "k": k})

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
    N        = len(mu)
    rate     = float(cfg["rate"])
    amp_mean = float(cfg["amplitude_mean"])
    amp_std  = float(cfg.get("amplitude_std", amp_mean * 0.3))
    n_shots  = int(rng.poisson(rate * N))
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
    true_signals : lista di dict {t0, a, b, tau_rise, k}
    """
    rng = np.random.default_rng(seed)

    N          = int(config["series"]["length"])
    k          = float(config["background"]["k"])
    noise_cfg  = config["noise"]
    noise_type = noise_cfg["type"].lower()

    mu = np.full(N, k, dtype=np.float64)

    if noise_type == "colored":
        _add_colored_noise(mu, noise_cfg["colored"], rng)
        np.clip(mu, 1.0, None, out=mu)
    elif noise_type == "shot":
        _add_shot_noise(mu, noise_cfg["shot"], rng)
    elif noise_type != "poisson":
        raise ValueError(
            f"noise.type non riconosciuto: '{noise_type}'. "
            "Valori validi: poisson, colored, shot."
        )

    true_signals = _inject_signals(mu, k, config["signals"], rng)

    np.clip(mu, 1e-9, None, out=mu)
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
      - **both**  : stesso t0, b, tau_rise in entrambe le serie; a1 ≠ a2
      - **only1** : segnale solo nella serie 1
      - **only2** : segnale solo nella serie 2

    Ritorna
    -------
    data1      : array float64 — serie 1
    data2      : array float64 — serie 2
    true_both  : lista di dict {t0, a1, b, tau_rise, a2, k1, k2}
    true_only1 : lista di dict {t0, a1, b, tau_rise, k1}
    true_only2 : lista di dict {t0, a2, b, tau_rise, k2}
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
    a2f_min   = float(dual_cfg["a2_factor"]["min"])
    a2f_max   = float(dual_cfg["a2_factor"]["max"])
    log_a2f   = bool(dual_cfg["a2_factor"].get("log_scale", True))

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

    t      = np.arange(N, dtype=np.float64)
    n_min  = int(sig_cfg["n_per_realization"]["min"])
    n_max  = int(sig_cfg["n_per_realization"]["max"])
    n_sig  = int(rng.integers(n_min, n_max + 1))
    a_min  = float(sig_cfg["a"]["min"])
    a_max  = float(sig_cfg["a"]["max"])
    log_a  = bool(sig_cfg["a"].get("log_scale", True))
    b_min  = float(sig_cfg["b"]["min"])
    b_max  = float(sig_cfg["b"]["max"])
    log_b  = bool(sig_cfg["b"].get("log_scale", True))
    min_sep = int(sig_cfg.get("min_separation", 200))

    tau_cfg = sig_cfg.get("tau_rise", None)
    tau_min = float(tau_cfg["min"]) if tau_cfg else 0.0
    tau_max = float(tau_cfg["max"]) if tau_cfg else 0.0
    log_tau = bool(tau_cfg.get("log_scale", False)) if tau_cfg else False

    true_both:  list[dict] = []
    true_only1: list[dict] = []
    true_only2: list[dict] = []
    occupied:   list[int]  = []

    for _ in range(n_sig):
        a1 = float(np.exp(rng.uniform(np.log(a_min), np.log(a_max)))) if log_a \
             else float(rng.uniform(a_min, a_max))
        a1 = min(a1, k1 * 0.30)

        b = float(np.exp(rng.uniform(np.log(b_min), np.log(b_max)))) if log_b \
            else float(rng.uniform(b_min, b_max))

        if tau_cfg and tau_max > 0:
            if log_tau and tau_min > 0:
                tau_rise = float(np.exp(rng.uniform(np.log(tau_min), np.log(tau_max))))
            else:
                tau_rise = float(rng.uniform(tau_min, tau_max))
        else:
            tau_rise = 0.0

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
            fac = float(np.exp(rng.uniform(np.log(a2f_min), np.log(a2f_max)))) if log_a2f \
                  else float(rng.uniform(a2f_min, a2f_max))
            a2 = min(max(a1 * fac, 0.1), k2 * 0.30)
            mu1 += _signal_profile(t, t0, a1, b, tau_rise)
            mu2 += _signal_profile(t, t0, a2, b, tau_rise)
            true_both.append({
                "t0": t0, "a1": a1, "b": b, "tau_rise": tau_rise,
                "a2": a2, "k1": k1, "k2": k2,
            })

        elif p < frac_both + (1.0 - frac_both) / 2.0:
            mu1 += _signal_profile(t, t0, a1, b, tau_rise)
            true_only1.append({"t0": t0, "a1": a1, "b": b, "tau_rise": tau_rise, "k1": k1})

        else:
            a2 = min(a1, k2 * 0.30)
            mu2 += _signal_profile(t, t0, a2, b, tau_rise)
            true_only2.append({"t0": t0, "a2": a2, "b": b, "tau_rise": tau_rise, "k2": k2})

    np.clip(mu1, 1e-9, None, out=mu1)
    np.clip(mu2, 1e-9, None, out=mu2)

    data1 = rng.poisson(mu1).astype(np.float64)
    data2 = rng.poisson(mu2).astype(np.float64)

    return data1, data2, true_both, true_only1, true_only2
