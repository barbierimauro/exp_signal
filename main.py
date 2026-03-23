"""
find_signals.py
===============
Cerca segnali della forma

    mu(t) = k                                                            per t <= t0
    mu(t) = k - a * (1 - exp(-(t-t0)/tau_rise)) * exp(-(t-t0)/b)  per t > t0

in serie temporali con rumore di Poisson, anche con N = 1 000 000 punti.

Per tau_rise = 0 (default), il modello si riduce all'onset istantaneo:

    mu(t) = k - a * exp(-(t-t0)/b)   per t > t0

Parametri del segnale
─────────────────────
k         : livello di fondo (conteggi/campione)
t0        : istante di onset del Forbush Decrease (indice intero)
a         : ampiezza del calo (conteggi; tipicamente 0.5–25% di k per FD)
b         : scala temporale di recupero (campioni; tipicamente 12–240 h per FD)
tau_rise  : scala temporale dell'onset (campioni; 0 = onset istantaneo)

─── Algoritmo ────────────────────────────────────────────────────────────────

Per ogni coppia (a, b) della griglia, il profilo LLR su TUTTI i t0
viene calcolato in O(N) con filtri IIR anti-causali (tau_rise = 0
nella fase di scansione; libero nel raffinamento Nelder-Mead).

La formula LLR si espande in serie di Taylor:

    LLR(t0) = a · S1(t0) − Σₙ₌₁ᴼᴿᴰᴱᴿ (a/k)ⁿ/n · Cₙ(t0)

dove Cₙ(t0) = Σ_{s≥1} data[t0+s] · exp(−ns/b) è un filtro IIR con polo
αⁿ = exp(−n/b), calcolato una sola volta per ogni b e riusato per tutti gli a.

─── Griglia di default per Forbush Decrease ──────────────────────────────────

La griglia di default copre:
  b : da 12 ore a 10 giorni (adattata al campionamento tramite dt_seconds)
  a : da 0.5% a 25% di k  (range tipico dei FD)

─── Stima robusta del fondo ──────────────────────────────────────────────────

k viene stimato con due metodi robusti (selezionabili via k_method):
  sigma_clip  : media iterativa con rimozione a nsigma sigma (default)
  histogram   : moda dell'istogramma dei conteggi

Entrambi resistono al bias della media aritmetica in presenza di FD.

─── Correzione FPR per serie lunghe ──────────────────────────────────────────

calibrate_threshold() applica automaticamente la correzione di Bonferroni:
usa fpr_target = fpr / (N / n_cal) per mantenere il FPR globale nominale.

─── Funzioni pubbliche ────────────────────────────────────────────────────────

    estimate_background(data, method, ...)     →  float
    calibrate_threshold(k, N, ..., dt_seconds) →  float
    find_signals(data, threshold, ...,
                 dt_seconds, tau_rise_max)      →  list[dict]

Ogni dict ha le chiavi:
    t0, a, b, tau_rise, llr, a_err, b_err, tau_rise_err, k
"""

import warnings
import numpy as np
from scipy.signal import lfilter
from scipy.optimize import minimize


# ── costanti ──────────────────────────────────────────────────────────────────

_TAYLOR_ORDER    = 8     # ordine espansione log(1−x); errore < 0.02% per a/k < 0.95
_REFINE_MAX_ITER = 300
_N_CAL_DEFAULT   = 50_000


# ── stima robusta del fondo ───────────────────────────────────────────────────

def estimate_background(
    data: np.ndarray,
    method: str = 'sigma_clip',
    nsigma: float = 3.0,
    n_bins: int = 200,
) -> float:
    """
    Stima robusta del livello di fondo k per serie con segnali transienti.

    In presenza di Forbush Decrease, la media aritmetica è biasata verso
    il basso. Entrambi i metodi disponibili sono resistenti a questo bias.

    Parametri
    ---------
    data    : array 1D di conteggi
    method  : 'sigma_clip' (default) | 'histogram' | 'mean'
              'sigma_clip' : media iterativa con rimozione dei valori oltre
                             nsigma sigma. Robusto a cali transienti.
              'histogram'  : centro del bin più frequente dell'istogramma.
                             Utile quando il fondo ha distribuzione stretta.
              'mean'       : media aritmetica (non robusta; solo per confronto).
    nsigma  : soglia sigma per la rimozione iterativa (solo sigma_clip)
    n_bins  : numero di bin per il metodo histogram

    Ritorna
    -------
    k_est : float
    """
    data = np.asarray(data, dtype=np.float64)
    d = data[np.isfinite(data)]
    if len(d) == 0:
        raise ValueError("Array vuoto o tutti NaN.")

    if method == 'mean':
        return float(np.mean(d))

    elif method == 'histogram':
        counts, edges = np.histogram(d, bins=n_bins)
        peak_idx = int(np.argmax(counts))
        return float(0.5 * (edges[peak_idx] + edges[peak_idx + 1]))

    elif method == 'sigma_clip':
        clipped = d.copy()
        for _ in range(20):
            m = float(np.mean(clipped))
            s = float(np.std(clipped))
            if s == 0:
                break
            mask = np.abs(clipped - m) <= nsigma * s
            if mask.sum() >= len(clipped):
                break
            if mask.sum() < max(10, int(0.1 * len(d))):
                break   # non rimuovere troppi punti
            clipped = clipped[mask]
        return float(np.mean(clipped))

    else:
        raise ValueError(
            f"method='{method}' non riconosciuto. "
            "Valori validi: 'sigma_clip', 'histogram', 'mean'."
        )


# ── modello ───────────────────────────────────────────────────────────────────

def _mu(
    t: np.ndarray,
    t0: float,
    k: float,
    a: float,
    b: float,
    tau_rise: float = 0.0,
) -> np.ndarray:
    """
    Valor atteso del modello. Gestisce overflow per b piccolo o t grande.

        mu(t) = k                                                            per t <= t0
        mu(t) = k - a * (1 - exp(-(t-t0)/tau_rise)) * exp(-(t-t0)/b)  per t > t0

    Per tau_rise = 0: onset istantaneo (modello originale).
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

    return np.clip(k - decay, 1e-9, None)


# ── LLR puntuale (solo per raffinamento) ─────────────────────────────────────

def _llr_exact(
    data: np.ndarray,
    t0: int,
    k: float,
    a: float,
    b: float,
    tau_rise: float = 0.0,
) -> float:
    """LLR esatto su finestra [t0−3b, t0 + 10·max(b, tau_rise)]."""
    n   = len(data)
    b_w = max(b, tau_rise) if tau_rise > 0 else b
    i0  = max(0, t0 - int(3 * b))
    i1  = min(n - 1, t0 + int(10 * b_w))
    t   = np.arange(i0, i1 + 1, dtype=np.float64)
    nd  = data[i0 : i1 + 1]
    mu  = _mu(t, float(t0), k, a, b, tau_rise)
    v   = float(np.sum(nd * np.log(mu / k) - (mu - k)))
    return v if np.isfinite(v) else -np.inf


# ── filtro IIR anti-causale ───────────────────────────────────────────────────

def _anticausal(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    y[t] = Σ_{s≥1} x[t+s] · αˢ  per ogni t in O(N).

    Applica IIR causale su x ribaltato, ribalta il risultato,
    poi sottrae il termine s=0 che il filtro include di default.
    """
    y = lfilter(
        np.array([1.0]),
        np.array([1.0, -alpha]),
        x[::-1].astype(np.float64),
    )[::-1]
    return y - x.astype(np.float64)


# ── profilo LLR vettorizzato ──────────────────────────────────────────────────

def _llr_profile(
    data: np.ndarray,
    k: float,
    a_values: np.ndarray,
    b_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ritorna (profile, best_a, best_b) di lunghezza N dove
    profile[t0] = max_{a,b} LLR(t0, a, b).

    Scansione con onset istantaneo (tau_rise = 0).
    Per ogni b, i termini di Taylor vengono calcolati una volta sola
    e riusati per tutti gli a → O(N · Nb · ORDER) totale.
    """
    N       = len(data)
    profile = np.full(N, -np.inf)
    best_a  = np.zeros(N)
    best_b  = np.zeros(N)
    ones    = np.ones(N, dtype=np.float64)

    for b_val in b_values:
        alpha = float(np.exp(-1.0 / b_val))

        # Taylor: Cₙ[t] = Σ_{s≥1} data[t+s] · α^(n·s)
        taylor = []
        alp_n  = alpha
        for _ in range(_TAYLOR_ORDER):
            taylor.append(_anticausal(data, alp_n))
            alp_n *= alpha

        s1 = _anticausal(ones, alpha)   # S1[t] = Σ_{s≥1} αˢ

        for a_val in a_values:
            if a_val >= k:
                continue

            r   = a_val / k
            llr = a_val * s1        # termine lineare
            rn  = r
            for n, cn in enumerate(taylor, start=1):
                llr -= (rn / n) * cn    # termini log
                rn  *= r

            better          = llr > profile
            profile[better] = llr[better]
            best_a[better]  = a_val
            best_b[better]  = b_val

    return profile, best_a, best_b


# ── raffinamento MLE ─────────────────────────────────────────────────────────

def _refine(
    data: np.ndarray,
    t0: int,
    k: float,
    a0: float,
    b0: float,
    tau_rise_max: float = 0.0,
) -> dict:
    """
    Affina (a, b) — o (a, b, tau_rise) se tau_rise_max > 0 — con Nelder-Mead.

    Se tau_rise_max > 0, tau_rise viene stimato come parametro libero
    nell'intervallo [0, tau_rise_max].

    Ritorna dict con a, b, tau_rise, llr, a_err, b_err, tau_rise_err.
    """
    use_tau = tau_rise_max > 0.0

    if use_tau:
        def neg(p):
            a, b, tr = p
            if a <= 0 or a >= k or b <= 0.1 or tr < 0 or tr > tau_rise_max:
                return 1e12
            return -_llr_exact(data, t0, k, a, b, tr)
        x0 = [a0, b0, min(tau_rise_max * 0.3, b0 * 0.2)]
    else:
        def neg(p):
            a, b = p
            if a <= 0 or a >= k or b <= 0.1:
                return 1e12
            return -_llr_exact(data, t0, k, a, b)
        x0 = [a0, b0]

    res = minimize(
        neg, x0,
        method="Nelder-Mead",
        options={"xatol": 0.05, "fatol": 0.05, "maxiter": _REFINE_MAX_ITER},
    )

    a_fit  = float(np.clip(res.x[0], 1e-3, k * 0.9999))
    b_fit  = float(np.clip(res.x[1], 0.1, None))
    tr_fit = float(np.clip(res.x[2], 0.0, tau_rise_max)) if use_tau else 0.0
    llr_fit = -float(res.fun)

    # incertezze 1-sigma dall'Hessiano numerico
    a_err = b_err = tr_err = np.nan
    try:
        da = max(a_fit  * 0.02, 0.05)
        db = max(b_fit  * 0.02, 0.05)

        if use_tau:
            dt = max(tr_fit * 0.05, 0.5) if tr_fit > 0 else max(tau_rise_max * 0.02, 0.5)
            f00  = neg([a_fit,      b_fit,      tr_fit     ])
            fpa  = neg([a_fit + da, b_fit,      tr_fit     ])
            fma  = neg([a_fit - da, b_fit,      tr_fit     ])
            fpb  = neg([a_fit,      b_fit + db, tr_fit     ])
            fmb  = neg([a_fit,      b_fit - db, tr_fit     ])
            fpt  = neg([a_fit,      b_fit,      tr_fit + dt])
            fmt  = neg([a_fit,      b_fit,      tr_fit - dt])
            h00  = (fpa - 2*f00 + fma) / da**2
            h11  = (fpb - 2*f00 + fmb) / db**2
            h22  = (fpt - 2*f00 + fmt) / dt**2
            h01  = (neg([a_fit+da, b_fit+db, tr_fit]) - neg([a_fit+da, b_fit-db, tr_fit])
                    - neg([a_fit-da, b_fit+db, tr_fit]) + neg([a_fit-da, b_fit-db, tr_fit])
                    ) / (4 * da * db)
            H    = np.array([[h00, h01, 0.0],
                              [h01, h11, 0.0],
                              [0.0, 0.0, h22]])
            cov  = np.linalg.inv(H)
            if cov[0, 0] > 0: a_err  = float(np.sqrt(cov[0, 0]))
            if cov[1, 1] > 0: b_err  = float(np.sqrt(cov[1, 1]))
            if cov[2, 2] > 0: tr_err = float(np.sqrt(cov[2, 2]))
        else:
            f00 = neg([a_fit,      b_fit     ])
            fpa = neg([a_fit + da, b_fit     ])
            fma = neg([a_fit - da, b_fit     ])
            fpb = neg([a_fit,      b_fit + db])
            fmb = neg([a_fit,      b_fit - db])
            fpp = neg([a_fit + da, b_fit + db])
            fpm = neg([a_fit + da, b_fit - db])
            fmp = neg([a_fit - da, b_fit + db])
            fmm = neg([a_fit - da, b_fit - db])
            h00 = (fpa - 2*f00 + fma) / da**2
            h11 = (fpb - 2*f00 + fmb) / db**2
            h01 = (fpp - fpm - fmp + fmm) / (4 * da * db)
            cov = np.linalg.inv(np.array([[h00, h01], [h01, h11]]))
            if cov[0, 0] > 0: a_err = float(np.sqrt(cov[0, 0]))
            if cov[1, 1] > 0: b_err = float(np.sqrt(cov[1, 1]))

    except Exception:
        pass

    return {
        "a": a_fit, "b": b_fit, "tau_rise": tr_fit, "llr": llr_fit,
        "a_err": a_err, "b_err": b_err, "tau_rise_err": tr_err,
    }


# ── griglie di default ────────────────────────────────────────────────────────

def _default_grids(
    k: float,
    dt_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Griglie di default per la ricerca di Forbush Decrease.

    b_values : 11 valori log-uniformi da 12 ore a 10 giorni,
               convertiti in campioni usando dt_seconds.
               Valori < 2 campioni vengono scartati automaticamente.
    a_values : 16 valori log-uniformi da 0.5% a 25% di k.

    Parametri
    ---------
    k          : livello di fondo (conteggi/campione)
    dt_seconds : durata di ogni campione in secondi (default: 1.0)
                 Esempi: 3600 per 1h, 600 per 10min, 60 per 1min.
    """
    # b: da 12h a 10 giorni, convertito in campioni
    b_hours  = np.array([12, 18, 24, 36, 48, 60, 72, 96, 120, 168, 240], dtype=float)
    sph      = 3600.0 / dt_seconds          # campioni per ora
    b_values = b_hours * sph
    b_values = b_values[b_values >= 2.0]    # scarta valori troppo brevi
    if len(b_values) == 0:
        # fallback per dt molto grande (es. 1 giorno)
        b_values = np.array([2.0, 5.0, 10.0, 20.0, 50.0, 100.0], dtype=float)

    # a: da 0.5% a 25% di k
    a_min    = max(0.3, k * 0.005)
    a_values = np.geomspace(a_min, k * 0.25, 16)

    return a_values, b_values


# ── API pubblica ──────────────────────────────────────────────────────────────

def calibrate_threshold(
    k: float,
    N: int,
    n_cal: int = _N_CAL_DEFAULT,
    a_values: np.ndarray | None = None,
    b_values: np.ndarray | None = None,
    fpr: float = 0.01,
    n_sim: int = 500,
    seed: int | None = None,
    dt_seconds: float = 1.0,
) -> float:
    """
    Stima la soglia LLR per il tasso di falsi positivi globale `fpr`
    sull'intera serie di lunghezza N.

    Correzione di Bonferroni
    ------------------------
    fpr_locale = fpr / (N / n_cal)

    Parametri
    ---------
    k           : valor medio del fondo
    N           : lunghezza della serie reale
    n_cal       : lunghezza delle serie simulate (default: 50 000)
    fpr         : false positive rate GLOBALE desiderato
    n_sim       : simulazioni Monte Carlo
    seed        : seme per riproducibilità
    dt_seconds  : durata del campione in secondi (usato per la griglia b di default)

    Ritorna
    -------
    threshold : float
    """
    rng = np.random.default_rng(seed)
    if a_values is None: a_values, _ = _default_grids(k, dt_seconds)
    if b_values is None: _, b_values = _default_grids(k, dt_seconds)

    n_windows  = max(1, N / n_cal)
    fpr_locale = fpr / n_windows
    quantile   = 1.0 - fpr_locale

    max_llrs = []
    for _ in range(n_sim):
        null = rng.poisson(k, n_cal).astype(np.float64)
        prof, _, _ = _llr_profile(null, float(null.mean()), a_values, b_values)
        max_llrs.append(float(prof.max()))

    return float(np.quantile(max_llrs, quantile))


def find_signals(
    data: np.ndarray,
    threshold: float | None = None,
    a_values: np.ndarray | None = None,
    b_values: np.ndarray | None = None,
    k: float | None = None,
    k_method: str = 'sigma_clip',
    dt_seconds: float = 1.0,
    tau_rise_max: float = 0.0,
    refine: bool = True,
    max_signals: int = 50,
) -> list[dict]:
    """
    Trova tutti i segnali exp-drop nella serie temporale.

    Strategia greedy iterativa
    --------------------------
    1. Calcola profilo LLR(t0) = max_{a,b} LLR(t0,a,b) (scansione con tau_rise=0).
    2. Prendi il picco. Se LLR < threshold: fine.
    3. Affina (a, b) — e opzionalmente tau_rise — con Nelder-Mead.
    4. Sottrai il contributo anomalo dal segnale trovato dai residui.
    5. Ripeti dal passo 1.

    Parametri
    ---------
    data          : array 1D di conteggi (int o float)
    threshold     : soglia LLR. Usa calibrate_threshold() per una stima
                    precisa con correzione Bonferroni; se None usa
                    l'approssimazione chi2(df=2) p=0.01 (non corretta per N).
    a_values      : griglia di a (default: 16 valori log-uniformi in [0.5%k, 25%k])
    b_values      : griglia di b (default: 12h–10 giorni in campioni)
    k             : fondo (default: stimato automaticamente con k_method)
    k_method      : 'sigma_clip' (default) | 'histogram' | 'mean'
                    Metodo per la stima automatica del fondo. 'sigma_clip' e
                    'histogram' sono robusti alla presenza di Forbush Decrease.
    dt_seconds    : durata del campione in secondi (per la griglia b di default)
    tau_rise_max  : scala temporale massima dell'onset (campioni).
                    0 (default) = onset istantaneo (tau_rise non stimato).
                    Se > 0, tau_rise viene stimato con Nelder-Mead nel range
                    [0, tau_rise_max].
    refine        : se True, affina (a, b[, tau_rise]) con Nelder-Mead
    max_signals   : limite massimo di iterazioni greedy

    Ritorna
    -------
    Lista di dict ordinata per t0 crescente, ognuno con:
        t0           : int   — indice campione dell'onset
        a            : float — ampiezza del calo (conteggi)
        b            : float — scala temporale di recupero (campioni)
        tau_rise     : float — scala temporale di onset (campioni; 0 se non stimato)
        llr          : float — log-likelihood ratio
        a_err        : float — incertezza 1-sigma su a  (nan se refine=False)
        b_err        : float — incertezza 1-sigma su b  (nan se refine=False)
        tau_rise_err : float — incertezza 1-sigma su tau_rise (nan se non stimato)
        k            : float — valore di k usato
    """
    data  = np.asarray(data, dtype=np.float64)
    N     = len(data)
    t     = np.arange(N, dtype=np.float64)

    if k is None:
        k_est = estimate_background(data, method=k_method)
    else:
        k_est = float(k)

    if a_values is None: a_values, _ = _default_grids(k_est, dt_seconds)
    if b_values is None: _, b_values = _default_grids(k_est, dt_seconds)
    a_values = np.asarray(a_values, dtype=np.float64)
    b_values = np.asarray(b_values, dtype=np.float64)

    if threshold is None:
        from scipy.stats import chi2
        threshold = float(chi2.ppf(0.99, df=2) / 2)
        warnings.warn(
            f"Soglia non fornita: uso chi2(df=2) → {threshold:.2f}. "
            "Questo non corregge per la lunghezza della serie. "
            "Usa calibrate_threshold(k, N=len(data), dt_seconds=...) "
            "per una soglia precisa.",
            UserWarning, stacklevel=2,
        )

    residuals = data.copy()
    found: list[dict] = []

    for _ in range(max_signals):
        profile, ba, bb = _llr_profile(residuals, k_est, a_values, b_values)

        t0_best  = int(np.argmax(profile))
        llr_best = float(profile[t0_best])

        if llr_best < threshold:
            break

        a0 = float(ba[t0_best])
        b0 = float(bb[t0_best])

        if refine:
            fit = _refine(residuals, t0_best, k_est, a0, b0, tau_rise_max)
        else:
            fit = {
                "a": a0, "b": b0, "tau_rise": 0.0,
                "llr": llr_best,
                "a_err": np.nan, "b_err": np.nan, "tau_rise_err": np.nan,
            }

        found.append({
            "t0":           t0_best,
            "a":            fit["a"],
            "b":            fit["b"],
            "tau_rise":     fit["tau_rise"],
            "llr":          fit["llr"],
            "a_err":        fit["a_err"],
            "b_err":        fit["b_err"],
            "tau_rise_err": fit["tau_rise_err"],
            "k":            k_est,
        })

        # rimuovi il contributo anomalo del segnale trovato dai residui
        mu_sig    = _mu(t, float(t0_best), k_est, fit["a"], fit["b"], fit["tau_rise"])
        residuals = residuals - (mu_sig - k_est)

    found.sort(key=lambda r: r["t0"])
    return found
