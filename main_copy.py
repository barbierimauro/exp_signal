"""
find_signals.py
===============
Cerca segnali della forma

    mu(t) = k                           per t <= t0
    mu(t) = k - a * exp(-(t - t0) / b) per t > t0

in serie temporali con rumore di Poisson, anche con N = 1 000 000 punti.

─── Algoritmo ────────────────────────────────────────────────────────────────

Per ogni coppia (a, b) della griglia, il profilo LLR su TUTTI i t0
viene calcolato in O(N) con filtri IIR anti-causali, evitando qualsiasi
loop Python su t0.

La formula LLR si espande in serie di Taylor:

    LLR(t0) = a · S1(t0) − Σₙ₌₁ᴼᴿᴰᴱᴿ (a/k)ⁿ/n · Cₙ(t0)

dove Cₙ(t0) = Σ_{s≥1} data[t0+s] · exp(−ns/b) è un filtro IIR con polo
αⁿ = exp(−n/b), calcolato una sola volta per ogni b e riusato per tutti gli a.

Complessità: O(N · Nb · ORDER) ≈ 4–18 s su 1M punti (PC modesto, 1 core).

─── Correzione FPR per serie lunghe ──────────────────────────────────────────

La soglia va calibrata su una serie lunga almeno N/20. Con n_cal << N,
il test viene eseguito su ~N/n_cal finestre quasi-indipendenti e il FPR
effettivo è circa 1 − (1 − fpr)^(N/n_cal).

calibrate_threshold() applica automaticamente la correzione di Bonferroni:
usa fpr_target = fpr / (N / n_cal) per mantenere il FPR globale nominale.

─── Funzioni pubbliche ────────────────────────────────────────────────────────

    calibrate_threshold(k, N, n_cal, fpr, ...)  →  float
    find_signals(data, threshold, ...)          →  list[dict]

Ogni dict ha le chiavi: t0, a, b, llr, a_err, b_err, k

─── Uso rapido ───────────────────────────────────────────────────────────────

    import numpy as np
    from find_signals import calibrate_threshold, find_signals

    threshold = calibrate_threshold(k=30.0, N=len(data), seed=0)
    results   = find_signals(data, threshold=threshold)

    for r in results:
        print(f"t0={r['t0']}  a={r['a']:.2f}±{r['a_err']:.2f}"
              f"  b={r['b']:.2f}±{r['b_err']:.2f}  llr={r['llr']:.1f}")
"""

import warnings
import numpy as np
from scipy.signal import lfilter
from scipy.optimize import minimize


# ── costanti ──────────────────────────────────────────────────────────────────

_TAYLOR_ORDER    = 8     # ordine espansione log(1−x); errore < 0.02% per a/k < 0.95
_REFINE_MAX_ITER = 300
_N_CAL_DEFAULT   = 50_000


# ── modello ───────────────────────────────────────────────────────────────────

def _mu(t: np.ndarray, t0: float, k: float, a: float, b: float) -> np.ndarray:
    """Valor atteso del modello. Gestisce overflow per b piccolo o t grande."""
    exponent = np.where(t <= t0, 0.0, -(t - t0) / b)
    # clip a -500 evita overflow su exp senza alterare i valori fisici
    decay = np.where(t <= t0, 0.0, a * np.exp(np.clip(exponent, -500, 0)))
    return np.clip(k - decay, 1e-9, None)


# ── LLR puntuale (solo per raffinamento) ─────────────────────────────────────

def _llr_exact(data: np.ndarray, t0: int, k: float, a: float, b: float) -> float:
    """LLR esatto su finestra [t0−3b, t0+10b]."""
    n  = len(data)
    i0 = max(0, t0 - int(3 * b))
    i1 = min(n - 1, t0 + int(10 * b))
    t  = np.arange(i0, i1 + 1, dtype=np.float64)
    nd = data[i0 : i1 + 1]
    mu = _mu(t, float(t0), k, a, b)
    v  = float(np.sum(nd * np.log(mu / k) - (mu - k)))
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
) -> dict:
    """
    Affina (a, b) con Nelder-Mead partendo dalla stima su griglia.
    Ritorna dict con a, b, llr, a_err, b_err.
    """
    def neg(p):
        a, b = p
        if a <= 0 or a >= k or b <= 0.1:
            return 1e12
        return -_llr_exact(data, t0, k, a, b)

    res = minimize(
        neg, [a0, b0],
        method="Nelder-Mead",
        options={"xatol": 0.05, "fatol": 0.05, "maxiter": _REFINE_MAX_ITER},
    )

    a_fit   = float(np.clip(res.x[0], 1e-3, k * 0.9999))
    b_fit   = float(np.clip(res.x[1], 0.1, None))
    llr_fit = -float(res.fun)

    # incertezze 1-sigma dall'Hessiano numerico 2×2
    a_err = b_err = np.nan
    try:
        da  = max(a_fit * 0.02, 0.05)
        db  = max(b_fit * 0.02, 0.05)
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

    return {"a": a_fit, "b": b_fit, "llr": llr_fit, "a_err": a_err, "b_err": b_err}


# ── griglie di default ────────────────────────────────────────────────────────

def _default_grids(k: float) -> tuple[np.ndarray, np.ndarray]:
    a_min    = max(0.3, k * 0.02)
    a_values = np.geomspace(a_min, k * 0.93, 14)
    b_values = np.array([2, 3, 5, 7, 10, 14, 20, 28, 40, 55, 75, 100], dtype=float)
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
) -> float:
    """
    Stima la soglia LLR per il tasso di falsi positivi globale `fpr`
    sull'intera serie di lunghezza N.

    Correzione di Bonferroni
    ------------------------
    Le simulazioni usano serie di lunghezza n_cal (default 50 000) per
    velocità. Poiché la serie reale ha N >> n_cal campioni, il test viene
    eseguito su circa N/n_cal finestre quasi-indipendenti.
    Per mantenere il FPR globale nominale, viene usato:

        fpr_locale = fpr / (N / n_cal)

    Esempio: fpr=0.01, N=1M, n_cal=50k → fpr_locale = 0.0005 (soglia al 99.95°).
    La soglia è stabile rispetto a n_cal per n_cal ≥ 50 000.

    Parametri
    ---------
    k       : valor medio del fondo
    N       : lunghezza della serie reale (necessario per la correzione)
    n_cal   : lunghezza delle serie simulate (50 000 è sufficiente)
    fpr     : false positive rate GLOBALE desiderato
    n_sim   : simulazioni Monte Carlo (500 → ~60 s su un core)
    seed    : seme per riproducibilità

    Ritorna
    -------
    threshold : float
    """
    rng = np.random.default_rng(seed)
    if a_values is None: a_values, _ = _default_grids(k)
    if b_values is None: _, b_values = _default_grids(k)

    # correzione Bonferroni: soglia più alta per serie lunghe
    n_windows  = max(1, N / n_cal)
    fpr_locale = fpr / n_windows
    quantile   = 1.0 - fpr_locale

    max_llrs = []
    for _ in range(n_sim):
        null = rng.poisson(k, n_cal).astype(np.float64)
        prof, _, _ = _llr_profile(null, float(null.mean()), a_values, b_values)
        max_llrs.append(float(prof.max()))

    threshold = float(np.quantile(max_llrs, quantile))
    return threshold


def find_signals(
    data: np.ndarray,
    threshold: float | None = None,
    a_values: np.ndarray | None = None,
    b_values: np.ndarray | None = None,
    k: float | None = None,
    refine: bool = True,
    max_signals: int = 50,
) -> list[dict]:
    """
    Trova tutti i segnali exp-drop nella serie temporale.

    Strategia greedy iterativa
    --------------------------
    1. Calcola profilo LLR(t0) = max_{a,b} LLR(t0,a,b) sulla serie residua.
    2. Prendi il picco. Se LLR < threshold: fine.
    3. Affina (a, b) con Nelder-Mead.
    4. Sottrai il contributo anomalo del segnale: residui −= μ(t) − k.
    5. Ripeti dal passo 1.

    La sottrazione al passo 4 impedisce che segnali forti distorcano
    la stima dei segnali vicini nelle iterazioni successive.

    Nota sulle incertezze
    ---------------------
    a e b hanno una degenerazione parziale: il prodotto a·b (area del calo)
    è meglio vincolato dei singoli parametri. Le incertezze 1-sigma
    riflettono questa degenerazione e possono essere grandi per b >> 1
    o SNR basso.

    Parametri
    ---------
    data        : array 1D di conteggi (int o float)
    threshold   : soglia LLR. Usa calibrate_threshold() per una stima
                  precisa con correzione Bonferroni; se None usa
                  l'approssimazione chi2(df=2) p=0.01 (non corretta per N).
    a_values    : griglia di a (default: 14 valori log-uniformi in [0.02k, 0.93k])
    b_values    : griglia di b (default: [2,3,5,7,10,14,20,28,40,55,75,100])
    k           : valor medio del fondo (default: media dei dati)
    refine      : se True, affina (a,b) con Nelder-Mead dopo la griglia
    max_signals : limite massimo di iterazioni greedy

    Ritorna
    -------
    Lista di dict ordinata per t0 crescente, ognuno con:
        t0    : int   — indice della discontinuità
        a     : float — ampiezza del calo
        b     : float — scala temporale del recupero (campioni)
        llr   : float — log-likelihood ratio
        a_err : float — incertezza 1-sigma su a  (nan se refine=False)
        b_err : float — incertezza 1-sigma su b  (nan se refine=False)
        k     : float — valore di k usato
    """
    data  = np.asarray(data, dtype=np.float64)
    N     = len(data)
    t     = np.arange(N, dtype=np.float64)

    k_est = float(data.mean()) if k is None else float(k)
    if a_values is None: a_values, _ = _default_grids(k_est)
    if b_values is None: _, b_values = _default_grids(k_est)
    a_values = np.asarray(a_values, dtype=np.float64)
    b_values = np.asarray(b_values, dtype=np.float64)

    if threshold is None:
        from scipy.stats import chi2
        threshold = float(chi2.ppf(0.99, df=2) / 2)
        warnings.warn(
            f"Soglia non fornita: uso chi2(df=2) → {threshold:.2f}. "
            "Questo non corregge per la lunghezza della serie. "
            "Usa calibrate_threshold(k, N=len(data)) per una soglia precisa.",
            UserWarning, stacklevel=2,
        )

    residuals = data.copy()
    found     = []

    for _ in range(max_signals):
        profile, ba, bb = _llr_profile(residuals, k_est, a_values, b_values)

        t0_best  = int(np.argmax(profile))
        llr_best = float(profile[t0_best])

        if llr_best < threshold:
            break

        a0 = float(ba[t0_best])
        b0 = float(bb[t0_best])

        if refine:
            fit = _refine(residuals, t0_best, k_est, a0, b0)
        else:
            fit = {"a": a0, "b": b0, "llr": llr_best, "a_err": np.nan, "b_err": np.nan}

        found.append({
            "t0":    t0_best,
            "a":     fit["a"],
            "b":     fit["b"],
            "llr":   fit["llr"],
            "a_err": fit["a_err"],
            "b_err": fit["b_err"],
            "k":     k_est,
        })

        # rimuovi il contributo anomalo del segnale trovato dai residui
        mu_sig    = _mu(t, float(t0_best), k_est, fit["a"], fit["b"])
        residuals = residuals - (mu_sig - k_est)

    found.sort(key=lambda r: r["t0"])
    return found
