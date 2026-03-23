"""
io_utils.py
===========
Lettura di serie temporali da file CSV con colonna datetime ISO 8601.

Funzioni pubbliche
------------------
    load_csv(path, datetime_col, series1_col, ...)
        -> (timestamps, data1, data2_or_None, info)

    enrich_signals_with_time(signals, timestamps)
        -> signals  (aggiunge t0_datetime e t0_mjd a ogni segnale in-place)

Formato CSV atteso
------------------
Il file deve avere una riga di intestazione. Le colonne possono essere
in qualsiasi ordine; vengono selezionate per nome::

    timestamp,muons,neutrons,pressure_hPa
    2024-01-01T00:00:00,10234,1045,1013.2
    2024-01-01T01:00:00,10189,1031,1012.8
    ...

Correzione di pressione (opzionale)
------------------------------------
Se ``pressure_col`` è specificato, viene applicata la correzione barometrica:

    N_corr(t) = N_raw(t) * exp(β * (P(t) - P_ref))

dove β è il coefficiente barometrico (hPa⁻¹, tipicamente negativo per muoni
e neutroni: aumento di pressione → calo dei conteggi), e P_ref è la pressione
di riferimento (default: media temporale).

Formati datetime supportati
----------------------------
    2024-01-01T12:00:00
    2024-01-01T12:00:00.123456
    2024-01-01T12:00:00+02:00      (timezone ignorata, UTC assunta)
    2024-01-01 12:00:00
    2024-01-01
    01/01/2024 12:00:00
"""

import csv
import re
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


# ── epoca MJD ─────────────────────────────────────────────────────────────────

_MJD_EPOCH = datetime(1858, 11, 17)   # MJD = 0 corrisponde a 1858-11-17 00:00:00


def _to_mjd(dt: datetime) -> float:
    """
    Converte datetime in Modified Julian Date.

    MJD = JD − 2400000.5
    MJD = 0 corrisponde a 1858-11-17 00:00:00 UTC.
    """
    return (dt - _MJD_EPOCH).total_seconds() / 86400.0


# ── parser datetime robusto ───────────────────────────────────────────────────

_ISO_TZ_RE = re.compile(r"([+-]\d{2}:?\d{2}|Z)$")
_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
]


def _parse_dt(s: str) -> datetime:
    s = s.strip()
    try:
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=None)
    except ValueError:
        pass
    s_clean = _ISO_TZ_RE.sub("", s).strip()
    for fmt in _FORMATS:
        try:
            return datetime.strptime(s_clean, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Impossibile interpretare la data '{s}'. "
        f"Usa il formato ISO 8601, es. '2024-01-15T08:30:00'."
    )


# ── arricchimento risultati con datetime e MJD ────────────────────────────────

def enrich_signals_with_time(
    signals,
    timestamps: np.ndarray,
):
    """
    Aggiunge ``t0_datetime`` (stringa ISO) e ``t0_mjd`` (float) a ogni segnale,
    usando l'indice ``t0`` per indicizzare ``timestamps``.

    Accetta sia una lista di dict (canale singolo) sia un dict di liste
    (formato restituito da find_signals_dual).

    Parametri
    ---------
    signals    : list[dict] oppure dict[str, list[dict]]
    timestamps : array datetime64[ms] restituito da load_csv

    Ritorna
    -------
    Lo stesso oggetto modificato in-place.
    """
    if isinstance(signals, dict):
        for sigs in signals.values():
            enrich_signals_with_time(sigs, timestamps)
        return signals

    epoch = datetime(1970, 1, 1)
    for s in signals:
        idx = s.get("t0")
        if idx is None or idx < 0 or idx >= len(timestamps):
            continue
        ms = int(timestamps[idx].astype(np.int64))
        dt = epoch + timedelta(milliseconds=ms)
        s["t0_datetime"] = dt.strftime("%Y-%m-%d %H:%M:%S")
        s["t0_mjd"]      = _to_mjd(dt)
    return signals


# ── funzione principale ───────────────────────────────────────────────────────

def load_csv(
    path: str | Path,
    datetime_col: str,
    series1_col:  str,
    series2_col:  str | None = None,
    delimiter:    str  = ",",
    k1:           float | None = None,
    k2:           float | None = None,
    k_method:     str  = 'sigma_clip',  # 'sigma_clip' | 'histogram' | 'mean'
    fill_nan:     str  = "raise",       # "raise" | "interpolate" | "drop"
    pressure_col: str | None  = None,   # colonna pressione atmosferica (hPa)
    barometric_coeff1: float  = 0.0,    # β per series1 (hPa⁻¹; 0 = no correzione)
    barometric_coeff2: float  = 0.0,    # β per series2 (hPa⁻¹; 0 = no correzione)
    pressure_ref: float | None = None,  # pressione di riferimento; None = media temporale
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """
    Legge un CSV con colonna datetime ISO 8601 e una o due serie numeriche.

    Parametri
    ---------
    path              : percorso del file CSV
    datetime_col      : nome della colonna datetime
    series1_col       : nome della prima colonna numerica
    series2_col       : nome della seconda colonna (None → single-channel)
    delimiter         : separatore di campo (default: ',')
    k1, k2            : fondo noto (None → stimato automaticamente)
    k_method          : metodo di stima del fondo ('sigma_clip' | 'histogram' | 'mean')
                        'sigma_clip' e 'histogram' sono robusti alla presenza di FD.
    fill_nan          : cosa fare con valori mancanti:
                          "raise"       → lancia ValueError (default)
                          "interpolate" → interpolazione lineare
                          "drop"        → scarta le righe con NaN
    pressure_col      : nome della colonna pressione atmosferica in hPa.
                        Se specificata, viene applicata la correzione barometrica.
    barometric_coeff1 : β₁ (hPa⁻¹) per series1.
                        Tipico per muoni: da −0.001 a −0.003 hPa⁻¹.
                        0 = nessuna correzione (default).
    barometric_coeff2 : β₂ (hPa⁻¹) per series2. Analogo a barometric_coeff1.
    pressure_ref      : pressione di riferimento in hPa.
                        None (default) = media temporale della serie di pressione.

    Ritorna
    -------
    timestamps : ndarray datetime64[ms], shape (N,)
    data1      : ndarray float64, shape (N,)  — corretta per pressione se richiesto
    data2      : ndarray float64, shape (N,) oppure None
    info       : dict con:
                   N                 numero di campioni
                   dt_seconds        intervallo di campionamento mediano (s)
                   sampling_seconds  alias di dt_seconds (backward compat)
                   sampling_std_s    deviazione standard dell'intervallo (s)
                   irregular         True se il campionamento non è uniforme
                   start             timestamp iniziale (stringa ISO)
                   end               timestamp finale (stringa ISO)
                   k1                livello di fondo serie 1
                   k2                livello di fondo serie 2 (se presente)
                   missing1          numero di valori mancanti serie 1
                   missing2          numero di valori mancanti serie 2
                   pressure_ref      pressione di riferimento usata (se corretto)
    """
    from main import estimate_background

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    # ── lettura grezza ────────────────────────────────────────────────────────
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("Il file CSV sembra vuoto o privo di intestazione.")
        fieldnames = [f.strip() for f in reader.fieldnames]
        reader.fieldnames = fieldnames

        _check_col(datetime_col, fieldnames, path)
        _check_col(series1_col,  fieldnames, path)
        if series2_col is not None:
            _check_col(series2_col, fieldnames, path)
        if pressure_col is not None:
            _check_col(pressure_col, fieldnames, path)

        raw_dt, raw1, raw2, raw_p = [], [], [], []
        for row in reader:
            raw_dt.append(row[datetime_col].strip())
            raw1.append(row[series1_col].strip())
            if series2_col is not None:
                raw2.append(row[series2_col].strip())
            if pressure_col is not None:
                raw_p.append(row[pressure_col].strip())

    if len(raw_dt) == 0:
        raise ValueError("Il file CSV non contiene righe di dati.")

    # ── parse datetime ────────────────────────────────────────────────────────
    datetimes: list[datetime] = []
    parse_errors: list[int] = []
    for i, s in enumerate(raw_dt):
        try:
            datetimes.append(_parse_dt(s))
        except ValueError:
            parse_errors.append(i + 2)
    if parse_errors:
        sample = parse_errors[:5]
        raise ValueError(
            f"Impossibile interpretare {len(parse_errors)} valori datetime "
            f"(righe CSV: {sample}{'...' if len(parse_errors) > 5 else ''})."
        )

    # ── parse numeri ──────────────────────────────────────────────────────────
    def _to_float_array(raw: list[str], col_name: str) -> np.ndarray:
        arr = np.empty(len(raw), dtype=np.float64)
        for i, s in enumerate(raw):
            if s in ("", "nan", "NaN", "NA", "N/A", "null", "NULL", "None"):
                arr[i] = np.nan
            else:
                try:
                    arr[i] = float(s.replace(",", "."))
                except ValueError:
                    raise ValueError(
                        f"Colonna '{col_name}', riga {i+2}: "
                        f"valore non numerico '{s}'."
                    )
        return arr

    arr1 = _to_float_array(raw1, series1_col)
    arr2 = _to_float_array(raw2, series2_col) if raw2 else None
    arrP = _to_float_array(raw_p, pressure_col) if raw_p else None

    # ── gestione NaN ──────────────────────────────────────────────────────────
    nan1 = int(np.isnan(arr1).sum())
    nan2 = int(np.isnan(arr2).sum()) if arr2 is not None else 0

    all_nan_mask = np.isnan(arr1)
    if arr2 is not None:
        all_nan_mask = all_nan_mask | np.isnan(arr2)

    if all_nan_mask.any():
        if fill_nan == "raise":
            n_bad = int(all_nan_mask.sum())
            raise ValueError(
                f"{n_bad} righe contengono valori mancanti. "
                "Usa fill_nan='interpolate' o fill_nan='drop' per gestirli."
            )
        elif fill_nan == "drop":
            keep = ~all_nan_mask
            datetimes = [datetimes[i] for i in range(len(datetimes)) if keep[i]]
            arr1 = arr1[keep]
            if arr2 is not None:
                arr2 = arr2[keep]
            if arrP is not None:
                arrP = arrP[keep]
        elif fill_nan == "interpolate":
            arr1 = _interpolate_nan(arr1)
            if arr2 is not None:
                arr2 = _interpolate_nan(arr2)
        else:
            raise ValueError(
                f"fill_nan non riconosciuto: '{fill_nan}'. "
                "Valori validi: 'raise', 'interpolate', 'drop'."
            )

    # ── correzione barometrica ────────────────────────────────────────────────
    p_ref_used: float | None = None
    if arrP is not None and (barometric_coeff1 != 0.0 or barometric_coeff2 != 0.0):
        # gestisci NaN nella pressione con interpolazione
        if np.any(np.isnan(arrP)):
            arrP = _interpolate_nan(arrP)
        p_ref_used = float(pressure_ref) if pressure_ref is not None else float(np.mean(arrP))
        dP = arrP - p_ref_used
        if barometric_coeff1 != 0.0:
            arr1 = arr1 * np.exp(barometric_coeff1 * dP)
        if barometric_coeff2 != 0.0 and arr2 is not None:
            arr2 = arr2 * np.exp(barometric_coeff2 * dP)

    # ── timestamp → numpy datetime64 ─────────────────────────────────────────
    epoch = datetimes[0]
    rel_s = np.array(
        [(dt - epoch).total_seconds() for dt in datetimes],
        dtype=np.float64,
    )
    epoch_ns = np.datetime64(epoch.isoformat(), "ms")
    timestamps = epoch_ns + (rel_s * 1000).astype("timedelta64[ms]")

    # ── campionamento ─────────────────────────────────────────────────────────
    if len(rel_s) > 1:
        diffs     = np.diff(rel_s)
        dt_median = float(np.median(diffs))
        dt_std    = float(np.std(diffs))
        irregular = bool(dt_std > 0.05 * dt_median)
    else:
        dt_median = 1.0
        dt_std    = 0.0
        irregular = False

    N = len(arr1)

    # ── fondi ─────────────────────────────────────────────────────────────────
    k1_val = float(k1) if k1 is not None else estimate_background(arr1, method=k_method)
    k2_val: float | None = None
    if arr2 is not None:
        k2_val = float(k2) if k2 is not None else estimate_background(arr2, method=k_method)

    # ── info dict ─────────────────────────────────────────────────────────────
    info: dict = {
        "N":                N,
        "dt_seconds":       dt_median,
        "sampling_seconds": dt_median,   # alias per backward compatibility
        "sampling_std_s":   dt_std,
        "irregular":        irregular,
        "start":            datetimes[0].isoformat(sep=" "),
        "end":              datetimes[-1].isoformat(sep=" "),
        "k1":               k1_val,
        "missing1":         nan1,
        "missing2":         nan2,
    }
    if k2_val is not None:
        info["k2"] = k2_val
    if p_ref_used is not None:
        info["pressure_ref"] = p_ref_used

    if irregular:
        import warnings
        warnings.warn(
            f"Campionamento non uniforme rilevato "
            f"(mediana={dt_median:.3f} s, σ={dt_std:.3f} s). "
            "Il detector assume campioni equidistanti: i risultati potrebbero "
            "essere influenzati dalle variazioni di campionamento.",
            UserWarning,
            stacklevel=2,
        )

    return timestamps, arr1, arr2, info


# ── helpers ───────────────────────────────────────────────────────────────────

def _check_col(name: str, fieldnames: list[str], path: Path) -> None:
    if name not in fieldnames:
        raise ValueError(
            f"Colonna '{name}' non trovata in {path.name}. "
            f"Colonne disponibili: {fieldnames}"
        )


def _interpolate_nan(arr: np.ndarray) -> np.ndarray:
    """Interpolazione lineare dei NaN interni; scarta NaN ai bordi."""
    x  = np.arange(len(arr))
    ok = ~np.isnan(arr)
    if ok.sum() < 2:
        raise ValueError("Troppi valori mancanti per l'interpolazione.")
    return np.interp(x, x[ok], arr[ok])


# ── helper per i grafici: ticks datetime sull'asse x ─────────────────────────

def datetime_axis(
    ax,
    timestamps: np.ndarray,
    max_ticks: int = 6,
) -> None:
    """
    Imposta l'asse x di ``ax`` con etichette datetime leggibili.

    Parametri
    ---------
    ax         : matplotlib Axes
    timestamps : array datetime64[ms] restituito da load_csv
    max_ticks  : numero massimo di tick (default 6)
    """
    import matplotlib.ticker as _ticker

    N     = len(timestamps)
    step  = max(1, N // max_ticks)
    idx   = np.arange(0, N, step)
    labels = [str(timestamps[i])[:19].replace("T", "\n") for i in idx]

    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=8, rotation=0, ha="center")
    ax.xaxis.set_minor_locator(_ticker.AutoMinorLocator())
