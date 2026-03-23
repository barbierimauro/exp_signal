"""
io_utils.py
===========
Lettura di serie temporali da file CSV con colonna datetime ISO 8601.

La funzione principale ``load_csv`` produce array numpy direttamente
compatibili con ``find_signals`` e ``find_signals_dual``.

Formato CSV atteso
------------------
Il file deve avere una riga di intestazione.  Le colonne possono essere
in qualsiasi ordine; vengono selezionate per nome::

    timestamp,series_A,series_B
    2024-01-01T00:00:00,102,87
    2024-01-01T00:01:00,98,91
    ...

Formati datetime supportati (riconosciuti automaticamente)
----------------------------------------------------------
    2024-01-01T12:00:00
    2024-01-01T12:00:00.123456
    2024-01-01T12:00:00+02:00      (timezone ignorata, UTC assunta)
    2024-01-01 12:00:00
    2024-01-01 12:00:00.123
    2024-01-01                     (solo data → mezzanotte)
    01/01/2024 12:00:00
    01/01/2024 12:00

Funzioni pubbliche
------------------
    load_csv(path, datetime_col, series1_col, series2_col, ...)
        -> (timestamps, data1, data2_or_None, info)
"""

import csv
import re
import numpy as np
from pathlib import Path
from datetime import datetime, timezone


# ── parser datetime robusto ───────────────────────────────────────────────────

_ISO_TZ_RE = re.compile(
    r"([+-]\d{2}:?\d{2}|Z)$"           # rimuove timezone prima del parse
)
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
    """
    Converte una stringa in datetime.  Gestisce le varianti ISO 8601
    più comuni, con o senza timezone, con T o spazio come separatore.
    """
    s = s.strip()

    # prova fromisoformat di Python (gestisce la maggior parte dei casi)
    try:
        dt = datetime.fromisoformat(s)
        # scarta timezone: lavoriamo sempre in tempo locale / relativo
        return dt.replace(tzinfo=None)
    except ValueError:
        pass

    # rimuovi timezone se presente (es. +02:00, Z)
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


# ── funzione principale ───────────────────────────────────────────────────────

def load_csv(
    path: str | Path,
    datetime_col: str,
    series1_col:  str,
    series2_col:  str | None = None,
    delimiter:    str  = ",",
    k1:           float | None = None,
    k2:           float | None = None,
    fill_nan:     str  = "raise",   # "raise" | "interpolate" | "drop"
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """
    Legge un CSV con colonna datetime ISO 8601 e una o due serie numeriche.

    Parametri
    ---------
    path         : percorso del file CSV
    datetime_col : nome della colonna datetime
    series1_col  : nome della prima colonna numerica
    series2_col  : nome della seconda colonna numerica (None → single-channel)
    delimiter    : separatore di campo (default: ',')
    k1, k2       : livello di fondo da usare (None → media della serie)
    fill_nan     : cosa fare con valori mancanti/non numerici:
                     "raise"       → lancia ValueError (default)
                     "interpolate" → interpolazione lineare
                     "drop"        → scarta le righe con NaN
                     (NaN all'inizio o alla fine vengono sempre scartati)

    Ritorna
    -------
    timestamps : ndarray di datetime64[ms], shape (N,)
    data1      : ndarray float64, shape (N,)
    data2      : ndarray float64, shape (N,) oppure None
    info       : dict con le chiavi:
                   N                 numero di campioni
                   sampling_seconds  intervallo di campionamento mediano (s)
                   sampling_std_s    deviazione standard dell'intervallo (s)
                   irregular         True se il campionamento non è uniforme
                   start             timestamp iniziale (stringa ISO)
                   end               timestamp finale (stringa ISO)
                   k1                livello di fondo serie 1
                   k2                livello di fondo serie 2 (se presente)
                   missing1          numero di valori mancanti serie 1
                   missing2          numero di valori mancanti serie 2

    Note
    ----
    I dati vengono passati a ``find_signals`` / ``find_signals_dual`` come
    array numpy; il campo ``info`` contiene i metadati per ricostruire
    l'asse temporale nei grafici.

    Esempio
    -------
    >>> ts, d1, d2, info = load_csv(
    ...     "data.csv",
    ...     datetime_col="timestamp",
    ...     series1_col="counts_A",
    ...     series2_col="counts_B",
    ... )
    >>> print(f"N={info['N']}  dt={info['sampling_seconds']:.1f} s")
    >>> threshold = calibrate_threshold(info['k1'], N=info['N'])
    >>> signals = find_signals(d1, threshold=threshold, k=info['k1'])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    # ── lettura grezza ────────────────────────────────────────────────────────
    with open(path, newline="", encoding="utf-8-sig") as fh:
        # utf-8-sig gestisce l'eventuale BOM di Excel
        reader = csv.DictReader(fh, delimiter=delimiter)

        if reader.fieldnames is None:
            raise ValueError("Il file CSV sembra vuoto o privo di intestazione.")

        # normalizza i nomi delle colonne (strip spazi)
        fieldnames = [f.strip() for f in reader.fieldnames]
        reader.fieldnames = fieldnames

        _check_col(datetime_col, fieldnames, path)
        _check_col(series1_col,  fieldnames, path)
        if series2_col is not None:
            _check_col(series2_col, fieldnames, path)

        raw_dt, raw1, raw2 = [], [], []
        for lineno, row in enumerate(reader, start=2):
            raw_dt.append(row[datetime_col].strip())
            raw1.append(row[series1_col].strip())
            if series2_col is not None:
                raw2.append(row[series2_col].strip())

    if len(raw_dt) == 0:
        raise ValueError("Il file CSV non contiene righe di dati.")

    # ── parse datetime ────────────────────────────────────────────────────────
    datetimes: list[datetime] = []
    parse_errors: list[int] = []
    for i, s in enumerate(raw_dt):
        try:
            datetimes.append(_parse_dt(s))
        except ValueError:
            parse_errors.append(i + 2)   # linea nel file (1-based + header)
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
        elif fill_nan == "interpolate":
            arr1 = _interpolate_nan(arr1)
            if arr2 is not None:
                arr2 = _interpolate_nan(arr2)
        else:
            raise ValueError(
                f"fill_nan non riconosciuto: '{fill_nan}'. "
                "Valori validi: 'raise', 'interpolate', 'drop'."
            )

    # ── timestamp → numpy datetime64 ─────────────────────────────────────────
    # usiamo float (secondi dall'epoca) per costruire datetime64[ms]
    epoch = datetimes[0]
    rel_s = np.array(
        [(dt - epoch).total_seconds() for dt in datetimes],
        dtype=np.float64,
    )
    # datetime64[ms] assoluti
    epoch_ns = np.datetime64(epoch.isoformat(), "ms")
    timestamps = epoch_ns + (rel_s * 1000).astype("timedelta64[ms]")

    # ── campionamento ─────────────────────────────────────────────────────────
    if len(rel_s) > 1:
        diffs        = np.diff(rel_s)
        dt_median    = float(np.median(diffs))
        dt_std       = float(np.std(diffs))
        # tolleranza 5 % per decidere se il campionamento è uniforme
        irregular    = bool(dt_std > 0.05 * dt_median)
    else:
        dt_median    = 1.0
        dt_std       = 0.0
        irregular    = False

    N = len(arr1)

    # ── fondi ─────────────────────────────────────────────────────────────────
    k1_val = float(k1) if k1 is not None else float(np.nanmean(arr1))
    k2_val: float | None = None
    if arr2 is not None:
        k2_val = float(k2) if k2 is not None else float(np.nanmean(arr2))

    # ── info dict ─────────────────────────────────────────────────────────────
    info: dict = {
        "N":                N,
        "sampling_seconds": dt_median,
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
    x   = np.arange(len(arr))
    ok  = ~np.isnan(arr)
    if ok.sum() < 2:
        raise ValueError("Troppi valori mancanti per l'interpolazione.")
    return np.interp(x, x[ok], arr[ok])


# ── helper per i grafici: ticks datetime sull'asse x ─────────────────────────

def datetime_axis(
    ax,
    timestamps: np.ndarray,
    max_ticks:  int = 6,
) -> None:
    """
    Imposta l'asse x di ``ax`` con etichette datetime leggibili.

    Parametri
    ---------
    ax         : matplotlib Axes
    timestamps : array datetime64[ms] restituito da load_csv
    max_ticks  : numero massimo di tick (default 6)

    Uso
    ---
    >>> fig, ax = plt.subplots()
    >>> ax.step(np.arange(len(d1)), d1)
    >>> datetime_axis(ax, timestamps)
    """
    import matplotlib.ticker as _ticker

    N       = len(timestamps)
    step    = max(1, N // max_ticks)
    idx     = np.arange(0, N, step)
    labels  = [str(timestamps[i])[:19].replace("T", "\n") for i in idx]

    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=8, rotation=0, ha="center")
    ax.xaxis.set_minor_locator(_ticker.AutoMinorLocator())
