# exp_signal — Forbush Decrease Detector

Ricerca automatica di **Forbush Decrease** (FD) in serie temporali di conteggi di raggi cosmici, con supporto nativo per detector duali **muoni + neutroni epitermali**.

---

## Fisica del problema

Un **Forbush Decrease** è una diminuzione transitoria del flusso di raggi cosmici galattici (GCR) causata dal passaggio di un'espulsione di massa coronale (CME). Il segnale ha una forma caratteristica:

```
mu(t) = k                                                            per t ≤ t₀
mu(t) = k − a · (1 − exp(−(t−t₀)/τ_rise)) · exp(−(t−t₀)/b)   per t > t₀
```

| Parametro | Significato | Valori tipici FD |
|-----------|-------------|-----------------|
| `k` | fondo (conteggi/campione) | dipende dal detector |
| `t₀` | istante di onset | — |
| `a` | ampiezza del calo | 0.5 – 25% di `k` |
| `b` | scala temporale di recupero | 12 h – 10 giorni |
| `τ_rise` | durata dell'onset | 3 – 48 ore |

I due detector rispondono allo **stesso evento solare** con ampiezze diverse (`a₁ ≠ a₂`) ma stessa scala temporale `b`, motivando il test statistico congiunto.

---

## Architettura

```
pipeline.py          ← entry point: tre modalità (simulate / evaluate_dual / search_file)
├── main.py          ← detector core: LLR profile + Nelder-Mead refinement
├── dual_search.py   ← ricerca congiunta su due canali (muoni + neutroni)
├── simulate.py      ← generazione dati sintetici con FD iniettati
├── evaluate.py      ← valutazione su Monte Carlo (efficienza, FP rate)
└── io_utils.py      ← lettura CSV, correzione barometrica, MJD, datetime
```

---

## Algoritmo

### Scansione O(N)

Per ogni coppia `(a, b)` della griglia, il profilo LLR su **tutti** i possibili `t₀` viene calcolato in O(N) con filtri IIR anti-causali (espansione di Taylor al 8° ordine):

```
LLR(t₀) = a · S₁(t₀) − Σₙ₌₁⁸ (a/k)ⁿ/n · Cₙ(t₀)
```

dove `Cₙ(t₀) = Σ_{s≥1} data[t₀+s] · exp(−ns/b)` è calcolato **una sola volta** per ogni `b` e riusato per tutti gli `a`.

Complessità: **O(N · N_b · 8)** — ~4–18 s per N=1 000 000 su 1 core.

### Griglia di default per FD

| Parametro | Valori | Note |
|-----------|--------|------|
| `b` | 12, 18, 24, 36, 48, 60, 72, 96, 120, 168, 240 ore | convertiti in campioni da `dt_seconds` |
| `a` | 16 valori log-uniformi in [0.5%, 25%] di `k` | |

### Raffinamento MLE

Dopo la scansione su griglia, ogni picco viene raffinato con **Nelder-Mead** in 2D `(a, b)` o 3D `(a, b, τ_rise)`. Le incertezze 1σ sono stimate dall'Hessiano numerico.

### Stima robusta del fondo

Due metodi resistenti alla presenza di FD nella serie:
- **`sigma_clip`** (default): media iterativa con rimozione a 3σ
- **`histogram`**: moda dell'istogramma dei conteggi

La media aritmetica è biasata verso il basso in presenza di FD e non è raccomandata.

### Correzione di Bonferroni per serie lunghe

```
fpr_locale = fpr_globale / (N / n_cal)
```

La soglia viene calibrata su serie di lunghezza `n_cal` e corretta per il numero di finestre quasi-indipendenti della serie reale.

---

## Installazione

```bash
pip install numpy scipy matplotlib colorednoise pyyaml
```

---

## Utilizzo rapido

### Ricerca su file CSV reale

```python
from pipeline import run_pipeline
results = run_pipeline("pipeline_config.yaml")
```

Il CSV deve avere almeno una colonna datetime e una (o due) colonne di conteggi:

```
timestamp,neutrons,muons,P_hPa
2024-01-01T00:00:00,1045,10234,1013.2
2024-01-01T01:00:00,1031,10189,1012.8
...
```

### API diretta

```python
import numpy as np
from main import calibrate_threshold, find_signals
from io_utils import load_csv, enrich_signals_with_time

# leggi dati con correzione barometrica per i muoni
ts, neutrons, muons, info = load_csv(
    "data.csv",
    datetime_col      = "timestamp",
    series1_col       = "neutrons",
    series2_col       = "muons",
    pressure_col      = "P_hPa",
    barometric_coeff2 = -0.002,   # β muoni: −0.002 hPa⁻¹
)

dt = info["dt_seconds"]   # campionamento effettivo in secondi

# calibra soglia (Bonferroni automatico)
thr = calibrate_threshold(info["k1"], N=info["N"], dt_seconds=dt)

# cerca FD con modello onset graduale (τ_rise fino a 48 h)
sigs = find_signals(
    neutrons,
    threshold    = thr,
    dt_seconds   = dt,
    tau_rise_max = 48 * 3600 / dt,   # 48 h in campioni
)

# arricchisci con datetime e MJD
enrich_signals_with_time(sigs, ts)

for s in sigs:
    print(f"{s['t0_datetime']}  MJD={s['t0_mjd']:.4f}  "
          f"a/k={s['a']/s['k']*100:.1f}%  b={s['b']*dt/3600:.1f}h  "
          f"τ={s['tau_rise']*dt/3600:.1f}h  LLR={s['llr']:.1f}")
```

### Ricerca duale muoni + neutroni

```python
from dual_search import calibrate_thresholds_dual, find_signals_dual
from io_utils import enrich_signals_with_time

thr1, thr2, thrj = calibrate_thresholds_dual(
    info["k1"], info["k2"], info["N"], dt_seconds=dt)

results = find_signals_dual(
    neutrons, muons, thr1, thr2, thrj,
    k1=info["k1"], k2=info["k2"],
    dt_seconds=dt, tau_rise_max=48*3600/dt,
)
enrich_signals_with_time(results, ts)

# categories: "both", "joint_only", "only1", "only2"
for s in results["both"]:
    print(f"{s['t0_datetime']}  MJD={s['t0_mjd']:.4f}  "
          f"a1={s['a1']:.0f}  a2={s['a2']:.0f}  b={s['b1']*dt/3600:.1f}h")
```

---

## Configurazione YAML

### `pipeline_config.yaml` — ricerca su dati reali

```yaml
mode: search_file

series:
  dt_seconds: 3600.0      # campionamento in secondi (1h)

input:
  file:          data.csv
  datetime_col:  timestamp
  series1_col:   neutrons
  series2_col:   muons     # null per single-channel
  pressure_col:  P_hPa     # null se non disponibile
  barometric_coeff1: 0.0   # β neutroni (hPa⁻¹)
  barometric_coeff2: -0.002 # β muoni (hPa⁻¹)

detection:
  fpr:                0.01
  k_method:           sigma_clip   # sigma_clip | histogram | mean
  tau_rise_max_hours: 48.0         # ore; 0 = onset istantaneo

output:
  report_file: results.txt
  plot_file:   results.pdf
```

### `simulation_config.yaml` — valutazione su dati sintetici

```yaml
series:
  length:     8760        # campioni (1 anno a dt=1h)
  dt_seconds: 3600.0

background:
  k: 1000.0               # neutroni epitermali: 500–3000 c/h

signals:
  a:    { min: 5.0,  max: 200.0, log_scale: true }  # 0.5%–20% di k
  b:    { min: 24.0, max: 240.0, log_scale: true }  # campioni
  tau_rise: { min: 6.0, max: 48.0 }                 # campioni
```

---

## Output

### Report ASCII (`search_file`)

```
  #  t0_datetime           MJD             a      ±      b      ±    τ_rise    llr
  1  2024-01-15 10:00:00   60324.41667   82.3   4.1   56.2   6.3    11.4     47.3
  2  2024-03-08 22:00:00   60377.91667   61.7   5.2   38.9   5.1     8.2     29.1
```

- **`t0_datetime`**: data e ora UTC dell'onset
- **`MJD`**: Modified Julian Date dell'onset (MJD=0 → 1858-11-17)
- **`a`**: ampiezza del calo in conteggi/campione (con incertezza 1σ)
- **`b`**: scala temporale di recupero in campioni (con incertezza 1σ)
- **`τ_rise`**: durata dell'onset in campioni (0 se non stimato)
- **`llr`**: log-likelihood ratio

### Grafici PDF

Prodotti automaticamente dalla pipeline in stile pubblicazione:

- Serie temporale con FD annotati per categoria
- Scatter `a₁` vs `a₂` (test spettrale rigidità)
- Scatter `b₁` vs `b₂` (consistenza scala temporale)
- Efficienza di detection vs ampiezza (con IC 68% Clopper-Pearson)

---

## Dettagli tecnici

### Correzione barometrica

```
N_corr(t) = N_raw(t) · exp(β · (P(t) − P_ref))
```

| Detector | β tipico (hPa⁻¹) |
|----------|-----------------|
| Monitor muoni standard | −0.001 ÷ −0.003 |
| Neutron monitor COSMOS/HYDROINNOVA | −0.0003 ÷ −0.001 |
| Neutroni epitermali (suolo) | piccolo; dipende dall'altitudine |

`P_ref` è di default la media temporale della serie di pressione. La correzione è applicata prima della stima del fondo e della ricerca FD.

### Modified Julian Date

```python
MJD = (datetime - datetime(1858, 11, 17)).total_seconds() / 86400
```

Esempi: 2024-01-01 → MJD 60310.0 · 2025-01-01 → MJD 60675.0

### Campionamento

Il codice non fa assunzioni sull'intervallo di campionamento. La griglia `b` viene calcolata a partire dall'intervallo mediano dei timestamp:

| dt | b griglia (campioni) | b griglia (fisico) |
|----|---------------------|--------------------|
| 1 min | 720 – 14 400 | 12 h – 10 gg |
| 10 min | 72 – 1 440 | 12 h – 10 gg |
| 1 h | 12 – 240 | 12 h – 10 gg |

### Validità dell'espansione di Taylor

L'espansione è valida per `a/k < 0.95`. Con la griglia di default (`a_max = 0.25·k`), l'errore relativo è < 0.001%.

---

## Struttura dei moduli

| Modulo | Contenuto |
|--------|-----------|
| `main.py` | `estimate_background()`, `calibrate_threshold()`, `find_signals()` |
| `dual_search.py` | `calibrate_thresholds_dual()`, `find_signals_dual()`, grafici e report dual |
| `io_utils.py` | `load_csv()` (con correzione pressione), `enrich_signals_with_time()`, `_to_mjd()` |
| `simulate.py` | `generate_data()`, `generate_dual_data()` — dati sintetici con `τ_rise` |
| `evaluate.py` | `run_evaluation()` — efficienza su Monte Carlo |
| `pipeline.py` | `run_pipeline()` — entry point per tutte le modalità |

---

## Modalità della pipeline

| `mode` | Descrizione |
|--------|-------------|
| `search_file` | Legge CSV reale, cerca FD, produce grafici e report con datetime/MJD |
| `evaluate_dual` | Valutazione statistica su serie sintetiche dual-channel |
| `simulate` | Valutazione su serie sintetica single-channel |

```bash
python pipeline.py pipeline_config.yaml
```
