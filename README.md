# Analisi comparativa probabilistica di sistemi propulsivi carbon-neutral

Codice della tesi magistrale *"Comparative analysis of carbon-neutral
aviation propulsion systems utilizing a probabilistic approach"*
(Gianluca Nocerino, Laurea Magistrale in Ingegneria Aeronautica, Sapienza
Università di Roma, A.A. 2025/2026; relatore prof. Riccardo Malpica Galassi).

Il punto di partenza è il modello deterministico di:

> Adler, E.J., Martins, J.R.R.A. (2025). *Energy demand comparison for
  carbon-neutral flight*. Progress in Aerospace Sciences, 152, 101051.
> https://doi.org/10.1016/j.paerosci.2024.101051

Dati range e velocità di crociera, il modello dimensiona un aeromobile
(fase tank-to-wake) per quattro sistemi propulsivi carbon-neutral (batteria
elettrica, fuel cell a idrogeno, combustione di idrogeno liquido, combustione
di e-SAF), ciascuno con fan intubato o elica, per un totale di otto
architetture. Calcola poi l'elettricità rinnovabile necessaria a produrre
l'energia imbarcata (fase well-to-tank), normalizzata per passeggero e miglio
nautico: la *Renewable Electricity Intensity* (REI), in MJ/(pax·nmi).
L'architettura ottima, per ogni missione, è quella con la REI minima.

Il codice originale dell'articolo non è disponibile: il modello è stato
ricostruito a partire dalla sua descrizione e calibrato sui risultati
pubblicati. Su questa base la tesi passa da un approccio deterministico a uno
probabilistico (Uncertainty Quantification): invece di un valore unico per
ogni parametro, si assegnano distribuzioni di probabilità ai parametri
incerti, le si propaga attraverso il modello e si ottiene una distribuzione
della REI. La mappa del sistema ottimo dell'articolo diventa così una mappa
probabilistica, e la ricerca dell'ottimo deterministico lascia il posto a
quella della scelta *robusta*.

Il lavoro, e quindi il codice, è organizzato in tre macro-fasi:

1. **ricostruzione e calibrazione** del modello deterministico;
2. **transizione probabilistica**: identificazione e classificazione delle
   incertezze (parametrica, di calibrazione), propagazione, sensibilità
   globale;
3. **incertezza di forma del modello** e **scenari energetici realistici**
   (2024, 2035, 2050).

## Struttura del codice

```
carbon_neutral_aviation/
├── src/cnav/                          # il pacchetto Python
│   ├── model/                         # il modello deterministico
│   │   ├── units.py                       # conversioni di unità di misura
│   │   ├── constants.py                   # tutti gli input: TechAssumptions e WellToTankEfficiencies
│   │   ├── atmosphere.py                  # atmosfera standard (velocità del suono, densità)
│   │   ├── propulsive_efficiency.py       # fattori di scala dell'efficienza propulsiva (fan ed elica)
│   │   ├── mission.py                     # classe Mission (range, velocità, fan o elica)
│   │   ├── T_to_W_definitions.py          # formule tank-to-wake condivise + stime empiriche di peso
│   │   ├── well_to_tank.py                # catena well-to-tank
│   │   ├── propulsion_systems.py          # classe base e una sottoclasse per sistema propulsivo
│   │   ├── aircraft_sizer.py              # ciclo iterativo di dimensionamento
│   │   ├── energy_intensity.py            # well-to-tank + tank-to-wake -> MJ/(pax·nmi)
│   │   └── model_form.py                  # varianti concettuali del modello (Fase 8)
│   ├── calibration/                   # calibrazione rispetto all'articolo
│   │   ├── cost_functions.py              # funzioni di costo WLS e osservazioni digitalizzate
│   │   ├── deterministic_calibration.py   # Fase 3: theta* = argmin J (Differential Evolution)
│   │   └── calibration_uncertainty.py     # Fase 5: regione accettabile Theta_acc = {theta : J <= J_thr}
│   ├── sensitivity/                   # sensibilità locale
│   │   ├── sensitivity_analysis_tools.py  # elasticità, tornado plot, sweep, superfici 3D
│   │   └── local_sensitivity.py           # Fase 2: aggregazione per architettura e per propulsore
│   ├── uncertainty/                   # quantificazione e propagazione dell'incertezza
│   │   ├── distributions.py               # Fase 4: PDF, correlazioni, LHS, fusione con Theta_acc
│   │   ├── propagation.py                 # Fase 6: modello su ogni campione e su tutta la griglia (R, V)
│   │   ├── output_distributions.py        # Fase 7: PDF della REI in singoli punti operativi
│   │   ├── technology_map.py              # Fase 7: P_j(R, V), regioni robuste, dispersione dei confini
│   │   └── model_form_uncertainty.py      # Fase 8: parametri comuni e specifici di ciascuna forma
│   └── sensitivity_global/            # Fase 9: sensibilità globale
│       ├── factors.py                     # spazio dei fattori indipendenti + blocco di calibrazione
│       ├── outputs.py                     # gli output: REI per architettura su ogni nodo della griglia
│       ├── sobol.py                       # stima degli indici di Sobol (S1, ST)
│       ├── aggregation.py                 # aggregazione pesata con la varianza, mappe del parametro dominante
│       └── plots.py                       # tornado aggregato, mappe, convergenza
├── execution/                         # gli script eseguibili, una cartella per fase (in ordine di esecuzione)
│   ├── 01_riproduzione/                   # riproduzione dei grafici dell'articolo
│   │   ├── plot_intensity_vs_range.py         # REI in funzione del range a velocità fissa
│   │   ├── plot_best_system_map.py            # mappa deterministica del sistema ottimo
│   │   └── plot_scaling_factors.py            # fattori di scala dell'efficienza propulsiva
│   ├── 02_sensibilita_locale/             # sensibilità locale (elasticità, tornado plot)
│   │   ├── sensitivity_analysis.py            # demo degli strumenti
│   │   └── local_sensitivity_report.py        # analisi della tesi
│   ├── 03_calibrazione/                   # calibrazione deterministica
│   │   ├── cost_function_computation.py       # J nel punto nominale
│   │   └── deterministic_calibration_execution.py  # theta* e J_min (Differential Evolution)
│   ├── 04_pdf_parametri/                  # PDF dei parametri incerti
│   │   ├── OEWtoMTOW.py                       # regressione bayesiana di OEW/MTOW (b, r_pivot)
│   │   ├── retrofit_H2.py                     # retrofit a LH2: relazione gamma -> M_OEW,H2
│   │   ├── plot_parameter_distributions.py    # PDF dichiarate vs campione propagato
│   │   └── plot_pdf_singole.py                # un grafico per parametro (-> pdf_singole/)
│   ├── 05_incertezza_calibrazione/        # calibrazione probabilistica
│   │   └── calibration_uncertainty_execution.py  # -> theta_acc.csv
│   ├── 06_propagazione/                   # propagazione Monte Carlo (LHS)
│   │   ├── uncertainty_propagation_execution.py  # -> propagation_result.npz
│   │   └── propagation_checkpoints/           # checkpoint a blocchi, per riprendere un run interrotto
│   ├── 07_mappa_probabilistica/           # grafici dal cubo della propagazione
│   │   ├── plot_probabilistic_technology_map.py  # mappa del sistema ottimo, bande, confini
│   │   └── plot_output_distributions.py       # PDF della REI nei punti operativi
│   ├── 08_model_form/                     # confronto fra le varianti del modello
│   │   ├── model_form_comparison.py
│   │   └── _ckpt_model_form_<forma>/          # checkpoint, uno per forma
│   ├── 09_sensibilita_globale/            # sensibilità globale
│   │   ├── sobol_analysis.py                  # campionamento e indici di Sobol (-> sobol_samples.npz, sobol_indices.csv, ...)
│   │   ├── sobol_figures.py                   # tornado aggregato, mappe del parametro dominante, convergenza
│   │   ├── shapley_calibration.py             # scomposizione del blocco di calibrazione con gli effetti di Shapley
│   │   ├── sobol_cache/                       # cache delle valutazioni del modello
│   │   ├── mappe/                             # mappe del parametro dominante
│   │   └── shapley/                           # risultati della scomposizione di Shapley
│   └── 10_aggiunte_well_to_tank/          # filiere reali e scenari 2024 / 2035 / 2050
│       ├── supply_chain_maps.py               # mappe per energia primaria e CO2, per scenario
│       ├── energy_mix_effects.py              # mix elettrico unico + cherosene (-> kerosene_cube.npz)
│       ├── supply_chain_vs_kerosene.py        # filiere reali contro cherosene
│       └── plot_filiere_singole.py            # PDF dei parametri di filiera (-> pdf_singole_filiere/)
└── tests/
    └── test_basic.py                  # test di controllo del codice
```

Gli output di ogni script (figure, .csv, .npz) vengono scritti nella cartella
della fase che li produce. Le cartelle di `execution/` non vanno quindi
spostate l'una rispetto all'altra: alcuni script leggono gli output di altre
fasi (`06_propagazione/propagation_result.npz`,
`05_incertezza_calibrazione/theta_acc.csv`) con un percorso relativo.

Il pacchetto è organizzato in sottopacchetti concettualmente distinti.
`cnav.model` è il modello deterministico vero e proprio, a sua volta a strati:
`constants`/`units`/`atmosphere` sono i mattoncini di base;
`T_to_W_definitions.py` contiene tutte le formule condivise fra i quattro
sistemi propulsivi; `propulsion_systems.py` definisce solo le differenze fra
un sistema e l'altro; `aircraft_sizer.py` è il ciclo iterativo generico;
`energy_intensity.py` è il livello più alto. `cnav.calibration`,
`cnav.sensitivity`, `cnav.uncertainty` e `cnav.sensitivity_global` lavorano
*sopra* il modello, senza modificarlo: per aggiungere un quinto sistema
propulsivo (es. una configurazione ibrida) basta scrivere una nuova
sottoclasse di `PropulsionSystem` in `cnav/model/propulsion_systems.py`, e
comparirà automaticamente sia nella calibrazione sia nella mappa
probabilistica.

`from cnav import ...` espone direttamente i nomi più usati del modello e
della calibrazione (es. `Mission`, `TechAssumptions`, `compute_intensity`,
`fan_cost`); gli altri strumenti si importano dal rispettivo sottopacchetto
(es. `from cnav.sensitivity.sensitivity_analysis_tools import ...`).

## Installazione

```bash
pip install -r requirements.txt
```

### Eseguire i test

```bash
pytest tests/
```

## Uso rapido

```python
from cnav import (Mission, TechAssumptions, WellToTankEfficiencies,
                  build_energy_carriers, build_default_systems, compute_intensity)

tech = TechAssumptions()
wtt = WellToTankEfficiencies()
systems = build_default_systems(build_energy_carriers(wtt))

mission = Mission(range_nmi=500, cruise_speed_kt=450, propulsor="fan")
for system in systems:
    result = compute_intensity(mission, system, tech)
    print(system.name, result.intensity_MJ_per_pax_nmi)
```

## Il modello deterministico (Fase 1)

### Dimensionamento (tank-to-wake)

Dati range R e velocità di crociera, il ciclo parte da un MTOW di tentativo e
ad ogni iterazione calcola:

- il **peso a vuoto**, OEW = (OEW/MTOW)·MTOW + correzioni. La frazione
  OEW/MTOW segue la correlazione empirica a·MTOW^b + c, con coefficienti
  distinti per fan ed elica; per l'elica si usa un *smooth maximum* delle due
  correlazioni, per estenderla ai velivoli più pesanti; per la batteria la
  frazione è costante e pari a 0.37. Le correzioni dipendono dall'architettura:
  - *combustione di LH2*: peso del serbatoio criogenico, legato
    all'efficienza gravimetrica γ;
  - *fuel cell*: motori elettrici al posto di quelli convenzionali, fuel
    cell, sistema di gestione termica (TMS) e serbatoio;
  - *batteria*: motori elettrici e TMS;
  - *e-SAF*: nessuna, è una soluzione *drop-in*.

  Ai velivoli a idrogeno si applica inoltre il moltiplicatore M_OEW,H2 sulla
  frazione di peso a vuoto. Il TMS dipende dal calore da smaltire e, oltre al
  peso, aggiunge una resistenza aerodinamica che riduce L/D; per i sistemi a
  idrogeno L/D è ulteriormente moltiplicato per 0.95;
- il **peso dell'energia** imbarcata, somma di tre contributi:
  - *salita*: energia meccanica per raggiungere la quota di crociera;
  - *crociera*: equazione di Breguet, nella forma a massa costante per la
    batteria;
  - *riserva*: distanza dall'aeroporto alternato, pari a min(R, 200 nmi),
    più 45 minuti di loiter alla velocità di crociera;
- il **payload**, con una correlazione empirica sul range;

fino a convergenza di MTOW, nel rispetto del vincolo MTOW ≤ MTOW_max. La
quota di crociera usata per il Mach è quella effettivamente raggiunta, che
per missioni brevi può essere inferiore a quella nominale (35000 ft per il
fan, 25000 ft per l'elica).

**Efficienza complessiva.** Dipende dall'architettura: motore elettrico ×
efficienza propulsiva per la batteria; fuel cell × motore × efficienza
propulsiva per la fuel cell; per la combustione, efficienza del turbofan o
del turboelica (funzioni della potenza) per il relativo fattore di scala.
L'efficienza propulsiva è il prodotto di un valore di riferimento e di un
fattore di scala dipendente dal Mach di volo:

- **fan**: efficienza propulsiva di un fan ideale senza perdite (Michel,
  2011), funzione del Mach e del rapporto di compressione del fan,
  normalizzata per il suo valore al Mach di riferimento M_ref;
- **elica**: funzione a tratti, con crescita esponenziale (fattore k_r) fino
  al Mach di picco M_peak e decadimento gaussiano (larghezza σ_d) oltre.

### Catena energetica (well-to-tank)

L'energia imbarcata viene riportata alla fonte con il rendimento complessivo
di catena di ciascun vettore energetico (elettricità 0.85, LH2 0.46,
e-SAF 0.26), il reciproco del fattore di energia primaria. La REI è
l'elettricità rinnovabile così ottenuta divisa per numero di passeggeri e
range; il numero di passeggeri è il payload diviso per il peso del
passeggero.

### Tutti gli input in un unico posto

`TechAssumptions` (in `constants.py`) raccoglie ogni numero usato dal modello:
nessuna funzione ha valori scritti al suo interno, tutte li ricevono come
argomento. `WellToTankEfficiencies` contiene i rendimenti di catena. I
poteri calorifici (idrogeno, e-SAF) restano fuori, come costanti fisse in
`constants.py`: sono proprietà chimiche.

Entrambe sono `dataclass` immutabili con un metodo `.with_changes(...)` che
genera copie con parametri diversi, comodo per il campionamento:

```python
tech_perturbato = tech.with_changes(e_battery_Wh_per_kg=450.0)
```

### Riproduzione dei grafici dell'articolo

```bash
python execution/01_riproduzione/plot_intensity_vs_range.py
python execution/01_riproduzione/plot_best_system_map.py
python execution/01_riproduzione/plot_scaling_factors.py
```

## Calibrazione (Fasi 3 e 5)

La ricostruzione esatta del modello non è possibile per sei parametri non
specificati nell'articolo, che vengono determinati risolvendo un problema
inverso. Per ciascuno si definisce un intervallo plausibile:

| Parametro | Simbolo | Intervallo | Motivazione |
|---|---|---|---|
| Peso del passeggero | W_pax | 75 – 105 kg | EASA: 76.5 kg senza bagagli, 99.1 kg con bagagli |
| Rapporto di compressione del fan | π_fan | 1.30 – 1.55 | turbofan commerciali attuali |
| Mach di riferimento (fan) | M_ref | 0.78 – 0.85 | Mach di crociera tipico dei turbofan |
| Mach di picco dell'elica | M_peak | 0.45 – 0.60 | velocità di crociera dei turboelica moderni |
| Fattore di crescita dell'elica | k_r | 5 – 20 | curva dei fattori di scala dell'articolo |
| Larghezza del decadimento dell'elica | σ_d | 0.02 – 0.08 | curva dei fattori di scala dell'articolo |

**Funzione di costo** (`cost_functions.py`). È ai minimi quadrati pesati
sugli scarti relativi fra il modello e le osservazioni, estratte con
WebPlotDigitizer dai grafici dell'articolo su 20 range, a 450 kn per il fan e
250 kn per l'elica. Gli output confrontati sono la REI di ciascun sistema e,
per le architetture a batteria, il range massimo fattibile. Pesi:

- per sistema: 1, tranne la batteria (3), che ha meno osservazioni;
- per range: maggiori ai range più corti e più lunghi, dove la discrepanza è
  più marcata;
- per il range massimo fattibile: 3.

Se il modello non converge dove esiste un'osservazione, il termine è
sostituito da una penalità costante pari a 10. Le funzioni di fan ed elica
(quella del fan esclude la fuel cell, assente nel grafico di riferimento)
sono normalizzate con il reciproco del loro valore in un punto fissato e
sommate nella funzione globale J(θ).

- **Fase 3, calibrazione deterministica.** J viene minimizzata con
  Differential Evolution (popolazione iniziale LHS, strategia `best1bin`,
  popolazione 15, massimo 60 generazioni), ripetuta in 5 run indipendenti.
  Lo scopo non è trovare il "vero" θ, perché una stima puntuale contraddice
  la premessa stessa della UQ, ma il minimo assoluto J_min = 1.9958, che
  fissa la soglia della fase successiva. I valori di default dei sei
  parametri in `TechAssumptions` corrispondono al punto θ* trovato.
- **Fase 5, calibrazione probabilistica.** 10000 campioni LHS nello spazio
  dei parametri di calibrazione; si accettano quelli con
  J ≤ J_thr = 1.15·J_min = 2.2951. Le combinazioni accettate formano le
  righe della matrice Θ_acc (`theta_acc.csv`).

```bash
python execution/03_calibrazione/cost_function_computation.py
python execution/03_calibrazione/deterministic_calibration_execution.py
python execution/05_incertezza_calibrazione/calibration_uncertainty_execution.py   # -> theta_acc.csv
```

## Caratterizzazione dell'incertezza (Fasi 2 e 4)

### Sensibilità locale (Fase 2)

Prima della UQ, uno screening individua i parametri più influenti.
L'indicatore è l'elasticità S = ∂ln y / ∂ln θ, con la derivata approssimata
alle differenze finite, calcolata su più combinazioni di range e velocità per
ogni architettura. Un parametro è rilevante se il coefficiente supera 0.05; i
risultati sono aggregati (coefficiente medio e frequenza di superamento della
soglia) per architettura (8 categorie) e per propulsore (fan / elica) e
visualizzati in tornado plot.

```bash
python execution/02_sensibilita_locale/local_sensitivity_report.py
```

### Classificazione delle incertezze

Unendo i risultati della sensibilità locale a considerazioni tecnologiche e
fisiche, i parametri incerti sono classificati in tre tipi:

| Parametri | Tipo di incertezza | Correlazioni |
|---|---|---|
| TtW: energia specifica della batteria, efficienza e potenza specifica della fuel cell, efficienza del motore elettrico, efficienze propulsive di fan ed elica, efficienza gravimetrica del serbatoio, moltiplicatore M_OEW,H2 | parametrica (tecnologica) | M_OEW,H2 linearmente legato a γ |
| WtT: rendimenti di catena di elettricità, LH2, e-SAF | parametrica (tecnologica) | LH2 ed e-SAF correlati dall'elettrolisi comune |
| Peso del passeggero, π_fan, M_ref, M_peak, k_r, σ_d | di calibrazione | |
| Efficienza aerodinamica, tempo di loiter | di forma del modello | |
| Coefficienti b e c di OEW/MTOW (fan ed elica) | parametrica e di forma del modello | b e c correlati fra loro |

### PDF dei parametri tecnologici (Fase 4)

Definite in `distributions.build_default_specs`:

| Parametro | Distribuzione | Fonte |
|---|---|---|
| Energia specifica batteria (livello velivolo) | Triangolare 135 – 225 – 300 Wh/kg | 45% del valore di cella (300 e 500 Wh/kg) |
| Potenza specifica fuel cell | Triangolare 1.0 – 1.5 – 2.0 kW/kg | PEMFC in retrofit (ICCT), ZeroAvia |
| Efficienza fuel cell | Triangolare 0.45 – 0.55 – 0.65 | PEMFC in retrofit (ICCT) |
| Efficienza gravimetrica serbatoio γ | Triangolare 0.47 – 0.50 – 0.58 | FlyZero |
| Moltiplicatore M_OEW,H2 | derivato da γ | retrofit semplificato |
| Rendimento catena elettricità | Uniforme 0.82 – 0.88 | 0.85 ± 0.03 |
| Rendimento catena LH2 | Triangolare 0.42 – 0.46 – 0.47 | elettrolisi PEM |
| Rendimento catena e-SAF | Triangolare 0.21 – 0.26 – 0.30 | efficienza di sintesi |
| Efficienza motore elettrico | 0.85 + 0.14·Beta(3.6, 3) | motori PMSM |
| Efficienza propulsiva fan | 0.70 + 0.10·Beta(4, 4) | articolo di riferimento |
| Efficienza propulsiva elica | 0.80 + 0.10·Beta(4, 4) | articolo di riferimento |
| Esponente b (fan / elica) | lognormale riflessa | regressione bayesiana |
| r_pivot (fan / elica) | normale troncata a ±4σ | regressione bayesiana |
| Coefficiente c (fan / elica) | derivato da b e r_pivot | |

I valori dell'articolo per batteria (300 Wh/kg) e fuel cell (2 kW/kg) sono
superiori a quelli a livello di velivolo descritti dalle PDF, che tengono
conto dell'integrazione a bordo: per questo la mappa probabilistica non è
centrata su quella deterministica. `check_nominal_consistency()` riporta dove
cade ciascun valore nominale all'interno della propria PDF.

**Correlazioni.** Alcune dipendenze sono imposte per non generare
combinazioni fisicamente impossibili:

1. M_OEW,H2 non è campionato: è ricavato da γ con la relazione lineare
   M_OEW,H2 = −0.08535·γ + 1.139503, ottenuta con un esercizio di retrofit
   semplificato (`retrofit_H2.py`) in cui la fusoliera di un velivolo
   convenzionale viene allungata per ospitare un serbatoio cilindrico.
2. I rendimenti di catena sono correlati in modo non lineare dai passaggi
   comuni: le perdite di distribuzione di rete per tutti e tre, l'elettrolisi
   per LH2 ed e-SAF. La dipendenza fra LH2 ed e-SAF è imposta con una copula
   gaussiana, il cui ρ è derivato da `shared_factor_rho()` come quota di
   varianza spiegata dal fattore comune.
3. Per la frazione OEW/MTOW si mantiene a costante e si stimano b e c con una
   regressione bayesiana su dati storici di velivoli (database FAST,
   `OEWtoMTOW.py`). Per rimuovere la forte correlazione non lineare fra b e
   c si introduce il parametro ausiliario r_pivot, il valore della frazione
   in corrispondenza di MTOW_pivot: b e r_pivot si campionano come variabili
   indipendenti e c = r_pivot − a·MTOW_pivot^b si ricava analiticamente.

**Due origini di incertezza, tenute separate.** La matrice θ che viene
propagata ha due blocchi di colonne:

- *letteratura* (`distributions.build_default_specs`): parametri con PDF
  dichiarata, campionati per trasformata inversa da un disegno LHS;
- *calibrazione* (`theta_acc.csv`): non si ricostruiscono marginali, ogni
  combinazione accettata è trattata come un'unica entità e si pescano
  **righe intere** di Θ_acc, per conservare la dipendenza fra i parametri
  emersa dalla calibrazione.

`assemble_theta` verifica che i due blocchi siano disgiunti e solleva un
errore se un parametro compare in entrambi.

Gli script di grafica (`plot_parameter_distributions.py`,
`plot_pdf_singole.py`) sovrappongono le PDF dichiarate al campione
effettivamente propagato, e quindi richiedono `propagation_result.npz`.

```bash
python execution/04_pdf_parametri/OEWtoMTOW.py
python execution/04_pdf_parametri/retrofit_H2.py
python execution/04_pdf_parametri/plot_parameter_distributions.py   # dopo la Fase 6
python execution/04_pdf_parametri/plot_pdf_singole.py               # dopo la Fase 6
```

## Propagazione e mappa probabilistica (Fasi 6 e 7)

```bash
python execution/06_propagazione/uncertainty_propagation_execution.py              # -> propagation_result.npz
python execution/07_mappa_probabilistica/plot_probabilistic_technology_map.py      # mappe, dal .npz
python execution/07_mappa_probabilistica/plot_output_distributions.py              # PDF della REI nei punti operativi
```

La propagazione è un Monte Carlo con Latin Hypercube Sampling (N = 1500) su
una griglia di 45 range (10 – 10000 nmi) × 40 velocità di crociera
(180 – 500 kn): 1800 punti di progetto, quindi 2.7 milioni di
dimensionamenti completi. Il run scrive checkpoint a blocchi
(`propagation_checkpoints/`) e si riprende da dove si era interrotto. Il
risultato (`propagation_result.npz`) contiene il cubo completo delle
intensity: cambiare la soglia di robustezza, aggregare fan ed elica o
guardare un confine diverso non richiede di rilanciare il modello.

Dal cubo la Fase 7 ricava:

- le **PDF della REI** in singoli punti operativi, per le quattro
  architetture con lo stesso propulsore. La densità è espressa per decade
  (asse logaritmico) e scalata per la probabilità di fattibilità: l'area
  sotto ogni curva è la probabilità che il sistema chiuda il ciclo di
  dimensionamento;
- le **bande di probabilità** (5° – 95° percentile) della REI in funzione
  del range a velocità fissa, controparte probabilistica dell'analogo
  grafico dell'articolo;
- la **mappa del sistema più probabilmente ottimo** su tutto il piano
  range-velocità. Una scelta è *robusta* se l'architettura è ottima con
  probabilità almeno del 90%: il contorno P = 0.90 delimita queste regioni,
  e fuori da esse la scelta resta statisticamente ambigua.

## Sensibilità globale (Fase 9)

```bash
python execution/09_sensibilita_globale/sobol_analysis.py        # campioni, valutazioni e indici
python execution/09_sensibilita_globale/sobol_figures.py         # tornado, mappe, convergenza
python execution/09_sensibilita_globale/shapley_calibration.py   # scomposizione del blocco di calibrazione
```

La propagazione dice quanto è incerta la REI, ma non quali parametri causano
questa incertezza. Per questo si calcolano gli indici di Sobol totali S_T,
cioè la frazione di varianza in cui un parametro è coinvolto, da solo o in
interazione con altri.

- **Gruppi.** I parametri tecnologici sono valutati singolarmente; i sei
  parametri di calibrazione come un unico blocco, per conservarne le
  correlazioni; i coefficienti della frazione di peso a vuoto nel gruppo
  "Others" (g = 12 gruppi in totale).
- **Stima.** Routine Python dedicata, perché le librerie esterne non
  gestiscono in modo flessibile i blocchi di parametri: sequenza di Sobol a
  bassa discrepanza e stimatore di Jansen, con costo N·(g + 2) valutazioni
  (N = 1024). Lo stimatore è verificato sulla funzione di Ishigami.
- **Accorgimenti.** I punti non fattibili sono esclusi a coppie, e le
  missioni con più del 10% di campioni non fattibili sono escluse del tutto.
  La decomposizione si applica a log(REI), perché la REI diverge vicino ai
  limiti di fattibilità.
- **Aggregazione.** Gli indici dei diversi output (architettura, punto di
  missione) sono combinati con la media pesata con la varianza Var(log REI)
  (indice di Sobol generalizzato, Lamboni et al. 2011; Gamboa et al. 2014).
  Ne derivano le **mappe del parametro dominante** a tre livelli (tutte le
  architetture, per propulsore, per singola architettura) e il **tornado
  aggregato** su tutto l'inviluppo, con barre di errore fra il valore minimo
  e massimo dell'indice.
- **Effetti di Shapley.** Per capire quale dei parametri di calibrazione
  guida la varianza, l'indice del blocco viene ripartito fra i sei parametri
  in proporzione ai loro effetti di Shapley,
  S_T,j ≈ S_T,blocco · Sh_j / ΣSh. Gli effetti sono stimati direttamente
  dalle matrici di Sobol già valutate, con un approccio nearest-neighbour,
  senza nuove simulazioni.

## Incertezza di forma del modello (Fase 8)

L'incertezza di forma nasce dalle scelte di modellazione, e non è
rappresentabile con PDF sui parametri. Si valuta sostituendo singolarmente
tre ipotesi del modello base con alternative, implementate come rami in
`cnav/model/model_form.py`. La forma di default (`ModelForm()`) riproduce
esattamente il modello base.

| Ramo | Modello base | Alternativa |
|---|---|---|
| `aero_model` | `"fixed"`: L/D costante (20 per il fan, 15 per l'elica) | `"polar_decay"`: L/D degrada quando la quota di crociera effettiva è inferiore a quella nominale; con polare parabolica e atmosfera ISA, (L/D) = (L/D)_base · 2ξ/(1 + ξ²), con ξ = ρ(h_nom)/ρ(h) |
| `oew_model` | `"paper"`: a·MTOW^b + c | `"raymer"`: OEW/MTOW = a·MTOW^c (MTOW in libbre), coefficienti di Raymer per jet da trasporto (fan) e bimotori turboelica (elica) |
| `reserve_model` | `"paper"`: alternato + 45 min di loiter alla velocità di crociera | `"easa"`: conforme alle linee guida EASA, alternato invariato, loiter al 75% della velocità di crociera, contingenza pari al 5% del range |

I preset in `MODEL_FORM_PRESETS` sono `base`, `polare`, `raymer`, `easa` e
`tutte`.

```bash
python execution/08_model_form/model_form_comparison.py
```

Lo script ripete la propagazione per ciascuna forma e, per ognuna, produce la
mappa del sistema più probabilmente ottimo e la mappa delle regioni in cui
l'architettura ottima differisce da quella del modello base, più un confronto
complessivo delle mappe di tutte le forme.

## Scenari energetici realistici (Fase 10)

L'ipotesi di fondo dell'articolo, mantenuta nelle fasi precedenti, è che
tutti i vettori energetici derivino da elettricità esclusivamente
rinnovabile: solo così le architetture sono davvero a emissioni nulle. La
Fase 10 introduce filiere realistiche e la loro evoluzione nel tempo.

**Metriche.** Con filiere non rinnovabili la REI non basta, e si valutano due
metriche distinte: l'**intensità di energia primaria** [MJ/(pax·nmi)], che
generalizza la REI includendo fonti rinnovabili e non, e l'**intensità di
carbonio** [g CO2/(pax·nmi)]. Per ciascun vettore (rete elettrica, LH2, SAF)
si introducono due parametri con PDF triangolare: il rendimento di catena
realistico e l'intensità di carbonio, ricavati con GREET modellando le
filiere alimentate dal relativo mix elettrico.

**Orizzonti temporali.**

| Anno | Mix elettrico | SAF | LH2 | PDF |
|---|---|---|---|---|
| 2024 | IEA, dati storici | 100% HEFA (da colza) | 100% da gas naturale | moda = valore calcolato, estremi al 95% e 105% |
| 2035 | IEA WEO 2025: CPS, STEPS, Net Zero | 75% HEFA, 25% Fischer-Tropsch | 50% gas naturale, 50% elettrolisi | estremi dagli scenari CPS e Net Zero, moda da STEPS |
| 2050 | IEA WEO 2025: CPS, STEPS, Net Zero | 25% HEFA, 75% Fischer-Tropsch | 100% elettrolisi | estremi dagli scenari CPS e Net Zero, moda da STEPS |

Il SAF Fischer-Tropsch usa idrogeno da elettrolisi PEM e carbonio da Direct
Air Capture; le quote delle filiere di SAF derivano da IATA. Per ogni anno le
quote delle filiere sono fisse, quindi l'incertezza sui parametri del 2035 e
del 2050 deriva solo dalla variabilità del mix elettrico fra i tre scenari.

**Cherosene.** È incluso come termine di paragone: rendimento di catena e
intensità di carbonio a monte da GREET, emissioni di combustione
(tank-to-wake) da ICAO Annex 16, Vol. IV.

I parametri di filiera entrano solo a valle del dimensionamento, quindi gli
script ripartono dal cubo della Fase 6 senza ripropagare il modello:

```bash
python execution/10_aggiunte_well_to_tank/supply_chain_maps.py               # mappe per energia primaria e CO2 (tutti gli scenari; oppure: attuale 2035 2050)
python execution/10_aggiunte_well_to_tank/energy_mix_effects.py              # mix unico + cherosene -> kerosene_cube.npz
python execution/10_aggiunte_well_to_tank/supply_chain_vs_kerosene.py        # filiere reali contro cherosene
python execution/10_aggiunte_well_to_tank/plot_filiere_singole.py            # PDF dei parametri di filiera
```

`supply_chain_vs_kerosene.py` richiede `kerosene_cube.npz`, prodotto da
`energy_mix_effects.py` con lo stesso numero di campioni.

## Ordine di esecuzione

Le fasi dipendono l'una dall'altra attraverso i file intermedi:

```
03_calibrazione  ->  05_incertezza_calibrazione (theta_acc.csv)
                                 |
                                 v
                      06_propagazione (propagation_result.npz)
                                 |
       +-------------------------+-------------------------+
       v                         v                         v
07_mappa_probabilistica   04_pdf_parametri (grafici)   10_aggiunte_well_to_tank

08_model_form e 09_sensibilita_globale ripartono da theta_acc.csv
```

## Principali risultati

- I confini deterministici fra le tecnologie si dissolvono in zone di
  transizione: una scelta robusta (P ≥ 0.90) non esiste su tutto
  l'inviluppo, in particolare a range brevi con velocità medie e a range
  medio-lunghi con velocità basse.
- Nell'ipotesi di filiera rinnovabile, la combustione di LH2 con fan è la
  soluzione robusta da range brevi ad alta velocità fino ai range lunghi; la
  batteria con elica a range molto brevi e velocità basse; la fuel cell con
  elica a range medi e velocità molto basse. I sistemi a combustione hanno
  distribuzioni della REI molto più strette di quelli elettrochimici.
- La varianza della REI è guidata soprattutto dal blocco di calibrazione (in
  particolare dal Mach di picco dell'elica, nella fascia di velocità
  intermedie) e dai coefficienti della frazione di peso a vuoto (a range e
  velocità elevati, per l'effetto valanga sul peso). Fra i parametri
  tecnologici dominano potenza specifica ed efficienza della fuel cell,
  mentre l'energia specifica della batteria domina localmente alle missioni
  brevi e lente.
- L'incertezza di forma del modello è critica quanto quella parametrica e di
  calibrazione: ipotesi apparentemente locali possono ridefinire la scelta
  ottima.
- Con filiere realistiche, minimizzare l'energia primaria e minimizzare le
  emissioni di CO2 porta a scelte diverse nel 2024 e nel 2035; nel 2050 le
  due metriche convergono e l'efficienza energetica diventa un indicatore
  affidabile della prestazione climatica.

## Riferimenti

Adler, E.J., Martins, J.R.R.A. (2025). Energy demand comparison for
carbon-neutral flight. *Progress in Aerospace Sciences*, 152, 101051.

Michel, U. (2011). The benefits of variable area fan nozzles on turbofan
engines. 49th AIAA Aerospace Sciences Meeting, AIAA 2011-226.

Alves, P., Silvestre, M., Gamboa, P. (2020). Aircraft Propellers — Is There a
Future? *Energies*, 13(16), 4157.

Raymer, D.P. (1992). *Aircraft Design: A Conceptual Approach*, 2nd ed. AIAA.

Lamboni, M., Monod, H., Makowski, D. (2011). Multivariate sensitivity
analysis to measure global contribution of input factors in dynamic models.
*Reliability Engineering & System Safety*, 96(4), 450–459.

Gamboa, F., Janon, A., Klein, T., Lagnoux, A. (2014). Sensitivity analysis
for multidimensional and functional outputs. *Electronic Journal of
Statistics*, 8(1), 575–603.

IDEAS Lab, University of Michigan (2025). *Future Aircraft Sizing Tool
(FAST)*. https://github.com/ideas-um/FAST

International Energy Agency (2025). *World Energy Outlook 2025*.

Argonne National Laboratory (2023). *GREET — Greenhouse gases, Regulated
Emissions, and Energy use in Technologies*.

ICAO (2023). *Annex 16, Volume IV — CORSIA*, 2nd ed.