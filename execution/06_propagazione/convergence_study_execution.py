"""
Studio di convergenza della propagazione Monte Carlo (LHS).

Grandezza monitorata: la probabilità di dominanza P(T = j), cioè la
probabilità che la tecnologia j sia quella ottimale (intensità minima)
in una cella della griglia. È la grandezza da cui si costruisce la
mappa probabilistica.

Per ogni N in N_LIST si lanciano R_REPLICHE propagazioni indipendenti
(seed diversi). Il modello viene valutato SOLO su poche celle di
controllo, scelte automaticamente dalla mappa di riferimento
(propagation_result.npz). L'errore della stima di P a un dato N è la
deviazione standard dei suoi valori fra le repliche; in ogni cella si
prende il massimo sulle 8 etichette, poi il massimo fra le celle del
gruppo.

Due gruppi di celle, con due errori e due tolleranze distinti:

  - "confine": dove sqrt(P(1-P)) è massima, cioè P vicina a 0.5 (due
    tecnologie si contendono la cella). Lì la stima di P ha la varianza
    più alta: è l'errore MASSIMO della mappa. Tolleranza TOL_BOUNDARY,
    pensata per la lettura generale del campo di P (curve di livello
    distanti almeno 0.15)

  - "soglia": dove max_j P_j è più vicina a P_THRESHOLD = 0.90, cioè
    sulla linea tratteggiata della mappa probabilistica. Lì si decide se
    una cella è "robusta" o "decision-uncertain". Tolleranza
    TOL_THRESHOLD, scelta con la regola 2*sigma <= metà della distanza
    fra soglie adiacenti di robust_area_fraction (0.90 e 0.95):
    2*sigma <= 0.025, cioè sigma <= 0.0125

Risultato: due grafici, uno per gruppo, con una curva per cella + il
massimo sulle celle, in log-log contro N, con la retta N^(-1/2) di
riferimento e la tolleranza del gruppo.

Presuppone:
  - theta_acc.csv  (execution/05_incertezza_calibrazione/)
  - propagation_result.npz  (questa cartella), usato SOLO per scegliere
    le celle di controllo

Uso (prima un giro di prova: N_LIST = [50, 100, 200], N_TESI = 100,
R_REPLICHE = 3, poi cancella convergence_checkpoints/):
    python execution/06_propagazione/convergence_study_execution.py

Ogni replica viene salvata in convergence_checkpoints/: se il run si
interrompe, rilanciando riparte da dove era arrivato. Cambiando celle,
seed o theta_acc lo script se ne accorge e si ferma invece di mescolare
risultati: in quel caso cancella la cartella.
"""
import contextlib
import hashlib
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cnav.model.aircraft_sizer import AircraftSizer
from cnav.model.energy_intensity import compute_intensity
from cnav.model.mission import Mission
from cnav.model.propulsion_systems import build_default_systems
from cnav.model.well_to_tank import build_energy_carriers
from cnav.uncertainty import (
    assemble_theta,
    build_default_correlations,
    build_default_specs,
    load_theta_acc,
)
from cnav.uncertainty.propagation import (
    PROPULSORS,
    TECHNOLOGIES,
    PropagationResult,
    technology_labels,
    theta_row_to_tech_wtt,
)

# ---------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent
THETA_ACC_PATH = OUT_DIR.parent / "05_incertezza_calibrazione" / "theta_acc.csv"
REFERENCE_PATH = OUT_DIR / "propagation_result.npz"
CHECKPOINT_DIR = OUT_DIR / "convergence_checkpoints"
FIG_BOUNDARY_PATH = OUT_DIR / "convergence_boundary.png"
FIG_THRESHOLD_PATH = OUT_DIR / "convergence_threshold.png"
CSV_PATH = OUT_DIR / "convergence_study.csv"

# metà, doppio, quadruplo... attorno al valore usato in tesi
N_LIST = [375, 750, 1500, 3000, 6000]
N_TESI = 1500                  # il valore di cui si vuole dimostrare la convergenza
R_REPLICHE = 10                # repliche indipendenti per ogni N
SEED_BASE = 1000               # lontano dal seed 0 del run di riferimento
N_WORKERS = 1                  # >1 parallelizza sui campioni

# celle di confine (P ~ 0.5)
N_BOUNDARY_CELLS = 8
TOL_BOUNDARY = 0.02

# celle di soglia (max_j P_j ~ P_THRESHOLD)
P_THRESHOLD = 0.90             # la stessa THRESHOLD della mappa probabilistica
N_THRESHOLD_CELLS = 8
THRESHOLD_WINDOW = 0.05        # si accettano solo celle con |max_j P_j - 0.90| <= questo
TOL_THRESHOLD = 0.0125         # 2*sigma <= (0.95 - 0.90) / 2

MIN_SEP = 3                    # distanza minima fra celle scelte (in celle di griglia)

MS_PER_EVAL = 0.7              # solo per la stima del tempo

# dimensioni dei caratteri dei grafici
FS_TITLE = 16
FS_LABEL = 14
FS_TICKS = 12
FS_LEGEND = 12

GROUPS = ("confine", "soglia")


# ---------------------------------------------------------------------
# Scelta delle celle di controllo
# ---------------------------------------------------------------------
def _pick_separated(score, n, excluded, min_sep, mask=None):
    """Le n celle col punteggio più alto, distanti almeno min_sep (in
    indici di griglia) fra loro e da quelle già scelte"""
    chosen = []
    for flat in np.argsort(score, axis=None)[::-1]:
        if len(chosen) == n:
            break
        i, j = np.unravel_index(flat, score.shape)
        if not np.isfinite(score[i, j]):
            continue
        if mask is not None and not mask[i, j]:
            continue
        if any(max(abs(i - a), abs(j - b)) < min_sep for a, b in chosen + excluded):
            continue
        chosen.append((int(i), int(j)))
    return chosen


def select_cells(ref: PropagationResult) -> list:
    """Celle di confine (P ~ 0.5) e celle di soglia (max_j P_j ~ 0.90)"""
    n_labels = len(ref.labels)
    P = np.stack([(ref.best_index == l).mean(axis=0) for l in range(n_labels)])
    pmax = P.max(axis=0)

    # confine: errore teorico sqrt(P(1-P)) massimo sulle etichette
    sd = np.sqrt(P * (1.0 - P)).max(axis=0)
    boundary = _pick_separated(sd, N_BOUNDARY_CELLS, [], MIN_SEP, mask=sd > 0)

    # soglia: max_j P_j il più vicino possibile a P_THRESHOLD
    dist = np.abs(pmax - P_THRESHOLD)
    threshold = _pick_separated(-dist, N_THRESHOLD_CELLS, boundary, MIN_SEP,
                                mask=dist <= THRESHOLD_WINDOW)
    if len(threshold) < N_THRESHOLD_CELLS:
        print(f"  attenzione: solo {len(threshold)} celle di soglia entro "
              f"±{THRESHOLD_WINDOW} da P = {P_THRESHOLD}")

    cells = []
    for kind, picks in (("confine", boundary), ("soglia", threshold)):
        for i, j in picks:
            # le due etichette più probabili: sono quelle che si contendono la cella
            top = np.argsort(P[:, i, j])[::-1][:2]
            cells.append(dict(i=i, j=j, kind=kind,
                              speed=float(ref.grid.speeds_kt[i]),
                              range=float(ref.grid.ranges_nmi[j]),
                              prima=ref.labels[top[0]], p_prima=float(P[top[0], i, j]),
                              seconda=ref.labels[top[1]], p_seconda=float(P[top[1], i, j])))
    return cells


# ---------------------------------------------------------------------
# Valutazione del modello su punti sparsi
# ---------------------------------------------------------------------
def evaluate_points(row, speeds, ranges) -> np.ndarray:
    """Come propagation.evaluate_sample, ma su una lista di punti (V, R)
    invece che sul prodotto cartesiano della griglia.
    Ritorna (8, n_punti) float32, NaN = non fattibile"""
    tech, wtt = theta_row_to_tech_wtt(row)
    sizer = AircraftSizer(tech)
    systems = build_default_systems(build_energy_carriers(wtt))

    out = np.full((len(TECHNOLOGIES) * len(PROPULSORS), len(speeds)),
                  np.nan, dtype=np.float32)
    for k, (v, r) in enumerate(zip(speeds, ranges)):
        for p_idx, propulsor in enumerate(PROPULSORS):
            mission = Mission(range_nmi=float(r), cruise_speed_kt=float(v),
                              propulsor=propulsor)
            for s_idx, system in enumerate(systems):
                try:
                    res = compute_intensity(mission, system, tech, sizer)
                    out[s_idx * len(PROPULSORS) + p_idx, k] = res.intensity_MJ_per_pax_nmi
                except Exception:
                    pass  # resta NaN: non fattibile
    return out


_SPEEDS = None
_RANGES = None


def _worker_init(speeds, ranges):
    global _SPEEDS, _RANGES
    _SPEEDS, _RANGES = speeds, ranges


def _worker_eval(row):
    return evaluate_points(row, _SPEEDS, _RANGES)


# ---------------------------------------------------------------------
# Una replica (con checkpoint)
# ---------------------------------------------------------------------
def run_replica(n, seed, theta_acc, specs, corr, speeds, ranges, pool):
    theta = assemble_theta(n, theta_acc, specs=specs, correlations=corr, seed=seed)
    sha = hashlib.sha1(np.ascontiguousarray(theta.to_numpy(dtype=float)).tobytes()).hexdigest()[:16]

    path = CHECKPOINT_DIR / f"N{n:05d}_seed{seed}.npz"
    if path.exists():
        d = np.load(path)
        same = (str(d["theta_sha1"]) == sha
                and d["speeds"].shape == speeds.shape
                and np.allclose(d["speeds"], speeds)
                and np.allclose(d["ranges"], ranges))
        if same:
            return d["intensities"], True
        raise ValueError(
            f"{path.name} è stato calcolato con celle o theta diversi da quelli "
            f"attuali. Cancella {CHECKPOINT_DIR} per ricalcolare.")

    rows = [theta.iloc[k].to_dict() for k in range(n)]
    if pool is None:
        res = [evaluate_points(row, speeds, ranges) for row in rows]
    else:
        res = list(pool.map(_worker_eval, rows,
                            chunksize=max(1, n // (4 * N_WORKERS))))
    x = np.stack(res)                                   # (N, 8, n_celle)
    np.savez_compressed(path, intensities=x, theta_sha1=sha,
                        speeds=speeds, ranges=ranges)
    return x, False


def dominance_probability(x):
    """P(T = j) di UNA replica, cella per cella.
    x: (N, 8, C) intensità.  Ritorna (8, C): per ogni etichetta, la
    frazione di campioni in cui è la tecnologia ottimale"""
    n_labels = x.shape[1]
    all_nan = np.all(np.isnan(x), axis=1)
    best = np.argmin(np.where(np.isnan(x), np.inf, x), axis=1)   # (N, C)
    best[all_nan] = -1
    return np.stack([(best == l).mean(axis=0) for l in range(n_labels)])


# ---------------------------------------------------------------------
# Analisi e grafici
# ---------------------------------------------------------------------
def slope(ns, err):
    ok = err > 0
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(ns[ok]), np.log(err[ok]), 1)[0])


def plot_convergence(ns, sigma, title, color, tol, path):
    """sigma: (n_N, n_celle) del solo gruppo da disegnare"""
    fig, ax = plt.subplots(figsize=(8, 6))

    first = True
    for c in range(sigma.shape[1]):
        if np.any(sigma[:, c] > 0):
            ax.plot(ns, sigma[:, c], color=color, lw=0.8, alpha=0.6,
                    label="single cells" if first else None)
            first = False

    err_max = sigma.max(axis=1)
    ax.plot(ns, err_max, "k-o", lw=2, label="maximum over cells")

    n_anchor = N_TESI if N_TESI in ns else ns[-1]
    k = int(np.flatnonzero(ns == n_anchor)[0])
    if err_max[k] > 0:
        ax.plot(ns, err_max[k] * (ns / n_anchor) ** -0.5, "r--", label=r"$\propto N^{-1/2}$")
    ax.axhline(tol, color="0.3", ls=":", label=f"tolerance ({tol:g})")
    ax.axvline(N_TESI, color="0.6", lw=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log", nonpositive="mask")
    ax.set_xlabel("Number of samples N", fontsize=FS_LABEL)
    ax.set_ylabel("Standard error of P(T = j)", fontsize=FS_LABEL)
    ax.set_title(f"{title}\nobserved slope of the maximum: {slope(ns, err_max):.2f}",
                 fontsize=FS_TITLE, fontweight="bold")
    ax.tick_params(which="both", labelsize=FS_TICKS)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=FS_LEGEND)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def report_group(ns, sigma, title, tol):
    err_max = sigma.max(axis=1)
    err_rms = np.sqrt((sigma ** 2).mean(axis=1))
    worst = sigma.argmax(axis=1)
    print(f"\n{title}  (tolleranza {tol:g})")
    print(pd.DataFrame(dict(N=ns, errore_max=err_max, errore_rms=err_rms,
                            cella_peggiore=worst)).to_string(index=False,
                                                             float_format="%.4f"))
    print(f"  pendenza osservata del massimo in log-log: {slope(ns, err_max):.2f} "
          "(attesa circa -0.5)")
    if N_TESI in ns:
        e = err_max[int(np.flatnonzero(ns == N_TESI)[0])]
        esito = "SOTTO" if e <= tol else "SOPRA"
        print(f"  -> a N = {N_TESI}: errore massimo {e:.4f}, {esito} la tolleranza {tol:g}")


def main():
    pd.set_option("display.width", 200)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # --- celle di controllo ------------------------------------------
    ref = PropagationResult.load(REFERENCE_PATH)
    if ref.labels != technology_labels():
        raise ValueError("Le etichette del run di riferimento non corrispondono "
                         "a quelle attuali del modello")
    cells = select_cells(ref)
    kinds = np.array([c["kind"] for c in cells])
    for g in GROUPS:
        if not (kinds == g).any():
            raise ValueError(f"Nessuna cella di tipo '{g}' trovata nel run di riferimento")
    speeds = np.array([c["speed"] for c in cells])
    ranges = np.array([c["range"] for c in cells])

    print("=" * 78)
    print("CELLE DI CONTROLLO (P dal run di riferimento)")
    print("=" * 78)
    print(pd.DataFrame([dict(cella=k, tipo=c["kind"], range_nmi=round(c["range"], 1),
                             velocita_kt=round(c["speed"], 1),
                             prima=c["prima"], P1=round(c["p_prima"], 3),
                             seconda=c["seconda"], P2=round(c["p_seconda"], 3))
                        for k, c in enumerate(cells)]).to_string(index=False))

    n_tot = sum(N_LIST) * R_REPLICHE
    t_est = (n_tot * len(cells) * len(TECHNOLOGIES) * len(PROPULSORS)
             * MS_PER_EVAL / 1000 / max(N_WORKERS, 1))
    print(f"\n{len(N_LIST)} valori di N x {R_REPLICHE} repliche = {n_tot} campioni "
          f"su {len(cells)} celle: stima ~{t_est / 60:.0f} min (con {N_WORKERS} worker)")

    # --- repliche ----------------------------------------------------
    specs = build_default_specs()
    corr = build_default_correlations(specs)
    theta_acc = load_theta_acc(THETA_ACC_PATH)

    ns = np.array(N_LIST)
    sigma = np.zeros((len(ns), len(cells)))

    pool_ctx = (ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init,
                                    initargs=(speeds, ranges))
                if N_WORKERS > 1 else contextlib.nullcontext())
    t0 = time.time()
    with pool_ctx as pool:
        for a, n in enumerate(ns):
            Ps = []
            for r in range(R_REPLICHE):
                seed = SEED_BASE + 100 * int(n) + r
                x, resumed = run_replica(int(n), seed, theta_acc, specs, corr,
                                         speeds, ranges, pool)
                Ps.append(dominance_probability(x))
                print(f"  N={n:5d}  replica {r + 1:2d}/{R_REPLICHE}"
                      f"{'  (ripresa da checkpoint)' if resumed else ''}"
                      f"   {time.time() - t0:.0f}s trascorsi")

            Ps = np.stack(Ps)                            # (R, 8, C)
            # deviazione standard FRA LE REPLICHE = errore della stima a
            # questo N; in ogni cella il massimo sulle 8 etichette
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                sigma[a] = Ps.std(axis=0, ddof=1).max(axis=0)
            # stima identica in tutte le repliche: ~1e-16 per arrotondamento, è zero
            sigma[a][sigma[a] < 1e-12] = 0.0

    # --- risultati ---------------------------------------------------
    print("\n" + "=" * 78)
    print("ERRORE DELLA PROBABILITÀ DI DOMINANZA AL VARIARE DI N")
    print("=" * 78)
    sb, st = sigma[:, kinds == "confine"], sigma[:, kinds == "soglia"]
    report_group(ns, sb, "Celle di CONFINE (P ~ 0.5, errore massimo della mappa)",
                 TOL_BOUNDARY)
    report_group(ns, st, f"Celle di SOGLIA (max P ~ {P_THRESHOLD}, classificazione robusta/incerta)",
                 TOL_THRESHOLD)
    print(f"\nNota: con {R_REPLICHE} repliche la deviazione standard stessa è incerta "
          f"di circa ±{100 / np.sqrt(2 * (R_REPLICHE - 1)):.0f}%: "
          "guarda l'andamento, non il singolo punto.")

    pd.DataFrame([dict(N=int(n), cella=c, tipo=cell["kind"], range_nmi=cell["range"],
                       velocita_kt=cell["speed"], errore=sigma[a, c])
                  for a, n in enumerate(ns) for c, cell in enumerate(cells)]
                 ).to_csv(CSV_PATH, index=False)
    print(f"\nErrori per cella salvati in {CSV_PATH}")

    plot_convergence(ns, sb, "Convergence at the Technology Boundaries (P ≈ 0.5)",
                     "tab:orange", TOL_BOUNDARY, FIG_BOUNDARY_PATH)
    plot_convergence(ns, st, f"Convergence at the Robustness Threshold (P ≈ {P_THRESHOLD:.2f})",
                     "tab:green", TOL_THRESHOLD, FIG_THRESHOLD_PATH)
    print(f"Grafici salvati in {FIG_BOUNDARY_PATH.name} e {FIG_THRESHOLD_PATH.name}")


if __name__ == "__main__":
    # indispensabile con N_WORKERS > 1
    main()
