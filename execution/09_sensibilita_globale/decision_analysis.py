"""
Quale parametro incerto governa la SCELTA dell'architettura migliore,
al variare della missione (range, velocità).

Uso (DOPO sobol_analysis.py):
    python execution/09_sensibilita_globale/decision_analysis.py

Non valuta il modello: legge tutti i blocchi del disegno (A, B, AB_i)
da sobol_cache/ e ne usa la stessa configurazione (griglia, N, seed,
fattori selezionati, candidati), importandola da sobol_analysis.py.
Pochi secondi.

In ogni nodo e in ogni campione si individua l'architettura vincente
(intensity minima fra le fattibili in quel campione); l'indice di Sobol
totale della decisione del fattore i è

    S_T,i = P(il vincitore cambia ricampionando solo i)
            / P(il vincitore cambia ricampionando tutto)

(vedi src/cnav/sensitivity_global/decision.py). La mappa riporta il
fattore con S_T massimo; dove un'architettura vince con probabilità
>= P_ROBUST la scelta è robusta e nessun parametro la governa.

IL BLOCCO DI CALIBRAZIONE SPACCHETTATO (SPLIT_CALIBRATION = True)
-----------------------------------------------------------------
Come in shapley_calibration.py, ma sulla decisione. Giocatori: i
parametri di calibrazione (correlati, righe di theta_acc). Per ogni
sottoinsieme S

    val(S) = 1 - E[ sum_a Var(1{D=a} | c_S) ] / G

Lo stimatore a vicini più prossimi di shapley_calibration.py, sommato
sulle categorie, si riduce a

    E[ sum_a Var(1{D=a} | c_S) ] ≈ P( D_i != D_nn_S(i) )

(nn_S(i): il campione più vicino a i nelle sole coordinate di S, con
le stesse regole sulle righe ripetute di theta_acc). Gli effetti di
Shapley sommano a val(tutti) = S_1 del blocco sulla decisione;
l'S_T del blocco si ripartisce fra i parametri in proporzione ai loro
effetti (st_equivalent di shapley_calibration.py), come per
l'intensity. A video il controllo: somma degli effetti contro S_1 del
blocco stimato con la Sobol (metodo indipendente).

Produce, nella sottocartella decisione/:
  decision_indices.csv   nodo x fattore: P_switch, S1, ST (con IC), gini, vincitore
  shapley_decision.csv   (se SPLIT_CALIBRATION) nodo x parametro di
                         calibrazione: effetto di Shapley e S_T equivalente
  decision_map.csv       una riga per nodo: leader, secondo, P_leader, ...
  decision_map.png       la mappa
"""
import dataclasses
import json
import sys
import time
from itertools import combinations
from math import factorial
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sobol_analysis as SA        # stessa configurazione e stessa cache della Sobol
import shapley_calibration as SC   # stesso stimatore, stessi nomi e colori
from cnav.uncertainty import load_theta_acc
from cnav.sensitivity_global import (
    FACTOR_COLORS,
    FACTOR_LABELS,
    ROBUST_LABEL,
    FactorSpace,
    decision_field,
    decision_map,
    decision_matrix,
    grid_outputs,
    outputs_table,
    plot_influence_map,
    sobol_design,
)

# ---------------------------------------------------------------------
# Configurazione (il resto viene da sobol_analysis.py)
# ---------------------------------------------------------------------
OUT_DIR = HERE / "decisione"
DPI = 140

# sopra questa probabilità di vittoria la scelta è "robusta": nessun
# parametro dominante. Va dichiarata in tesi insieme alla figura
P_ROBUST = 0.90
N_BOOTSTRAP = SA.N_BOOTSTRAP       # 0 per saltare IC e P_leader

# True: i parametri di calibrazione compaiono singolarmente (Shapley)
# al posto di calibration_block
SPLIT_CALIBRATION = True

ROBUST_COLOR = "#E0E0E0"
ROBUST_TEXT = f"Robust choice (P$_{{best}}$ $\\geq$ {P_ROBUST:.0%})"
MAP_TITLE = "Parameter Governing the Choice of the Best Architecture"
LEGEND_TITLE = r"Largest $S_{T_i}$ of the choice"


# =====================================================================
# Dati
# =====================================================================

def load_design_and_Y():
    """(design, Y, meta, ranges, speeds, space) dalla cache di sobol_analysis.py"""
    theta_acc = load_theta_acc(SA.THETA_ACC_PATH)
    space = FactorSpace.default(theta_acc)
    all_outputs = grid_outputs(SA.RANGES_NMI, SA.SPEEDS_KT)

    meta_file = SA.CACHE_DIR / "meta.json"
    if not meta_file.exists():
        raise SystemExit("Nessuna valutazione in sobol_cache/: lanciare prima sobol_analysis.py")
    meta_json = json.loads(meta_file.read_text())
    if meta_json.get("signature") != SA._signature(space, all_outputs):
        raise SystemExit("La cache in sobol_cache/ è di un'altra configurazione: "
                         "rilanciare sobol_analysis.py")

    design = sobol_design(space, SA.SELECTED_FACTORS, N=SA.N_BASE, seed=SA.SEED)
    keys = SA._block_keys(design)
    missing = [k for k in keys if not (SA.CACHE_DIR / f"{k}.npy").exists()]
    if missing:
        raise SystemExit(f"Blocchi mancanti in sobol_cache/ ({', '.join(missing)}): "
                         "rilanciare sobol_analysis.py")

    kept = set(meta_json["columns"])
    outputs = [o for o in all_outputs if o.name in kept]
    Y = pd.DataFrame(np.vstack([np.load(SA.CACHE_DIR / f"{k}.npy") for k in keys]),
                     columns=[o.name for o in outputs])
    ranges = sorted({o.range_nmi for o in all_outputs})
    speeds = sorted({o.speed_kt for o in all_outputs})
    return design, Y, outputs_table(outputs), ranges, speeds, space


# =====================================================================
# Shapley sulla decisione
# =====================================================================

def shapley_decision(X_cal: pd.DataFrame, D_ab: np.ndarray) -> tuple:
    """Effetti di Shapley dei parametri di calibrazione (colonne di X_cal)
    sulla decisione, per ogni nodo.

    X_cal (n, p) parametri di calibrazione dei campioni A+B
    D_ab  (n, K) vincitore per campione e nodo, sugli stessi campioni
    Ritorna (shapley (K, p), totale (K,) = sum_j shapley)
    """
    R, rid = SC._ranks(X_cal), SC._row_ids(X_cal)
    p = R.shape[1]
    G = 1.0 - sum((D_ab == c).mean(0) ** 2 for c in np.unique(D_ab))   # (K,)

    val = {0: np.zeros(D_ab.shape[1])}
    for size in range(1, p + 1):
        for S in combinations(range(p), size):
            nn = SC._nearest_other(R[:, list(S)], rid, exclude_same_row=size < p)
            diff = (D_ab != D_ab[nn]).mean(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                val[sum(1 << j for j in S)] = np.where(G > 0, 1.0 - diff / G, np.nan)

    full = (1 << p) - 1
    w = [factorial(s) * factorial(p - s - 1) / factorial(p) for s in range(p)]
    shap = np.zeros((D_ab.shape[1], p))
    for j in range(p):
        bit = 1 << j
        for m in range(full + 1):
            if not m & bit:
                shap[:, j] += w[bin(m).count("1")] * (val[m | bit] - val[m])
    return shap, val[full]


def split_calibration(dfl, design, D, space):
    """DecisionField con i parametri di calibrazione al posto del blocco
    (S_T, P_switch e repliche bootstrap del blocco ripartiti in
    proporzione agli effetti di Shapley), e la tabella degli effetti"""
    name = SC.CALIBRATION_FACTOR_NAME
    if name not in dfl.group_names:
        raise SystemExit(f"'{name}' non è fra i SELECTED_FACTORS di sobol_analysis.py: "
                         "aggiungerlo e rilanciare sobol_analysis.py")
    jb = dfl.group_names.index(name)

    theta = space.theta_from_unit(design.U[:2 * design.N])
    X_cal = theta[list(space.theta_acc.columns)].reset_index(drop=True)
    params = list(X_cal.columns)
    shap, total = shapley_decision(X_cal, D[:2 * design.N])

    st_eq = SC.st_equivalent(shap, dfl.ST[jb])                    # (K, p)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(dfl.ST[jb][:, None] > 0, st_eq / dfl.ST[jb][:, None], 0.0)
    share = np.nan_to_num(share)                                  # (K, p)

    keep = [i for i in range(len(dfl.group_names)) if i != jb]
    new = dataclasses.replace(
        dfl,
        group_names=[dfl.group_names[i] for i in keep] + params,
        ST=np.vstack([dfl.ST[keep], st_eq.T]),
        S1=np.vstack([dfl.S1[keep], np.full((len(params), dfl.ST.shape[1]), np.nan)]),
        P_switch=np.vstack([dfl.P_switch[keep], dfl.P_switch[jb][None] * share.T]),
        P_boot=np.concatenate([dfl.P_boot[keep],
                               dfl.P_boot[jb][None] * share.T[:, None, :]], axis=0),
    )

    nd = dfl.nodes
    tab = pd.DataFrame({
        "range_nmi": np.repeat(nd["range_nmi"].to_numpy(), len(params)),
        "velocita_kt": np.repeat(nd["velocita_kt"].to_numpy(), len(params)),
        "parametro": np.tile(params, len(nd)),
        "shapley": shap.ravel(), "ST_equivalente": st_eq.ravel(),
        "gini": np.repeat(nd["gini"].to_numpy(), len(params)),
    })

    # controllo: somma degli effetti contro S_1 del blocco (Sobol), sui
    # nodi in cui la scelta non è robusta, pesati con G come l'indice
    # generalizzato
    ok = (nd["P_vincitore"].to_numpy() < P_ROBUST) & np.isfinite(total) \
        & np.isfinite(dfl.S1[jb]) & np.isfinite(dfl.ST[jb])
    if ok.any():
        g = nd["gini"].to_numpy()[ok]
        print("\n" + "=" * 78)
        print("SHAPLEY DELLA CALIBRAZIONE SULLA SCELTA (nodi contesi, pesi = Gini)")
        print("=" * 78)
        agg = (g[:, None] * shap[ok]).sum(0) / g.sum()
        for p_, v in sorted(zip(params, agg), key=lambda t: -t[1]):
            print(f"  {SC._label(p_):<30s} {v:7.3f}")
        print(f"\nSomma degli effetti di Shapley:   {(g * total[ok]).sum() / g.sum():.3f}")
        print(f"S_1[calibration_block] (Sobol):   {(g * dfl.S1[jb][ok]).sum() / g.sum():.3f}"
              "   <- deve essere vicino alla somma")
        print(f"S_T[calibration_block] (Sobol):   {(g * dfl.ST[jb][ok]).sum() / g.sum():.3f}"
              "   <- ripartito fra i parametri nella mappa")

    cal_colors = {p_: SC.CALIBRATION_PALETTE[i % len(SC.CALIBRATION_PALETTE)]
                  for i, p_ in enumerate(params)}
    FACTOR_COLORS.update(cal_colors)
    FACTOR_LABELS.update({p_: SC._label(p_) for p_ in params})
    return new, tab


# =====================================================================
# Figura e lancio
# =====================================================================

def plot_decision_map(mp: pd.DataFrame, ranges, speeds):
    colors = {**FACTOR_COLORS, ROBUST_LABEL: ROBUST_COLOR}
    labels = {**FACTOR_LABELS, ROBUST_LABEL: ROBUST_TEXT}
    # senza n_architetture plot_influence_map non aggiunge al sottotitolo
    # la dicitura sui pesi (qui non ce ne sono)
    ax = plot_influence_map(mp.drop(columns="n_architetture"), ranges, speeds,
                            "", title=MAP_TITLE, legend_title=LEGEND_TITLE,
                            colors=colors, labels=labels)
    ax.set_title(MAP_TITLE, fontsize=18, fontweight="bold")
    return ax.figure


def main():
    pd.set_option("display.width", 200)
    OUT_DIR.mkdir(exist_ok=True)

    t0 = time.time()
    design, Y, meta, ranges, speeds, space = load_design_and_Y()
    D, nodes = decision_matrix(Y, meta, candidates=SA.BEST_CANDIDATES)
    dfl = decision_field(design, D, nodes, n_boot=N_BOOTSTRAP, seed=SA.SEED)
    dfl.table.to_csv(OUT_DIR / "decision_indices.csv", index=False)
    print(f"Indici della decisione calcolati in {time.time() - t0:.0f}s "
          f"({len(nodes)} nodi, {design.n_groups} fattori)")

    if SPLIT_CALIBRATION:
        t0 = time.time()
        dfl, tab = split_calibration(dfl, design, D, space)
        tab.to_csv(OUT_DIR / "shapley_decision.csv", index=False)
        print(f"\nEffetti di Shapley calcolati in {time.time() - t0:.0f}s")

    mp = decision_map(dfl, ranges, speeds, p_robust=P_ROBUST,
                      include_residual=SA.LEADER_INCLUDE_RESIDUAL)
    mp.to_csv(OUT_DIR / "decision_map.csv", index=False)

    feas = mp[mp["fattibile"]]
    print("\n" + "=" * 78)
    print(f"PARAMETRO DOMINANTE DELLA SCELTA (robusta se P_best >= {P_ROBUST:.0%})")
    print("=" * 78)
    print(feas["leader"].value_counts().rename("nodi").to_frame()
          .assign(quota=lambda d: (d["nodi"] / len(feas)).round(3)).to_string())
    contesi = feas[feas["leader"] != ROBUST_LABEL]
    if len(contesi) and contesi["P_leader"].notna().any():
        print(f"\nNodi contesi: {len(contesi)}; primato netto (P_leader >= 0.9) in "
              f"{(contesi['P_leader'] >= 0.9).mean():.0%} di essi")

    fig = plot_decision_map(mp, ranges, speeds)
    fig.savefig(OUT_DIR / "decision_map.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nRisultati in {OUT_DIR}")


if __name__ == "__main__":
    main()
