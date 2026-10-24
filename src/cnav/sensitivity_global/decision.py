"""
Sensibilità della SCELTA dell'architettura migliore.

L'output non è l'intensity ma, in ogni nodo (R, V), la decisione

    D(theta) = argmin_a E_a(theta)

fra le architetture fattibili in quel campione (un'architettura
infattibile non entra nel confronto; se nessuna è fattibile D vale
NO_FEASIBLE). D cambia solo quando cambia il segno di qualche
log(E_a / E_b): è la generalizzazione a più contendenti dell'indicatore
1{Delta > 0}.

Nessuna valutazione nuova: D si ricava riga per riga dalle colonne già
in cache, in tutti i blocchi del disegno (A, B, AB_i).

L'INDICE
--------
D è categorica: la si scrive come vettore one-hot (1{D=a})_a e se ne
prende l'indice di Sobol generalizzato (Gamboa et al. 2014, lo stesso
principio dell'aggregazione in aggregation.py). La varianza totale è

    sum_a Var(1{D=a}) = 1 - sum_a p_a^2 = G       (impurità di Gini)

e lo stimatore di Jansen, sommato sulle categorie, si riduce a

    S_T,i = P( D(A) != D(AB_i) ) / G

cioè la probabilità che il vincitore CAMBI ricampionando solo il
fattore i (tutti gli altri fermi), rapportata a quella che si ha
ricampionando tutto: per A e B indipendenti, infatti,
G = P(D(A) != D(B)). Allo stesso modo (Saltelli 2010) il primo ordine:

    S_1,i = [ P(D(B) = D(AB_i)) - P(D(B) = D(A)) ] / G

LA SCELTA ROBUSTA
-----------------
Dove un'architettura vince quasi sempre (P_vincitore >= p_robust) la
scelta non dipende dall'incertezza: G ~ 0, l'indice è 0/0 e un
"parametro dominante" non esiste. Questi nodi hanno leader =
ROBUST_LABEL. Se a vincere "quasi sempre" è NO_FEASIBLE, il nodo è
infattibile.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .outputs import BEST_SYSTEM
from .sobol import RESIDUAL_GROUP

__all__ = [
    "NO_FEASIBLE",
    "ROBUST_LABEL",
    "DecisionField",
    "decision_matrix",
    "decision_field",
    "decision_map",
]

NO_FEASIBLE = -1
ROBUST_LABEL = "robust_choice"
_NONE_LABEL = "none feasible"


# ---------------------------------------------------------------------
# La decisione, riga per riga
# ---------------------------------------------------------------------

def decision_matrix(Y: pd.DataFrame, meta: pd.DataFrame,
                    candidates: Optional[list] = None) -> tuple:
    """D (n_righe, n_nodi): per ogni riga di Y (tutti i blocchi) e ogni
    nodo, l'indice dell'architettura vincente nella lista del nodo;
    NO_FEASIBLE se nessuna è fattibile.

    nodes: una riga per nodo con range_nmi, velocita_kt e architetture
    (stesso ordine degli indici di D)
    candidates: etichette di architettura da mettere in gara; None = tutte
    """
    m = meta[(meta["sistema"] != BEST_SYSTEM) & meta["output"].isin(Y.columns)]
    if candidates is not None:
        m = m[m["architettura"].isin(candidates)]
    cols, nodes = [], []
    for (r, v), cell in m.groupby(["range_nmi", "velocita_kt"], sort=False):
        vals = Y[cell["output"].tolist()].to_numpy(dtype=float)
        fin = np.isfinite(vals)
        d = np.argmin(np.where(fin, vals, np.inf), axis=1)
        cols.append(np.where(fin.any(axis=1), d, NO_FEASIBLE).astype(np.int16))
        nodes.append({"range_nmi": float(r), "velocita_kt": float(v),
                      "architetture": list(cell["architettura"])})
    return np.column_stack(cols), pd.DataFrame(nodes)


# ---------------------------------------------------------------------
# Indici
# ---------------------------------------------------------------------

@dataclass
class DecisionField:
    """nodes        una riga per nodo: gini, vincitore, P_vincitore
    group_names  fattori/gruppi (primo asse degli array)
    P_switch     (g, K) P(il vincitore cambia ricampionando solo i)
    S1, ST       (g, K) indici della decisione
    P_boot       (g, B, K) repliche bootstrap di P_switch (B = 0 se
                 n_boot = 0); servono a dire quanto è netto il primato
    table        tidy: nodo x fattore
    """
    nodes: pd.DataFrame
    group_names: list
    P_switch: np.ndarray
    S1: np.ndarray
    ST: np.ndarray
    P_boot: np.ndarray
    table: pd.DataFrame
    N: int


def decision_field(design, D: np.ndarray, nodes: pd.DataFrame, n_boot: int = 200,
                   seed: int = 0, n: Optional[int] = None) -> DecisionField:
    """Indici di Sobol della decisione in ogni nodo. D deve essere
    calcolata su Y = design.U senza riordinare le righe"""
    if len(D) != design.n_blocks * design.N:
        raise ValueError(f"D ha {len(D)} righe, il disegno ne prevede "
                         f"{design.n_blocks * design.N}")
    n = design.N if n is None else int(n)
    Db = D.reshape(design.n_blocks, design.N, -1)[:, :n]
    dA, dB, dAB = Db[0], Db[1], Db[2:]                       # (n,K) (n,K) (g,n,K)
    g, K = dAB.shape[0], D.shape[1]

    sw = (dAB != dA[None]).astype(np.float32)                # il vincitore cambia
    eq_bab = (dAB == dB[None]).astype(np.float32)
    eq_ba = (dA == dB).astype(np.float32)
    cats = np.unique(Db[:2])
    onehot = [(dA == c).astype(np.float32) + (dB == c) for c in cats]

    def stats(w):
        """w (b, n): pesi di riga che sommano a n (bootstrap = conteggi)"""
        w = np.atleast_2d(np.asarray(w, dtype=np.float32))
        P = np.einsum("bn,gnk->gbk", w, sw) / n
        s1 = (np.einsum("bn,gnk->gbk", w, eq_bab) - (w @ eq_ba)[None]) / n
        p = np.stack([w @ oh for oh in onehot]) / (2.0 * n)  # (C, b, K)
        return P, s1, 1.0 - (p ** 2).sum(0), p

    P, s1n, G, p = stats(np.ones(n))
    P, s1n, G, p = P[:, 0], s1n[:, 0], G[0], p[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ST = np.where(G > 0, P / G, np.nan)
        S1 = np.where(G > 0, s1n / G, np.nan)

    if n_boot > 0:
        rng = np.random.default_rng(seed)                     # come sobol_field
        idx = rng.integers(0, n, size=(n_boot, n))
        W = np.stack([np.bincount(i, minlength=n) for i in idx])
        P_boot, _, G_boot, _ = stats(W)
        with np.errstate(divide="ignore", invalid="ignore"):
            STb = np.where(G_boot[None] > 0, P_boot / G_boot[None], np.nan)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ST_lo, ST_hi = np.nanpercentile(STb, [2.5, 97.5], axis=1)
    else:
        P_boot = np.zeros((g, 0, K), dtype=np.float32)
        ST_lo = ST_hi = np.full((g, K), np.nan)

    win = p.argmax(axis=0)
    nodes = nodes.copy()
    nodes["gini"] = G
    nodes["P_vincitore"] = p.max(axis=0)
    nodes["vincitore"] = [
        _NONE_LABEL if cats[win[k]] == NO_FEASIBLE
        else nodes["architetture"].iloc[k][int(cats[win[k]])]
        for k in range(K)]

    table = pd.DataFrame({
        "range_nmi": np.tile(nodes["range_nmi"].to_numpy(), g),
        "velocita_kt": np.tile(nodes["velocita_kt"].to_numpy(), g),
        "fattore": np.repeat(design.group_names, K),
        "P_switch": P.ravel(), "S1": S1.ravel(), "ST": ST.ravel(),
        "ST_lo": ST_lo.ravel(), "ST_hi": ST_hi.ravel(),
        "gini": np.tile(G, g),
        "vincitore": np.tile(nodes["vincitore"].to_numpy(), g),
        "P_vincitore": np.tile(nodes["P_vincitore"].to_numpy(), g),
    })
    table["interazione"] = table["ST"] - table["S1"]
    table["N"] = n

    return DecisionField(nodes=nodes, group_names=list(design.group_names),
                         P_switch=P, S1=S1, ST=ST, P_boot=P_boot, table=table, N=n)


# ---------------------------------------------------------------------
# La mappa
# ---------------------------------------------------------------------

def decision_map(dfl: DecisionField, ranges, speeds, p_robust: float = 0.95,
                 include_residual: bool = True) -> pd.DataFrame:
    """Per ogni nodo della griglia il fattore con S_T della decisione
    massimo. Stesse colonne di influence_map (si disegna con
    plot_influence_map), più:

      P_leader     frazione delle repliche bootstrap in cui il leader
                   resta in testa (vicina a 0.5: primato non netto)
      vincitore, P_vincitore, gini

    leader = ROBUST_LABEL dove il vincitore più probabile vince con
    probabilità >= p_robust: lì nessun parametro governa la scelta.
    fattibile = False dove non c'è nessuna architettura in gara, o
    dove quasi sempre nessuna è fattibile.

    include_residual=False toglie "others" dalla corsa al primo posto
    (non dal denominatore)
    """
    names = np.array(dfl.group_names)
    cand = np.array([include_residual or nm != RESIDUAL_GROUP for nm in names])
    R = dfl.nodes["range_nmi"].to_numpy()
    V = dfl.nodes["velocita_kt"].to_numpy()
    empty = {"leader": None, "ST_leader": np.nan, "secondo": None,
             "ST_secondo": np.nan, "P_leader": np.nan}
    rows = []
    for v in speeds:
        for r in ranges:
            base = {"range_nmi": float(r), "velocita_kt": float(v)}
            k = np.flatnonzero(np.isclose(R, r, rtol=1e-6) & np.isclose(V, v, rtol=1e-6))
            if len(k) == 0:
                rows.append({**base, "fattibile": False, **empty, "n_architetture": 0,
                             "vincitore": None, "P_vincitore": np.nan, "gini": np.nan})
                continue
            k = int(k[0])
            nd = dfl.nodes.iloc[k]
            extra = {"n_architetture": len(nd["architetture"]), "vincitore": nd["vincitore"],
                     "P_vincitore": float(nd["P_vincitore"]), "gini": float(nd["gini"])}
            if nd["P_vincitore"] >= p_robust:
                feasible = nd["vincitore"] != _NONE_LABEL
                rows.append({**base, **extra, "fattibile": feasible, **empty,
                             "leader": ROBUST_LABEL if feasible else None})
                continue
            score = np.where(cand, dfl.ST[:, k], -np.inf)
            o = np.argsort(-score)
            p_lead = np.nan
            if dfl.P_boot.shape[1]:
                sb = np.where(cand[:, None], dfl.P_boot[:, :, k], -np.inf)
                p_lead = float((sb.argmax(0) == o[0]).mean())
            rows.append({**base, **extra, "fattibile": True, "leader": names[o[0]],
                         "ST_leader": float(dfl.ST[o[0], k]), "secondo": names[o[1]],
                         "ST_secondo": float(dfl.ST[o[1], k]), "P_leader": p_lead})
    return pd.DataFrame(rows)
