#!/usr/bin/env python3
# metrics_analysis_integrated_generic.py – N obiettivi (ff_0..ff_{N-1})
# V-PAR 4.1 (plateau robusto, sentinelle ff0/ff1, plot iperbole + Pareto collegato)

import argparse
import ast
import os
import re
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- Utility functions ----------------------
INVALID_V = {0}
INVALID_E = {-100}
INVALID_P = {0}

# --- Aggiungi queste costanti/utility in alto (accanto a INVALID_* ) ---
SENTINELS_PER_OBJ = {
    0: {0.0},      # ff_0 (velocità): 0 è invalido
    1: {-100.0},   # ff_1 (energia): -100 è invalido
    # aggiungi altri indici se mai servisse: 2: {...}, ecc.
}

def clean_objective(values: np.ndarray, obj_idx: int) -> np.ndarray:
    """Rimuove NaN e sentinelle specifiche per l'obiettivo obj_idx."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    bad = SENTINELS_PER_OBJ.get(obj_idx, set())
    if bad:
        mask = np.ones(v.shape[0], dtype=bool)
        for b in bad:
            mask &= (v != b)
        v = v[mask]
    return v

def parse_chromosome(chromo_str):
    try:
        return tuple(ast.literal_eval(chromo_str))
    except Exception:
        return None

def euclidean_dist_matrix(vals):
    if len(vals) < 2:
        return 0.0
    dists = [np.linalg.norm(vals[i] - vals[j])
             for i, j in combinations(range(len(vals)), 2)]
    return float(np.mean(dists))

def compute_entropy(values):
    if len(values) == 0:
        return 0.0
    freqs = pd.Series(values).value_counts(normalize=True)
    return float(-np.sum(freqs * np.log2(freqs + 1e-12)))

def simpson_index(items):
    counts = pd.Series(items).value_counts()
    n = counts.sum()
    if n < 2:
        return 0.0
    return float(1.0 - np.sum(counts * (counts - 1)) / (n * (n - 1)))

def crowding_distance(front):
    n, m = front.shape
    dist = np.zeros(n)
    if n <= 2:
        return dist
    for i in range(m):
        vals = front[:, i]
        idx = np.argsort(vals)
        dist[idx[0]] = dist[idx[-1]] = np.inf
        diff = vals[idx[-1]] - vals[idx[0]]
        if diff < 1e-12:
            continue
        for k in range(1, n - 1):
            j = idx[k]
            prev_j = idx[k - 1]
            next_j = idx[k + 1]
            dist[j] += (vals[next_j] - vals[prev_j]) / diff
    return dist

def spacing_metrics(front, obj_idx=0):
    n = front.shape[0]
    if n < 2:
        return 0.0, 0.0
    vals = np.sort(front[:, obj_idx])
    gaps = np.diff(vals)
    avg = np.mean(gaps)
    spacing = float(np.sqrt(np.sum((gaps - avg) ** 2) / (len(gaps) - 1))) if len(gaps) > 1 else 0.0
    max_gap = float(np.max(gaps)) if len(gaps) else 0.0
    return spacing, max_gap

def global_pareto_front(all_points):
    is_dom = np.zeros(len(all_points), dtype=bool)
    for i, pi in enumerate(all_points):
        for pj in all_points:
            if all(pj >= pi) and any(pj > pi):
                is_dom[i] = True
                break
    return all_points[~is_dom]

def generational_distance(front, reference):
    if len(reference) == 0:
        return 0.0
    if len(front) == 0:
        return float(np.inf)
    d2 = [min(np.linalg.norm(r - s) for s in front) ** 2 for r in reference]
    return float(np.sqrt(sum(d2)) / len(reference))

def inverted_generational_distance(front, reference):
    if len(front) == 0:
        return 0.0
    if len(reference) == 0:
        return float(np.inf)
    d2 = [min(np.linalg.norm(s - r) for r in reference) ** 2 for s in front]
    return float(np.sqrt(sum(d2)) / len(front))

def additive_epsilon_indicator(front, reference):
    if len(front) == 0 or len(reference) == 0:
        return np.nan
    epsilons = []
    for r in reference:
        eps_r = min(max(r[i] - s[i] for i in range(len(r))) for s in front)
        epsilons.append(eps_r)
    return float(max(epsilons))

def hypervolume_approx(front, ref_point, samples=20000):
    dim = front.shape[1]
    pts = np.random.rand(samples, dim) * ref_point
    dominated = [(front >= p).all(axis=1).any() for p in pts]
    return float(np.prod(ref_point) * sum(dominated) / samples)

# ---------------------- Triple fisico (se presente) ----------------------
def triple_subplots_generation(group: pd.DataFrame, generation: int, out_dir: str, triangles: bool=True) -> None:
    needed = {'vel_v','vel_E','vel_P','eff_v','eff_E','eff_P','prog_v','prog_E','prog_P'}
    if not needed.issubset(group.columns):
        return
    vel = group[['vel_v','vel_E','vel_P']].rename(columns={'vel_v':'v','vel_E':'E','vel_P':'P'})
    eff = group[['eff_v','eff_E','eff_P']].rename(columns={'eff_v':'v','eff_E':'E','eff_P':'P'})
    prog= group[['prog_v','prog_E','prog_P']].rename(columns={'prog_v':'v','prog_E':'E','prog_P':'P'})
    bad_map = {'v': INVALID_V, 'E': INVALID_E, 'P': INVALID_P}
    for df_ in (vel,eff,prog):
        for col in df_.columns:
            df_.loc[df_[col].isin(bad_map[col]), col] = np.nan
    fig, axes = plt.subplots(1,3,figsize=(13,4))
    ax=axes[0]; ax.scatter(vel.v,vel.E,c='tab:red',label='vel-opt',alpha=0.6)
    ax.scatter(eff.v,eff.E,c='tab:green',label='eff-opt',alpha=0.6)
    ax.scatter(prog.v,prog.E,c='tab:blue',label='prog-opt',alpha=0.6)
    if triangles:
        xs=np.column_stack([vel.v,eff.v,prog.v]); ys=np.column_stack([vel.E,eff.E,prog.E])
        for x,y in zip(xs,ys): ax.plot([x[0],x[1],x[2],x[0]],[y[0],y[1],y[2],y[0]],color='gray',alpha=0.25,lw=0.8)
    ax.set_xlabel('velocity [m/s] ↑'); ax.set_ylabel('−energy [J/m] ↑'); ax.set_title('vel vs eff'); ax.legend(fontsize='small')
    ax=axes[1]
    mask=(~vel.v.isna()) & (~vel.P.isna())
    ax.scatter(vel.v[mask],vel.P[mask],c='tab:red',label='vel-opt',alpha=0.6)
    ax.scatter(eff.v,eff.P,c='tab:green',label='eff-opt',alpha=0.6)
    ax.scatter(prog.v,prog.P,c='tab:blue',label='prog-opt',alpha=0.6)
    if triangles:
        xs=np.column_stack([vel.v,eff.v,prog.v]); ys=np.column_stack([vel.P,eff.P,prog.P])
        valid=~np.isnan(xs).any(axis=1) & ~np.isnan(ys).any(axis=1)
        for x,y in zip(xs[valid],ys[valid]): ax.plot([x[0],x[1],x[2],x[0]],[y[0],y[1],y[2],y[0]],color='gray',alpha=0.25,lw=0.8)
    ax.set_xlabel('velocity [m/s] ↑'); ax.set_ylabel('progress [m] ↑'); ax.set_title('vel vs man')
    ax=axes[2]
    ax.scatter(eff.E,eff.P,c='tab:green',label='eff-opt',alpha=0.6)
    ax.scatter(vel.E,vel.P,c='tab:red',label='vel-opt',alpha=0.6)
    ax.scatter(prog.E,prog.P,c='tab:blue',label='prog-opt',alpha=0.6)
    if triangles:
        xs=np.column_stack([eff.E,vel.E,prog.E]); ys=np.column_stack([eff.P,vel.P,prog.P])
        for x,y in zip(xs,ys): ax.plot([x[0],x[1],x[2],x[0]],[y[0],y[1],y[2],y[0]],color='gray',alpha=0.25,lw=0.8)
    ax.set_xlabel('−energy [J/m] ↑'); ax.set_ylabel('progress [m] ↑'); ax.set_title('eff vs man')
    fig.suptitle(f'Generation {generation}: triple-of-triples',fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.93])
    out_path=os.path.join(out_dir,f'triple_gen_{generation}.png')
    fig.savefig(out_path,dpi=150); plt.close(fig)
    print(f'[triple_subplots] Plot salvato → {out_path}')

# ---------------------- N-objective helpers ----------------------
def infer_objective_columns(df: pd.DataFrame, n_obj: int | None):
    ff_cols = sorted((c for c in df.columns if re.fullmatch(r'ff_\d+', c)),
                     key=lambda x: int(x.split('_')[1]))
    if n_obj is None:
        if not ff_cols:
            raise ValueError("Nessuna colonna ff_i trovata nel CSV.")
        return ff_cols
    expected = [f'ff_{i}' for i in range(n_obj)]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Mancano nel CSV le colonne: {missing}")
    return expected

def all_pair_indices(n): return list(combinations(range(n), 2))
def pairwise_grid(n_pairs):
    ncols = int(np.ceil(np.sqrt(n_pairs)))
    nrows = int(np.ceil(n_pairs / ncols))
    return nrows, ncols

# --------- Plateau robusto su minimal_p (prima gen g tale che tutte le crescite future ≤ soglia)
def find_plateau_generation(minimal_p: pd.Series, threshold: float = 0.10, smooth_window: int = 3) -> float:
    """
    Restituisce la prima generazione g tale che le crescite percentuali future
    (dopo smoothing) non superano 'threshold'. Se non esiste → np.nan.
    """
    s = minimal_p.copy().astype(float)
    if s.isna().all() or len(s) < 2:
        return np.nan
    s_smooth = s.rolling(smooth_window, min_periods=1).mean()
    pct = s_smooth.pct_change().fillna(-np.inf).values

    # suffix max: max incremento da i+1 alla fine
    suffix_max = np.empty_like(pct)
    cur = -np.inf
    for k in range(len(pct)-1, -1, -1):
        cur = max(cur, pct[k])
        suffix_max[k] = cur

    idx = minimal_p.index.to_numpy()
    for i in range(len(idx)-1):  # l'ultimo indice non può essere "inizio plateau"
        if suffix_max[i] <= threshold:
            return float(idx[i])
    return np.nan


def plot_velocity_vs_energy_pareto_after_plateau(df: pd.DataFrame,
                                                 mdf: pd.DataFrame,
                                                 ff_cols: list[str],
                                                 out_dir: str,
                                                 plateau_threshold: float = 0.10,
                                                 vel_sentinel: float = 0.0,
                                                 energy_sentinel: float = -100.0,
                                                 fname: str = 'vel_vs_energy_pareto_after_plateau.png'):
    """
    x = ff_0 (velocità, ↑ meglio), y = -ff_1 (efficienza = -energia, ↑ meglio).
    Colormap per generazione + Pareto front con dominanza (max x, max y).
    """
    from matplotlib.colors import Normalize

    assert len(ff_cols) >= 2, "Servono almeno ff_0 (vel) e ff_1 (energia)."
    c_vel, c_en = ff_cols[0], ff_cols[1]

    # Plateau su minimal_p
    if 'minimal_p' in mdf.columns and not mdf['minimal_p'].isna().all():
        g_plateau = 3  # find_plateau_generation(mdf['minimal_p'], threshold=plateau_threshold)
    else:
        g_plateau = np.nan

    # Subset post-plateau
    if np.isnan(g_plateau):
        sub = df.copy(); subtitle = "(tutte le generazioni)"
    else:
        sub = df[df['generation'] >= g_plateau].copy()
        subtitle = f"(gens >= {int(g_plateau)})"

    # Filtri sentinella
    sub = sub[(sub[c_vel] != vel_sentinel) & (sub[c_en] != energy_sentinel)]
    if sub.empty:
        print("Nessun punto valido per velocità/energia dopo i filtri.")
        return

    # Dati + colormap
    x = sub[c_vel].to_numpy()     # ↑ meglio
    y = sub[c_en].to_numpy()     # efficienza = -energia → ↑ meglio
    gens = sub['generation'].to_numpy()
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=gens.min(), vmax=gens.max())
    cmap = plt.cm.viridis

    # Pareto front (max, max) nel piano (vel, -energia)
    pts = np.column_stack([x, y])
    dominated = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        if dominated[i]:
            continue
        xi, yi = pts[i]
        for j in range(len(pts)):
            if i == j:
                continue
            xj, yj = pts[j]
            if (xj >= xi and yj >= yi) and (xj > xi or yj > yi):
                dominated[i] = True
                break
    front = pts[~dominated]
    if front.size:
        front = front[np.argsort(front[:, 0])]

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(x, y, c=gens, cmap=cmap, norm=norm, s=14, alpha=0.75, label='Individui')
    if front.size:
        ax.plot(front[:, 0], front[:, 1], lw=2.2, label='Pareto front (max vel, max eff)')

    cbar = fig.colorbar(sc, ax=ax); cbar.set_label('Generation')
    ax.set_xlabel(f'{c_vel}  (velocità, ↑ meglio)')
    ax.set_ylabel(f'-{c_en}  (efficienza, ↑ meglio)')
    ax.set_title(f'Velocità vs Efficienza {subtitle}')
    ax.grid(True, alpha=0.4, ls=':'); ax.legend(loc='best')

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[plot_velocity_vs_energy_pareto_after_plateau] Plot salvato → {out_path}")

def plot_velocity_vs_progress_pareto_after_plateau(df: pd.DataFrame,
                                                   mdf: pd.DataFrame,
                                                   ff_cols: list[str],
                                                   out_dir: str,
                                                   plateau_threshold: float = 0.10,
                                                   vel_sentinel: float = 0.0,
                                                   progress_sentinel: float = 0.0,
                                                   fname: str = 'vel_vs_progress_pareto_after_plateau.png'):
    """
    x = ff_0 (velocità, ↑ meglio), y = ff_2 (progress, ↑ meglio).
    Dominanza (max x, max y) → front in alto a destra.
    """
    from matplotlib.colors import Normalize
    assert len(ff_cols) >= 3, "Servono almeno ff_0 (vel), ff_1 (energia), ff_2 (progress)."
    c_vel, c_prog = ff_cols[0], ff_cols[2]

    if 'minimal_p' in mdf.columns and not mdf['minimal_p'].isna().all():
        g_plateau = 3  # find_plateau_generation(mdf['minimal_p'], threshold=plateau_threshold)
    else:
        g_plateau = np.nan

    if np.isnan(g_plateau):
        sub = df.copy(); subtitle = "(tutte le generazioni)"
    else:
        sub = df[df['generation'] >= g_plateau].copy()
        subtitle = f"(gens >= {int(g_plateau)})"

    sub = sub[(sub[c_vel] != vel_sentinel) & (sub[c_prog] != progress_sentinel)]
    if sub.empty:
        print("Nessun punto valido per velocità/progress dopo i filtri.")
        return

    x = sub[c_vel].to_numpy()
    y = sub[c_prog].to_numpy()
    gens = sub['generation'].to_numpy()
    norm = Normalize(vmin=gens.min(), vmax=gens.max()); cmap = plt.cm.viridis

    pts = np.column_stack([x, y])
    dominated = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        if dominated[i]: continue
        xi, yi = pts[i]
        for j in range(len(pts)):
            if i == j: continue
            xj, yj = pts[j]
            if (xj >= xi and yj >= yi) and (xj > xi or yj > yi):
                dominated[i] = True; break
    front = pts[~dominated]
    if front.size: front = front[np.argsort(front[:, 0])]

    fig, ax = plt.subplots(figsize=(7,5))
    sc = ax.scatter(x, y, c=gens, cmap=cmap, norm=norm, s=14, alpha=0.75, label='Individui')
    if front.size: ax.plot(front[:,0], front[:,1], lw=2.2, label='Pareto front (max vel, max progress)')
    cbar = fig.colorbar(sc, ax=ax); cbar.set_label('Generation')

    ax.set_xlabel(f'{c_vel}  (velocità, ↑ meglio)')
    ax.set_ylabel(f'{c_prog}  (progress, ↑ meglio)')
    ax.set_title(f'Velocità vs Progress {subtitle}')
    ax.grid(True, alpha=0.4, ls=':'); ax.legend(loc='best')

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[plot_velocity_vs_progress_pareto_after_plateau] Plot salvato → {out_path}")

def plot_efficiency_vs_progress_pareto_after_plateau(df: pd.DataFrame,
                                                     mdf: pd.DataFrame,
                                                     ff_cols: list[str],
                                                     out_dir: str,
                                                     plateau_threshold: float = 0.10,
                                                     energy_sentinel: float = -100.0,
                                                     progress_sentinel: float = 0.0,
                                                     fname: str = 'eff_vs_progress_pareto_after_plateau.png'):
    """
    x = -ff_1 (efficienza = -energia, ↑ meglio), y = ff_2 (progress, ↑ meglio).
    Dominanza (max x, max y) → front in alto a destra.
    """
    from matplotlib.colors import Normalize
    assert len(ff_cols) >= 3, "Servono almeno ff_0 (vel), ff_1 (energia), ff_2 (progress)."
    c_en, c_prog = ff_cols[1], ff_cols[2]

    if 'minimal_p' in mdf.columns and not mdf['minimal_p'].isna().all():
        g_plateau = 3  # find_plateau_generation(mdf['minimal_p'], threshold=plateau_threshold)
    else:
        g_plateau = np.nan

    if np.isnan(g_plateau):
        sub = df.copy(); subtitle = "(tutte le generazioni)"
    else:
        sub = df[df['generation'] >= g_plateau].copy()
        subtitle = f"(gens >= {int(g_plateau)})"

    sub = sub[(sub[c_en] != energy_sentinel) & (sub[c_prog] != progress_sentinel)]
    if sub.empty:
        print("Nessun punto valido per efficienza/progress dopo i filtri.")
        return

    x = sub[c_en].to_numpy()   # ← correzione qui
    y =  sub[c_prog].to_numpy()
    gens = sub['generation'].to_numpy()
    norm = Normalize(vmin=gens.min(), vmax=gens.max()); cmap = plt.cm.viridis

    pts = np.column_stack([x, y])
    dominated = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        if dominated[i]: continue
        xi, yi = pts[i]
        for j in range(len(pts)):
            if i == j: continue
            xj, yj = pts[j]
            if (xj >= xi and yj >= yi) and (xj > xi or yj > yi):
                dominated[i] = True; break
    front = pts[~dominated]
    if front.size: front = front[np.argsort(front[:, 0])]

    fig, ax = plt.subplots(figsize=(7,5))
    sc = ax.scatter(x, y, c=gens, cmap=cmap, norm=norm, s=14, alpha=0.75, label='Individui')
    if front.size: ax.plot(front[:,0], front[:,1], lw=2.2, label='Pareto front (max eff, max progress)')
    cbar = fig.colorbar(sc, ax=ax); cbar.set_label('Generation')

    ax.set_xlabel(f'-{c_en}  (efficienza, ↑ meglio)')
    ax.set_ylabel(f'{c_prog}  (progress, ↑ meglio)')
    ax.set_title(f'Efficienza vs Progress {subtitle}')
    ax.grid(True, alpha=0.4, ls=':'); ax.legend(loc='best')

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[plot_efficiency_vs_progress_pareto_after_plateau] Plot salvato → {out_path}")


def plot_three_2d_pareto_stacked(df: pd.DataFrame,
                                 mdf: pd.DataFrame,
                                 ff_cols: list[str],
                                 out_dir: str,
                                 plateau_threshold: float = 0.10,  # kept for signature compatibility
                                 vel_sentinel: float = 0.0,
                                 energy_sentinel: float = -100.0,
                                 progress_sentinel: float = 0.0,
                                 fname: str = 'stacked_pareto_vel-energy_prog-energy_prog-vel.png'):
    """
    Three vertically stacked 2D Pareto fronts:
      1) Velocity vs Energy  (Energy on X, decreasing → right)
      2) Progress vs Energy  (Energy on X, decreasing → right)
      3) Progress vs Velocity

    - Energy is plotted as normal "Energy [J/m]" (no minus).
    - Axes arranged so that the 'best' region is top-right in all subplots.
    - Single slim colorbar for Generation on the far right.
    """
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    assert len(ff_cols) >= 3, "Need at least ff_0 (velocity), ff_1 (energy), ff_2 (progress)."
    c_vel, c_en, c_prog = ff_cols[0], ff_cols[1], ff_cols[2]

    # -------- Plateau: first generation with minimal_p == 500 --------
    if ('minimal_p' in mdf.columns) and (not mdf['minimal_p'].isna().all()):
        mask_plateau = (mdf['minimal_p'] == 500)
        if mask_plateau.any():
            g_plateau = float(mdf.index[mask_plateau].min())
        else:
            g_plateau = np.nan
    else:
        g_plateau = np.nan

    # Subset post-plateau
    if np.isnan(g_plateau):
        sub = df.copy()
        subtitle = "(all generations)"
    else:
        sub = df[df['generation'] >= g_plateau].copy()
        subtitle = f"(gens ≥ {int(g_plateau)})"

    if sub.empty:
        print("[plot_three_2d_pareto_stacked] No data to plot.")
        return

    # Helper: nondominated mask assuming max-max
    def nondominated_mask_max(pts: np.ndarray) -> np.ndarray:
        n = len(pts)
        dom = np.zeros(n, dtype=bool)
        for i in range(n):
            if dom[i]:
                continue
            xi, yi = pts[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj = pts[j]
                if (xj >= xi and yj >= yi) and (xj > xi or yj > yi):
                    dom[i] = True
                    break
        return ~dom

    # Shared colormap by generation (single colorbar for all)
    gens_all = sub['generation'].to_numpy()
    norm = Normalize(vmin=np.nanmin(gens_all), vmax=np.nanmax(gens_all))
    cmap = plt.cm.viridis
    smappable = ScalarMappable(norm=norm, cmap=cmap)  # for the shared colorbar

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=False, constrained_layout=True)
    plt.subplots_adjust(hspace=0.35)

    # ---------- 1) Velocity vs Energy (Energy on X, decreasing → right) ----------
    mask1 = (sub[c_vel] != vel_sentinel) & (sub[c_en] != energy_sentinel) & \
            (~sub[c_vel].isna()) & (~sub[c_en].isna())
    x1 = - sub.loc[mask1, c_en].to_numpy()      # Energy [J/m] → lower is better
    y1 = sub.loc[mask1, c_vel].to_numpy()     # Velocity [m/s] → higher is better
    g1 = sub.loc[mask1, 'generation'].to_numpy()

    axes[0].scatter(x1, y1, c=g1, cmap=cmap, norm=norm, s=14, alpha=0.75, label='Individuals')

    # Dominance for front: maximize (velocity, -energy)
    pts_dom1 = np.column_stack([y1, -x1])
    if len(pts_dom1):
        nd1 = nondominated_mask_max(pts_dom1)
        fr1 = np.column_stack([x1[nd1], y1[nd1]])
        if fr1.size:
            fr1 = fr1[np.argsort(fr1[:, 0])]  # sort by Energy (ascending)
            axes[0].plot(fr1[:, 0], fr1[:, 1], lw=2.2, label='Pareto front')

    #axes[0].invert_xaxis()  # Energy decreases to the right
    axes[0].set_xlabel('Cost of Transport [J/m]')
    axes[0].set_ylabel('Velocity [m/s]')
    axes[0].set_title(f'Velocity vs Efficiency', fontsize=14)
    axes[0].grid(True, alpha=0.4, ls=':')
    axes[0].legend(loc='best', fontsize='small')

    # ---------- 2) Progress vs Energy (Energy on X, decreasing → right) ----------
    mask2 = (sub[c_prog] != progress_sentinel) & (sub[c_en] != energy_sentinel) & \
            (~sub[c_prog].isna()) & (~sub[c_en].isna())
    x2 = - sub.loc[mask2, c_en].to_numpy()      # Energy [J/m]
    y2 = sub.loc[mask2, c_prog].to_numpy()    # Progress [m]
    g2 = sub.loc[mask2, 'generation'].to_numpy()

    axes[1].scatter(x2, y2, c=g2, cmap=cmap, norm=norm, s=14, alpha=0.75, label='Individuals')

    pts_dom2 = np.column_stack([y2, -x2])  # maximize (progress, -energy)
    if len(pts_dom2):
        nd2 = nondominated_mask_max(pts_dom2)
        fr2 = np.column_stack([x2[nd2], y2[nd2]])
        if fr2.size:
            fr2 = fr2[np.argsort(fr2[:, 0])]
            axes[1].plot(fr2[:, 0], fr2[:, 1], lw=2.2, label='Pareto front')

    #axes[1].invert_xaxis()
    axes[1].set_xlabel('Cost of Transport [J/m]')
    axes[1].set_ylabel('Progress [m]')
    axes[1].set_title(f'Progress vs Efficiency', fontsize=14)
    axes[1].grid(True, alpha=0.4, ls=':')
    axes[1].legend(loc='best', fontsize='small')

    # ---------- 3) Progress vs Velocity (both increase → best at top-right) ----------
    mask3 = (sub[c_prog] != progress_sentinel) & (sub[c_vel] != vel_sentinel) & \
            (~sub[c_prog].isna()) & (~sub[c_vel].isna())
    x3 = sub.loc[mask3, c_vel].to_numpy()     # Velocity [m/s]
    y3 = sub.loc[mask3, c_prog].to_numpy()    # Progress [m]
    g3 = sub.loc[mask3, 'generation'].to_numpy()

    axes[2].scatter(x3, y3, c=g3, cmap=cmap, norm=norm, s=14, alpha=0.75, label='Individuals')

    pts_dom3 = np.column_stack([x3, y3])      # maximize (velocity, progress)
    if len(pts_dom3):
        nd3 = nondominated_mask_max(pts_dom3)
        fr3 = np.column_stack([x3[nd3], y3[nd3]])
        if fr3.size:
            fr3 = fr3[np.argsort(fr3[:, 0])]
            axes[2].plot(fr3[:, 0], fr3[:, 1], lw=2.2, label='Pareto front')

    axes[2].set_xlabel('Velocity [m/s]')
    axes[2].set_ylabel('Progress [m]')
    axes[2].set_title(f'Progress vs Velocity', fontsize=14)
    axes[2].grid(True, alpha=0.4, ls=':')
    axes[2].legend(loc='best', fontsize='small')

    # -------- Slim shared colorbar on the far right --------
    # Use a ScalarMappable so the colorbar exists even if one subplot has no points.
    cbar = fig.colorbar(
        smappable,
        ax=axes,             # spans all subplots
        location='right',
        fraction=0.025,      # make it slim
        pad=0.02,            # small gap from the axes stack
        aspect=30            # tall & skinny look
    )
    cbar.set_label('Generation')

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_three_2d_pareto_stacked] Plot saved → {out_path}")



# ---------------------- Main analysis ----------------------
def main(csv_path, out_dir, n_obj=None, ff_sentinel=0):
    df = pd.read_csv(csv_path)
    df['chromosome_parsed'] = df['chromosome'].apply(parse_chromosome) if 'chromosome' in df.columns else None
    os.makedirs(out_dir, exist_ok=True)

    ff_cols = infer_objective_columns(df, n_obj)
    n_obj = len(ff_cols)

    all_fits = df[ff_cols].values
    ref_front = global_pareto_front(all_fits)
    maxs = np.max(all_fits, axis=0)
    ref_point = maxs + 0.1 * np.abs(maxs)
    mins = np.min(all_fits, axis=0)
    hv_shift = np.maximum(0.0, -mins) + 1e-9

    generations = sorted(df['generation'].unique())
    rec = []

    all_chromos = [c for c in (df['chromosome_parsed'] if 'chromosome_parsed' in df.columns else []) if c]
    max_genes = len(all_chromos[0]) if all_chromos else 0

    for g in generations:
        group = df[df['generation'] == g]


        # --- dentro il for g in generations: ---
        fits = group[ff_cols].values
        stats = {'generation': g}

        # SOSTITUISCI il loop delle statistiche con questo:
        for i, col in enumerate(ff_cols):
            vals_raw = fits[:, i]
            vals = clean_objective(vals_raw, i)   # <<< pulizia per-obiettivo
            if vals.size:
                stats[f'ff{i}_best']  = float(np.max(vals))
                stats[f'ff{i}_worst'] = float(np.min(vals))
                stats[f'ff{i}_mean']  = float(np.mean(vals))
                stats[f'ff{i}_std']   = float(np.std(vals))
            else:
                stats[f'ff{i}_best'] = stats[f'ff{i}_worst'] = stats[f'ff{i}_mean'] = stats[f'ff{i}_std'] = np.nan

        stats['minimal_p'] = (group['minimal_p'].iloc[0]
                              if ('minimal_p' in group.columns and len(group) > 0)
                              else np.nan)

        stats['phen_div'] = euclidean_dist_matrix(fits)
        fits_clean = fits[~np.isnan(fits).any(axis=1)]
        gen_front  = global_pareto_front(fits_clean)

        cd = crowding_distance(gen_front)
        cd = cd[np.isfinite(cd)]
        stats['crowd_mean'] = float(cd.mean()) if cd.size else 0.0
        stats['spacing'], stats['max_gap'] = spacing_metrics(gen_front, 0)
        stats['GD']      = generational_distance(gen_front, ref_front)
        stats['IGD']     = inverted_generational_distance(gen_front, ref_front)
        stats['epsilon'] = additive_epsilon_indicator(gen_front, ref_front)

        front_pos = gen_front + hv_shift
        ref_pos   = ref_point + hv_shift
        hv_raw    = hypervolume_approx(front_pos, ref_pos) if len(gen_front) else 0.0
        stats['HV'] = hv_raw / np.prod(ref_pos) if np.all(ref_pos > 0) else np.nan

        if 'chromosome_parsed' in group.columns:
            chromos = [c for c in group['chromosome_parsed'] if c]
        else:
            chromos = []
        if chromos:
            gm = np.array(chromos)
            stats['geno_entropy'] = float(np.mean([compute_entropy(gm[:, i]) for i in range(gm.shape[1])]))
            stats['simpson'] = simpson_index(chromos)
            n_genes = gm.shape[1]
            for j in range(n_genes):
                col = gm[:, j]
                col = col[~pd.isna(col)]
                N = len(col)
                if N > 1:
                    freqs = pd.Series(col).value_counts()
                    sim_j = 1.0 - (freqs*(freqs-1)).sum()/(N*(N-1))
                else:
                    sim_j = np.nan
                stats[f'simpson_gene_{j}'] = float(sim_j)
        else:
            stats['geno_entropy'] = 0.0
            stats['simpson'] = 0.0
            for j in range(max_genes):
                stats[f'simpson_gene_{j}'] = np.nan

        rec.append(stats)

    mdf = pd.DataFrame(rec).set_index('generation')

    # >>> scatter pairwise post-plateau con nuove sentinelle per ff0/ff1
    if 'minimal_p' in mdf.columns and not mdf['minimal_p'].isna().all():
        # >>> plot speciale richiesto: ff0 vs -ff1 + Pareto collegato
        plot_velocity_vs_energy_pareto_after_plateau(
            df, mdf, ff_cols, out_dir,
            plateau_threshold=0.1,
            vel_sentinel=0.0,
            energy_sentinel=-100.0
        )

        plot_velocity_vs_progress_pareto_after_plateau(
            df, mdf, ff_cols, out_dir,
            plateau_threshold=0.1,
            vel_sentinel=0.0,
            progress_sentinel=0.0
        )
        plot_efficiency_vs_progress_pareto_after_plateau(
            df, mdf, ff_cols, out_dir,
            plateau_threshold=0.1,
            energy_sentinel=-100.0,
            progress_sentinel=0.0
        )
        plot_three_2d_pareto_stacked(
            df, mdf, ff_cols, out_dir,
            plateau_threshold=0.1,
            vel_sentinel=0.0,
            energy_sentinel=-100.0,
            progress_sentinel=0.0
        )

    # ---- helper salvataggio
    def save_plot(fig, name):
        fig.savefig(os.path.join(out_dir, f'{name}.png'), dpi=150)
        plt.close(fig)

    # ---- Evoluzione di ogni ff_i
    for i, col in enumerate(ff_cols):
        fig, ax = plt.subplots()
        ax.plot(mdf.index, mdf[f'ff{i}_mean'], label=f'Mean {col}', lw=2)
        ax.fill_between(mdf.index,
                        mdf[f'ff{i}_mean'] - mdf[f'ff{i}_std'],
                        mdf[f'ff{i}_mean'] + mdf[f'ff{i}_std'],
                        alpha=0.3, label='±1 Std Dev')
        ax.plot(mdf.index, mdf[f'ff{i}_best'], '--', label=f'Best {col}', lw=2)
        ax.set_xlabel('Generation'); ax.set_ylabel(col)
        ax.set_title(f'Evolution of {col}: Mean ± Std & Best')
        ax.legend(loc='upper left'); ax.grid(True, alpha=0.4, ls=':')
        save_plot(fig, f'{col}_evolution')

    if 'minimal_p' in mdf.columns and not mdf['minimal_p'].isna().all():
        fig, ax = plt.subplots()
        ax.plot(mdf.index, mdf['minimal_p'], label='minimal_p', lw=2)
        ax.set_xlabel('Generation'); ax.set_ylabel('minimal_p')
        ax.set_title('Minimal P threshold over generations')
        ax.legend(loc='upper left'); ax.grid(True, alpha=0.4, ls=':')
        save_plot(fig, 'minimal_p_evolution')

    # ---- Diversità fenotipica
    def stacked_subplots(x, series: dict, xlabel, title, out_name):
        n = len(series)
        fig, axes = plt.subplots(n, 1, figsize=(8, 1.8 * n), sharex=True)
        if n == 1: axes = [axes]
        for ax, (lab, y) in zip(axes, series.items()):
            ax.plot(x, y, lw=2)
            ax.set_ylabel(lab); ax.grid(True, alpha=0.4, ls=':')
        axes[-1].set_xlabel(xlabel)
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        save_plot(fig, out_name)

    stacked_subplots(
        mdf.index,
        {'Phenotypic Div.': mdf['phen_div'],
         'Crowding Dist.':  mdf['crowd_mean'],
         'Spacing (σ)':     mdf['spacing'],
         'Max Gap':         mdf['max_gap']},
        xlabel='Generation',
        title='Phenotypic Diversity Metrics per Generation',
        out_name='diversity_phenotypic'
    )

    stacked_subplots(
        mdf.index,
        {'GD': mdf['GD'], 'IGD': mdf['IGD'], 'Epsilon': mdf['epsilon'], 'Hypervol.': mdf['HV']},
        xlabel='Generation',
        title='Pareto Convergence Indicators',
        out_name='pareto_convergence'
    )

    # ---- Simpson per gene
    gene_cols = sorted([c for c in mdf.columns if c.startswith('simpson_gene_')],
                       key=lambda s: int(s.split('_')[-1]))
    if gene_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        for c in gene_cols:
            ax.plot(mdf.index, mdf[c], lw=1, label=c.replace('simpson_gene_', 'gene '))
        ax.set_xlabel('Generation'); ax.set_ylabel("Simpson's Index (per gene)")
        ax.set_title('Simpson gene-wise per generazione')
        ax.legend(ncol=2, fontsize='small', loc='upper right')
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, 'simpson_gene_wise.png'), dpi=150)
        plt.close(fig)

    # ---- Diversità genotipica
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mdf.index, mdf['geno_entropy'], label='Avg Gene Entropy', lw=2)
    ax.plot(mdf.index, mdf['simpson'], label="Simpson's Index", lw=2)
    ax.set_xlabel('Generation'); ax.set_ylabel('Diversity')
    ax.set_title('Genotypic Diversity Metrics')
    ax.legend(); ax.grid(True, alpha=0.4, ls=':')
    save_plot(fig, 'diversity_genotypic')

    # ---- Triple-of-Triples per generazione (se disponibile)
    for g in generations:
        group_g = df[df['generation'] == g]
        triple_subplots_generation(group_g, g, out_dir)

    print(f"Tutte le metriche calcolate e i plot salvati in '{out_dir}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive EA metrics with generic N objectives')
    parser.add_argument('csv_path', type=str, nargs='?', default='/home/andrea/Documents/genesis/Genesis/src/winged_drone/deap_temp.csv',
                        help='Path to the DEAP CSV file')
    parser.add_argument('out_dir', type=str, nargs='?', default='/home/andrea/Documents/genesis/Genesis/src/winged_drone/ea_plots',
                        help='Directory where plots will be saved')
    parser.add_argument('--n_obj', type=int, default=3,
                        help='Numero di fitness (ff_0..ff_{N-1}). Se omesso, le rilevo dal CSV.')
    parser.add_argument('--ff_sentinel', type=float, default=0,
                        help='(legacy) non usato per ff0/ff1: ora sono fissati a 0 e -100 nei plot post-plateau')
    parser.add_argument('--plateau_threshold', type=float, default=0.10,
                        help='Soglia crescita percentuale per definire il plateau (default 0.10 = 10%)')
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    main(args.csv_path, args.out_dir, n_obj=args.n_obj, ff_sentinel=args.ff_sentinel)
