#!/usr/bin/env python3
# evolution.py – NSGA-II codesign con policy inheritance + checkpoint
# V-PAR 1.0  (14-lug-2025)
# ════════════════════════════════════════════════════════════════════════
#  • Sul laptop (1 GPU) resta seriale.
#  • Su server multi-GPU lancia training+evaluation in parallelo via Ray.
#  • Nessun altro file richiesto; serve solo `pip/conda install ray-default filelock`.
# ════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import os, datetime, time, random, copy
from pathlib import Path
from typing import Optional
import numpy as np, pandas as pd
from deap import base, creator, tools
from filelock import FileLock
import torch

# ──────── PARALLEL (auto-detect) ────────────────────────────────────────
def want_parallel() -> bool:
    """True se ci sono ≥2 GPU **o** GA_PARALLEL=1.  GA_PARALLEL=0 forza off."""
    flag = os.getenv("GA_PARALLEL", "auto").lower()
    if flag in ("0", "false", "no"):   return False
    if flag in ("1", "true", "yes"):   return True
    return torch.cuda.device_count() > 1

USE_PARALLEL = want_parallel()

# ───── valori non validi (sentinelle) ─────
INVALID_V = {0}           # velocità media non valida
INVALID_E = {-100}        # energia negativa corrispondente a +100 prima del flip
INVALID_P = {0}           # progress / manovrabilità non valida


# ──────── IMPORT RESTANTI  (dopo eventuale Ray init) ────────────────────
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 side-effect

from urdf_maker import UrdfMaker
from winged_drone_train import train as train_rl
import eval as eval_mod

# ╭──────────────────────────────────────────────────────────────────────╮
# │  FUNZIONE REMOTA  training+evaluation (1 GPU)                       │
# ╰──────────────────────────────────────────────────────────────────────╯
# -------------------------------------------------------
# funzione locale sempre presente
def _train_and_eval_sync(chromo, parent_info, tag, cfg, return_arrays=True):
    parent_exp, parent_ckpt = parent_info
    urdf_file = UrdfMaker(chromo).create_urdf()
    exp = Path(urdf_file).stem
    train_iters = cfg["TRAIN_ITERS_INHERIT"] if parent_exp and parent_ckpt \
                  else cfg["TRAIN_ITERS"]

    train_rl(exp, urdf_file, cfg["TRAIN_ENVS"], train_iters,
             parent_exp=parent_exp, parent_ckpt=parent_ckpt)

    out = eval_mod.evaluation(exp_name=exp, urdf_file=urdf_file, ckpt=train_iters,
                              envs=cfg["EVAL_ENVS"], vmin=cfg["VMIN"], vmax=cfg["VMAX"],
                              return_arrays=return_arrays)

    if return_arrays:
        v_dict, e_dict, p_dict, _, extra = out
        max_p = extra["max_p"]
    else:
        v_dict, e_dict, p_dict, _, max_p = out
    meta = dict(
        vel_v=v_dict["mean_v"], vel_E=-v_dict["mean_E"], vel_P=v_dict["mean_progress"],
        eff_v=e_dict["mean_v"], eff_E=-e_dict["mean_E"], eff_P=e_dict["mean_progress"],
        prog_v=p_dict["mean_v"], prog_E=-p_dict["mean_E"], prog_P=p_dict["mean_progress"],
        train_it=train_iters, exp_name=exp,
        max_p=max_p
    )

    ff   = [v_dict["mean_v"], -e_dict["mean_E"], p_dict["mean_progress"]]

    return ff, meta, (extra if return_arrays else None)


def _eval_only_sync(exp_name, urdf_file, ckpt, cfg, return_arrays=True):
    out = eval_mod.evaluation(
        exp_name=exp_name, urdf_file=urdf_file, ckpt=ckpt,
        envs=cfg["EVAL_ENVS"], vmin=cfg["VMIN"], vmax=cfg["VMAX"],
        return_arrays=return_arrays,
    )

    if return_arrays:
        v_dict, e_dict, p_dict, _, extra = out
        max_p = extra["max_p"]
    else:
        v_dict, e_dict, p_dict, _, max_p = out

    meta = dict(
        vel_v=v_dict["mean_v"], vel_E=-v_dict["mean_E"], vel_P=v_dict["mean_progress"],
        eff_v=e_dict["mean_v"], eff_E=-e_dict["mean_E"], eff_P=e_dict["mean_progress"],
        prog_v=p_dict["mean_v"], prog_E=-p_dict["mean_E"], prog_P=p_dict["mean_progress"],
        train_it=ckpt, exp_name=exp_name,
        max_p=max_p,
    )
    ff = [v_dict["mean_v"], -e_dict["mean_E"], p_dict["mean_progress"]]
    return ff, meta, (extra if return_arrays else None)


# versione Ray (solo se parallelo)
if USE_PARALLEL:
    import ray
    ray.init(log_to_driver=False)

    @ray.remote(num_gpus=1)
    def train_and_eval_remote(*args, **kw):
        return _train_and_eval_sync(*args, **kw)

    @ray.remote(num_gpus=1)
    def eval_only_remote(*args, **kw):
        return _eval_only_sync(*args, **kw)

else:  # --- seriale -------------------------------------------------
    def train_and_eval_remote(*args, **kw):
        return _train_and_eval_sync(*args, **kw)

    def eval_only_remote(*args, **kw):
        return _eval_only_sync(*args, **kw)



# ╭─────────────────────────────────────────────────────────────╮
# │  Post-processing helper                                     │
# ╰─────────────────────────────────────────────────────────────╯
class PostAnalyzer:
    def __init__(self, csv_path="deap_temp.csv",
                 stats_obj: Optional["Stats"] = None,
                 pkl_path: Optional[str] = None):
        self.df = pd.read_csv(csv_path)
        # ------------------------------------------------------------------
        # sostituisci ogni sentinella con NaN (così Matplotlib li ignora)
        for col in ("vel_v","eff_v","prog_v"):
            self.df.loc[self.df[col].isin(INVALID_V), col] = np.nan
        for col in ("vel_E","eff_E","prog_E"):
            self.df.loc[self.df[col].isin(INVALID_E), col] = np.nan
        for col in ("vel_P","eff_P","prog_P"):
            self.df.loc[self.df[col].isin(INVALID_P), col] = np.nan
        # ------------------------------------------------------------------
        self.stats = stats_obj
        if self.stats is None and pkl_path and Path(pkl_path).is_file():
            import pickle
            with open(pkl_path, "rb") as f:
                self.stats = pickle.load(f)

        self.vel = self.df[["vel_v",  "vel_E",  "vel_P"]].rename(
                                    columns={"vel_v":"v","vel_E":"E","vel_P":"P"})
        self.eff = self.df[["eff_v",  "eff_E",  "eff_P"]].rename(
                                    columns={"eff_v":"v","eff_E":"E","eff_P":"P"})
        self.prog= self.df[["prog_v","prog_E","prog_P"]].rename(
                                    columns={"prog_v":"v","prog_E":"E","prog_P":"P"})

    def _plot_triangles(self, ax, xs, ys, color="gray", alpha=0.25, lw=0.8):
        """
        Disegna segmenti che collegano i tre punti corrispondenti allo
        stesso individuo.  *xs* e *ys* sono liste/array di len == 3.
        """
        for x, y in zip(xs, ys):
            # chiudi il triangolo ripetendo il primo punto
            ax.plot([x[0], x[1], x[2], x[0]],
                    [y[0], y[1], y[2], y[0]],
                    color=color, alpha=alpha, lw=lw)
            
    # —— 1) pairwise scatter-matrix (tutte le triple accorpate) -------------
    def scatter_matrix(self, out="scatter_matrix.png"):
        cols = ["vel_v", "vel_E", "vel_P",
                "eff_v", "eff_E", "eff_P",
                "prog_v", "prog_E", "prog_P"]
        self.df[cols] = self.df[cols].fillna(method="ffill")
        pd.plotting.scatter_matrix(self.df[cols], figsize=(8, 8),
                                   alpha=0.4, diagonal="hist")
        plt.suptitle("All objectives (3×3) – raw scatter")
        plt.savefig(out, dpi=150); plt.close()
        print("✓ scatter-matrix →", out)

    # —— 2) progress (best per gen) ----------------------------------------
    def fronts_progress_V(self, out="best_vel_per_gen.png"):
        if self.stats is None:
            print("No stats – skip progress plot"); return
        V = np.where(np.isin(self.stats.V, list(INVALID_V)), np.nan, self.stats.V)
        plt.plot(np.nanmax(V, axis=1), label="velocity ↑")
        plt.xlabel("generation"); plt.ylabel("best value"); plt.legend()
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        print("✓ progress plot →", out)

    # —— 2) progress (best per gen) ----------------------------------------
    def fronts_progress_P(self, out="best_prog_per_gen.png"):
        if self.stats is None:
            print("No stats – skip progress plot"); return

        # curva 1 ─ migliore progress per generazione
        M = np.where(np.isin(self.stats.M, list(INVALID_P)), np.nan, self.stats.M)
        best_prog = np.nanmax(M, axis=1)

        # curva 2 ─ minimal_p salvato nel CSV
        if "minimal_p" in self.df.columns:
            min_p_ser = self.df.groupby("generation")["minimal_p"].first()
            # assicurati che abbia la stessa lunghezza di best_prog
            min_p_curve = min_p_ser.reindex(range(len(best_prog)))

        plt.figure()
        plt.plot(best_prog, label="best progress ↑")

        if "minimal_p" in self.df.columns:
            plt.plot(min_p_curve, "--", label="minimal p threshold")

        plt.xlabel("generation"); plt.ylabel("meters")
        plt.legend(); plt.tight_layout()
        plt.savefig(out, dpi=150); plt.close()
        print("✓ progress plot →", out)


    # —— 2) progress (best per gen) ----------------------------------------
    def fronts_progress_E(self, out="best_eff_per_gen.png"):
        if self.stats is None:
            print("No stats – skip progress plot"); return
        E = np.where(np.isin(self.stats.E, list(INVALID_E)), np.nan, self.stats.E)
        plt.plot(np.nanmax(E, axis=1), label="-energy ↑")
        plt.xlabel("generation"); plt.ylabel("best value"); plt.legend()
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        print("✓ progress plot →", out)

    # —— 3) 3-D Pareto front (tripla per tripla) ----------------------------
    def pareto_front(self, out="pareto_front_3d.png"):
        pts, labels = [], []
        for _, r in self.df.iterrows():
            pts.append((r.vel_v, r.vel_E, r.vel_P));  labels.append("vel")
            pts.append((r.eff_v, r.eff_E, r.eff_P));  labels.append("eff")
            pts.append((r.prog_v, r.prog_E, r.prog_P)); labels.append("prog")
        pts = np.asarray(pts)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        for tag, col, m in zip(["vel", "eff", "prog"],
                               ["tab:red", "tab:green", "tab:blue"],
                               ["o", "^", "s"]):
            mask = np.array(labels) == tag
            ax.scatter(pts[mask, 0], pts[mask, 1], pts[mask, 2],
                       c=col, marker=m, label=tag, alpha=0.7)
        ax.set_xlabel("velocity  ↑")
        ax.set_ylabel("-energy  ↑")
        ax.set_zlabel("progress ↑")
        plt.legend(); plt.tight_layout()
        plt.savefig(out, dpi=150); plt.close()
        print("✓ pareto 3-D →", out)

    # ---------------------------------------------------------------
    # 4) NUOVO – Pareto 2-D  (vel vs efficienza)
    # ---------------------------------------------------------------
    def pareto_vel_eff(self, out="pareto_vel_eff.png", triangles=True):
        plt.figure(figsize=(5,4))
        ax = plt.gca()

        # 1) scatter “classico”
        ax.scatter(self.vel.v,  self.vel.E,  c="tab:red",   label="vel‐opt", alpha=0.6)
        ax.scatter(self.eff.v,  self.eff.E,  c="tab:green", label="eff‐opt", alpha=0.6)
        ax.scatter(self.prog.v, self.prog.E, c="tab:blue",  label="prog‐opt", alpha=0.6)

        # 2) opzionale triangolo per individuo
        if triangles:
            xs = np.column_stack([self.vel.v,  self.eff.v,  self.prog.v])
            ys = np.column_stack([self.vel.E,  self.eff.E,  self.prog.E])
            self._plot_triangles(ax, xs, ys)

        ax.set_xlabel("velocity  [m/s]  ↑")
        ax.set_ylabel("−energy [J/m]  ↑")
        ax.set_title("Pareto: velocity vs efficiency")
        ax.legend(); plt.tight_layout()
        plt.savefig(out, dpi=150); plt.close()
        print("✓ pareto vel-eff →", out)


    # ---------------------------------------------------------------
    # 5) NUOVO – Pareto 2-D  (vel vs manovrabilità)
    # ---------------------------------------------------------------
    def pareto_vel_man(self, out="pareto_vel_maneuver.png", triangles=True):
        plt.figure(figsize=(5,4))
        plt.scatter(self.vel.v,   self.vel.P,  c="tab:red",   label="vel‐opt", alpha=0.6)
        plt.scatter(self.prog.v,  self.prog.P, c="tab:blue",  label="prog‐opt", alpha=0.6)
        plt.scatter(self.eff.v,   self.eff.P,  c="tab:green", label="eff‐opt", alpha=0.6)
        if triangles:
            xs = np.column_stack([self.vel.v,   self.eff.v,   self.prog.v])
            ys = np.column_stack([self.vel.P,   self.eff.P,   self.prog.P])
            self._plot_triangles(plt.gca(), xs, ys, color="gray", alpha=0.25, lw=0.8)
        plt.xlabel("velocity  [m/s]  ↑")
        plt.ylabel("progress [m]  ↑")
        plt.title("Pareto: velocity vs maneuverability")
        plt.legend(); plt.tight_layout()
        plt.savefig(out, dpi=150); plt.close()
        print("✓ pareto vel-man →", out)

    # ---------------------------------------------------------------
    # 5) NUOVO – Pareto 2-D  (eff vs manovrabilità)
    # ---------------------------------------------------------------
    def pareto_eff_man(self, out="pareto_eff_maneuver.png", triangles=True):
        plt.figure(figsize=(5,4))
        plt.scatter(self.eff.E , self.eff.P, c="tab:green", label="eff‐opt", alpha=0.6)
        plt.scatter(self.prog.E, self.prog.P, c="tab:blue",  label="prog‐opt", alpha=0.6)
        plt.scatter(self.vel.E , self.vel.P, c="tab:red",   label="vel‐opt", alpha=0.6)
        if triangles:
            xs = np.column_stack([self.eff.E, self.prog.E, self.vel.E])
            ys = np.column_stack([self.eff.P, self.prog.P, self.vel.P])
            self._plot_triangles(plt.gca(), xs, ys, color="gray", alpha=0.25, lw=0.8)
        plt.xlabel("−energy [J/m]  ↑")
        plt.ylabel("progress [m]  ↑")
        plt.title("Pareto: efficiency vs maneuverability")
        plt.legend(); plt.tight_layout()
        plt.savefig(out, dpi=150); plt.close()

    # —— master call --------------------------------------------------------
    def analyze(self, prefix="analysis"):
        self.scatter_matrix(f"{prefix}_scatter.png")
        self.fronts_progress_V(f"{prefix}_progress_vel.png")
        self.fronts_progress_P(f"{prefix}_progress_prog.png")
        self.fronts_progress_E(f"{prefix}_progress_eff.png")
        self.pareto_front (f"{prefix}_pareto3D.png")
        self.pareto_vel_eff(f"{prefix}_pareto_vel_eff.png")
        self.pareto_vel_man(f"{prefix}_pareto_vel_man.png")
        self.pareto_eff_man(f"{prefix}_pareto_eff_man.png")

# ╭─────────────────────────────────────────────────────────────╮
# │  Helpers – chromosome & DB                                  │
# ╰─────────────────────────────────────────────────────────────╯
class Chromosome_Drone:
    # 1) definisci le scelte ammissibili per ogni gene
    _CHOICES = [
        [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],      
        [0.15, 0.2, 0.25, 0.30, 0.35],      
        [-0.35, -0.30, -0.25, -0.20, -0.15],
        [-0.35, -0.30, -0.25, -0.20, -0.15],      
        [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],      
        [0.10, 0.12, 0.14, 0.16, 0.18, 0.20],                        
        [0.08, 0.10, 0.12, 0.14, 0.16, 0.18],                        
        [0.08, 0.10, 0.12, 0.14, 0.16, 0.18],                        
        [0.06, 0.08, 0.10, 0.12, 0.14, 0.16],                              
        [1.00],                              
        [-30.00, -20.00, -10.00, 0.00, 10.00, 20.00, 30.00],                 
        [0.10],                              
        [0.10, 0.12, 0.14, 0.16, 0.18],
        [1.5, 2.0, 2.5, 3.0],
        [1.5, 2.0, 2.5, 3.0],
        [1.0, 1.5, 2.0, 2.5, 3.0],
        [0, -1.0, -2.0, -3.0]           
    ]

    def random(self):
        """Pesca un valore random **all’interno della lista** di ogni gene."""
        return [random.choice(options) for options in self._CHOICES]

    # se ti serve ancora conoscere i bounds per la mutazione:
    @classmethod
    def bounds(cls):
        mins = [min(opts) for opts in cls._CHOICES]
        maxs = [max(opts) for opts in cls._CHOICES]
        return mins, maxs

class FitnessDB:
    def __init__(self, name: str, n_obj: int):
        self.n_obj = n_obj
        self.path  = Path(f"{name}.csv")
        self.df    = pd.read_csv(self.path) if self.path.exists() else self._blank()
        if not self.path.exists(): self.df.to_csv(self.path, index=False)

    def lookup(self, chromo):                          # evita ricalcoli
        row = self.df[self.df.chromosome == str(list(chromo))]
        if row.empty:
            return None
        # salva anche max_p sull'individuo dopo aver preso i ff (gestito fuori)
        return [row[f"ff_{i}"].min() for i in range(self.n_obj)]

    def insert(self, chromo, ff, meta):
        print(f"   ↘ writing CSV  gen={meta.get('generation')}  ff={ff}")
        row = self._blank().iloc[0]
        row.timestamp = time.time(); row.chromosome = str(list(chromo))
        for i, v in enumerate(ff): row[f"ff_{i}"] = v
        for k, v in meta.items():   row[k] = v
        lock = FileLock(str(self.path)+".lock")
        with lock:
            self.df = pd.concat([self.df, row.to_frame().T], ignore_index=True)
            row.to_frame().T.to_csv(self.path, mode="a",
                                     header=not self.path.exists(), index=False)

    def _blank(self):
        cols = ["timestamp", "chromosome"] + [f"ff_{i}" for i in range(self.n_obj)] + [
            "generation", "exp_name", "train_it", "max_p",
            "minimal_p",           # <‑‑ NUOVO
            "vel_v","vel_E","vel_P",
            "eff_v","eff_E","eff_P",
            "prog_v","prog_E","prog_P"]
        return pd.DataFrame([{c: np.nan for c in cols}])


    def get_row(self, chromo):
        """
        Ritorna la riga intera (pandas Series) relativa al cromosoma,
        oppure None se non esiste ancora nel CSV.
        """
        row = self.df[self.df.chromosome == str(list(chromo))]
        return None if row.empty else row.iloc[0]

# ╭─────────────────────────────────────────────────────────────╮
# │  Stats container                                            │
# ╰─────────────────────────────────────────────────────────────╯
class Stats:
    def __init__(self, n_pop, n_gen, n_obj):
        self.arr = np.zeros((n_obj, n_gen + 1, n_pop))
        self.arr.fill(np.nan)
    def record(self, gen, pop):
        invalid_sets = (INVALID_V, INVALID_E, INVALID_P)
        for j in range(self.arr.shape[0]):
            vals = [ind.fitness.values[j] for ind in pop]
            vals = [np.nan if v in invalid_sets[j] else v for v in vals]
            self.arr[j, gen] = vals
    @property
    def V(self): return self.arr[0]
    @property
    def E(self): return self.arr[1]
    @property
    def M(self): return self.arr[2]

# ╭─────────────────────────────────────────────────────────────╮
# │  NSGA-II main class                                         │
# ╰─────────────────────────────────────────────────────────────╯
class CodesignDEAP:
    TRAIN_ITERS           = 50
    TRAIN_ITERS_INHERIT   = 200
    TRAIN_ENVS            = 256
    EVAL_ENVS             = 256
    VMIN, VMAX            = 6.0, 18.0
    FAIL_VAL              = 1e6
    WEIGHTS               = (+1.0, -1.0, +1.0)   # maximize vel, -energy, maneu

    def __init__(self, n_pop=12, n_gen=20, cx_pb=0.8, mut_pb=0.3,
                 csv="deap_temp", inherit_policy=False,
                 use_dynamic_p=True, fixed_p=200.0, pct_above=50.0):
        assert n_pop % 4 == 0
        self.n_pop, self.n_gen = n_pop, n_gen
        self.cx_pb, self.mut_pb, self.inherit_policy = cx_pb, mut_pb, inherit_policy
        self.db    = FitnessDB(csv, 3)
        self.stats = Stats(n_pop, n_gen, 3)
        self.tag   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.use_dynamic_p = use_dynamic_p
        self.fixed_p       = fixed_p
        self.pct_above     = np.clip(pct_above, 0, 100)

        if "FitMulti" not in creator.__dict__:
            creator.create("FitMulti", base.Fitness, weights=self.WEIGHTS)
        if "Chrom" not in creator.__dict__:
            creator.create("Chrom", list, fitness=creator.FitMulti)

        self.tb = base.Toolbox()
        self.tb.register("attr", Chromosome_Drone().random)
        self.tb.register("ind",  tools.initIterate, creator.Chrom, self.tb.attr)
        self.tb.register("pop", tools.initRepeat, list, self.tb.ind)
        self.tb.register("mate", tools.cxOnePoint)
        self.tb.register("mutate", self._mutate, indpb=0.2)
        self.tb.register("select", tools.selNSGA2)
        self.tb.register("evaluate", self._evaluate)

    @staticmethod
    def _mutate(ind, indpb=0.2):
        for i, options in enumerate(Chromosome_Drone._CHOICES):
            if len(options) == 1:
                continue
            if random.random() < indpb:
                # evita di riestrarre lo stesso valore
                ind[i] = random.choice([v for v in options if v != ind[i]])
        return (ind,)


    def _evaluate(self, indiv):
        chromo = list(indiv)
        print(f"[evaluate] gen={getattr(self,'_gen',0)}  chr={chromo}")
        # ---- caso 1: cromosoma già presente → solo evaluation ----
        cached_row = self.db.get_row(chromo)
        if cached_row is not None:
            print(f"   ↪ cache‑hit  exp={cached_row.exp_name}  ckpt={int(cached_row.train_it)}   (solo evaluation)")
            urdf_file = UrdfMaker(chromo).create_urdf()
            exp_name  = cached_row.exp_name
            train_it  = int(cached_row.train_it)

            cfg_eval = dict(EVAL_ENVS=self.EVAL_ENVS,
                            VMIN=self.VMIN, VMAX=self.VMAX)

            if USE_PARALLEL:
                fut = eval_only_remote.remote(
                    exp_name, urdf_file, train_it, cfg_eval, True)
                indiv._pending_future = fut
                return (0.0, 0.0, 0.0)   # placeholder finché Ray non finisce

            # seriale
            ff, meta, extra = _eval_only_sync(
                exp_name, urdf_file, train_it, cfg_eval, True)
            print(f"   ✔ sync‑eval  ff={ff}  max_p={meta['max_p']:.2f}")
            indiv._meta_raw = meta
            indiv.max_p     = meta["max_p"]
            if extra:
                indiv._p_s = extra["p_s"];  indiv._v_s = extra["v_s"];  indiv._E_s = extra["E_s"]
            return tuple(ff)
        else:
            print(f"   ↪ NEW chromo → training for {self.TRAIN_ITERS} iters "
                f"(or {self.TRAIN_ITERS_INHERIT} if inheritance)")
            # ---- caso 2: cromosoma nuovo → training + evaluation ----
            parent_info = (getattr(indiv, "parent_exp", None),
                        getattr(indiv, "parent_ckpt", None))
            cfg = dict(TRAIN_ITERS=self.TRAIN_ITERS,
                    TRAIN_ITERS_INHERIT=self.TRAIN_ITERS_INHERIT,
                    TRAIN_ENVS=self.TRAIN_ENVS,
                    EVAL_ENVS=self.EVAL_ENVS,
                    VMIN=self.VMIN, VMAX=self.VMAX)

            if USE_PARALLEL:
                fut = train_and_eval_remote.remote(
                    chromo, parent_info, self.tag, cfg, True)
                indiv._pending_future = fut
                return (0.0, 0.0, 0.0)

            ff, meta, extra = _train_and_eval_sync(
                chromo, parent_info, self.tag, cfg, True)
            print(f"   ✔ sync‑train+eval  ff={ff}  max_p={meta['max_p']:.2f}")
            indiv._meta_raw = meta
            indiv.max_p     = meta["max_p"]
            if extra:
                indiv._p_s = extra["p_s"];  indiv._v_s = extra["v_s"];  indiv._E_s = extra["E_s"]
            return tuple(ff)



    def _pick_triples(self, p_s, v_s, E_s, minimal_p):
        idx_p = np.argmax(p_s)                     # progress max (non filtrato)
        mask  = np.where(p_s >= minimal_p)[0]
        if mask.size == 0:
            # sentinelle
            vel = dict(mean_v=0.0, mean_E=100.0, mean_progress=0.0)
            eff = dict(mean_v=0.0, mean_E=100.0, mean_progress=0.0)
        else:
            idx_v = mask[np.argmax(v_s[mask])]     # max vel in zona valida
            idx_e = mask[np.argmin(E_s[mask])]     # min energy in zona valida
            vel = dict(mean_v=v_s[idx_v], mean_E=E_s[idx_v], mean_progress=p_s[idx_v])
            eff = dict(mean_v=v_s[idx_e], mean_E=E_s[idx_e], mean_progress=p_s[idx_e])
        prog = dict(mean_v=v_s[idx_p], mean_E=E_s[idx_p], mean_progress=p_s[idx_p])
        return vel, eff, prog

    def _finalize_and_persist(self, ind, minimal_p):
        if not hasattr(ind, "_p_s"):
            return                     # individui falliti o già in cache

        vel_d, eff_d, prog_d = self._pick_triples(
            ind._p_s, ind._v_s, ind._E_s, minimal_p)

        # Fitness finale (+vel, -E, +prog)
        ff_final = (vel_d["mean_v"], -eff_d["mean_E"], prog_d["mean_progress"])

        # meta di origine (training o solo eval) + nuovi campi filtrati
        meta = dict(getattr(ind, "_meta_raw", {}))
        meta.update(dict(
            max_p  = ind.max_p,
            minimal_p = minimal_p,
            vel_v  = vel_d["mean_v"],  vel_E  = -vel_d["mean_E"],  vel_P  = vel_d["mean_progress"],
            eff_v  = eff_d["mean_v"],  eff_E  = -eff_d["mean_E"],  eff_P  = eff_d["mean_progress"],
            prog_v = prog_d["mean_v"], prog_E = -prog_d["mean_E"], prog_P = prog_d["mean_progress"],
        ))
        print(f"[finalize] gen={self._gen} chr={list(ind)} "
                f"vel={vel_d['mean_v']:.2f}  effE={eff_d['mean_E']:.2f}  "
                f"prog={prog_d['mean_progress']:.2f}")
        self.db.insert(list(ind), ff_final, dict(generation=self._gen, **meta))
        ind.fitness.values = ff_final


    # ────────────────────────────────────────────────────────────────
    #  NSGA‑II – evolutionary loop  (select ▸ vary ▸ train/eval ▸ survive)
    # ────────────────────────────────────────────────────────────────
    def run(self):
        # ─── GEN‑0 ───
        pop = self.tb.pop(self.n_pop)
        self._gen = 0
        self._train_eval_population(pop)          # train + eval + persist
        pop = tools.selNSGA2(pop, self.n_pop)     # assegna anche crowding
        self._after_generation(pop)

        # ─── GEN ≥ 1 ───
        for g in range(1, self.n_gen + 1):
            self._gen = g
            print(f"\n════════ Generation {g}/{self.n_gen} ════════")

            # 1) parent‑selection (richiede crowding_dist calcolata qui sopra)
            parents   = tools.selTournamentDCD(pop, len(pop))
            offspring = [self.tb.clone(p) for p in parents]

            # 2) variazione (+ inheritance) prima del training
            self._apply_variation(offspring, parents)

            # 3) train + eval dei figli  → persist / minimal_p
            self._train_eval_population(offspring)

            # 4) survivor‑selection NSGA‑II  → nuova POP (con crowding aggiornata)
            pop = tools.selNSGA2(pop + offspring, self.n_pop)

            # 5) logging / grafici
            self._after_generation(pop)

        # ─── save stats globali ───
        with open(f"stats_{self.tag}.pkl", "wb") as f:
            import pickle; pickle.dump(self.stats, f)
        print("Statistics saved ✔")
        return pop
    # -----------------------------------------------------------------

    def _train_eval_population(self, population):
        """
        Allena + valuta tutti gli individui senza fitness.
        Gestisce Ray e persiste la fitness filtrata tramite _finalize_and_persist().
        """
        # 1) lancia training/eval (può creare future Ray)
        for ind in population:
            if not ind.fitness.valid:
                ind.fitness.values = self.tb.evaluate(ind)

        # 2) aspetta i job Ray
        if USE_PARALLEL:
            pend = [ind for ind in population if hasattr(ind, "_pending_future")]
            if pend:
                print(f"  ⏳ waiting {len(pend)} Ray jobs…")
                results = ray.get([ind._pending_future for ind in pend])
                for ind, (ff, meta, extra) in zip(pend, results):
                    ind._meta_raw = meta
                    ind.max_p     = meta["max_p"]
                    if extra:
                        ind._p_s = extra["p_s"]; ind._v_s = extra["v_s"]; ind._E_s = extra["E_s"]
                    ind.fitness.values = tuple(ff)
                    del ind._pending_future
                    print(f"   ✅ Ray done chr={list(ind)} ff={ff} max_p={meta['max_p']:.2f}")

        # 3) minimal_p dinamico / fisso
        peaks = [getattr(ind, "max_p", np.nan) for ind in population]
        peaks = [p for p in peaks if not np.isnan(p)]
        if self.use_dynamic_p and peaks:
            perc = 100.0 - self.pct_above
            minimal_p = 0.9 * np.percentile(peaks, perc)
        else:
            minimal_p = self.fixed_p
        print(f"[Gen {self._gen}] minimal_p = {minimal_p:.2f} "
            f"(dynamic={self.use_dynamic_p}, pct_above={self.pct_above}%)")

        # 4) finalize → CSV
        for ind in population:
            if not hasattr(ind, "_persisted"):
                self._finalize_and_persist(ind, minimal_p)
                ind._persisted = True
    # -----------------------------------------------------------------

    def _apply_variation(self, offspring, parents):
        """
        Crossover, mutazione e opzionale inheritance **prima** del training.
        """
        # pulizia attributi custom
        for ch in offspring:
            for a in ("exp_name","parent_exp","parent_ckpt","_pending_future",
                    "_meta_raw","_p_s","_v_s","_E_s","_evaluated",
                    "_persisted","max_p"):
                if hasattr(ch, a):
                    delattr(ch, a)

        for i in range(0, len(offspring), 2):
            c1, c2 = offspring[i], offspring[i+1]

            # crossover
            if random.random() < self.cx_pb:
                self.tb.mate(c1, c2)
                if hasattr(c1.fitness, "values"): del c1.fitness.values
                if hasattr(c2.fitness, "values"): del c2.fitness.values

            # mutazione
            if random.random() < self.mut_pb:
                self.tb.mutate(c1);  del c1.fitness.values
            if random.random() < self.mut_pb:
                self.tb.mutate(c2);  del c2.fitness.values

            # inheritance → assegna exp/ckpt PRIMA del training
            if self.inherit_policy:
                infos = []
                for p in (parents[i], parents[i+1]):
                    if hasattr(p, "exp_name"):
                        ck_it = getattr(p, "train_it", self.TRAIN_ITERS)
                        ck = Path(f"logs/{p.exp_name}/model_{ck_it}.pt")
                        if ck.is_file():
                            infos.append((p.exp_name, ck_it))
                if infos:
                    c1.parent_exp, c1.parent_ckpt = random.choice(infos)
                    c2.parent_exp, c2.parent_ckpt = random.choice(infos)
    # -----------------------------------------------------------------

    def _after_generation(self, pop):
        """Aggiorna stats, fa grafici e stampa riepilogo della generazione."""
        g = self._gen
        self.stats.record(g, pop)
        if g % 3 == 0 or g == self.n_gen:
            out_dir = Path(f"g{g:02d}"); out_dir.mkdir(exist_ok=True)
            PostAnalyzer(self.db.path, self.stats).analyze(prefix=f"{out_dir}/")
        best_v = np.nanmax(self.stats.V[g])
        best_e = -np.nanmin(self.stats.E[g])
        best_p = np.nanmax(self.stats.M[g])
        print(f"--- Gen {g} summary  best_vel={best_v:.2f} "
            f"best_eff={best_e:.2f}  best_prog={best_p:.2f}")


# ╭─────────────────────────────────────────────────────────────╮
# │  CLI                                                       │
# ╰─────────────────────────────────────────────────────────────╯
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=4)
    ap.add_argument("--gen", type=int, default=3)
    ap.add_argument("--train_it", type=int, default=50)
    ap.add_argument("--inherit", action="store_true", default=False,)
    args = ap.parse_args()

    CodesignDEAP(n_pop=args.pop, n_gen=args.gen,
                 inherit_policy=args.inherit).run()
