#!/usr/bin/env python3
# evolution.py – NSGA-II codesign con policy inheritance + checkpoint
# V-PAR 1.1  (aggiornato per radice persistente temp/)
# -------------------------------------------------------------------

from __future__ import annotations
import os, datetime, time, random, copy, io, contextlib
from pathlib import Path
from typing import Optional
import numpy as np, pandas as pd
from deap import base, creator, tools
from filelock import FileLock
import torch
import ray

# ============================================================================
#  RUN ROOT (persistente)  ---------------------------------------------------
#  Viene fornito dal wrapper Slurm via env:
#    LOG_ROOT     -> /workspace/bind/temp/logs
#    URDF_DIR     -> /workspace/bind/temp/urdf_generated
#    ANALYSIS_DIR -> /workspace/bind/temp/analysis
#  Se mancano, creiamo fallback locali (in PWD) così il codice resta portabile.
# ============================================================================
URDF_BASE         = Path(os.getenv("URDF_DIR", "./urdf_generated")).expanduser().resolve()
_LOG_ROOT_ENV     = Path(os.getenv("LOG_ROOT", "./logs")).expanduser().resolve()
_ANALYSIS_ROOT_ENV= Path(os.getenv("ANALYSIS_DIR", "./analysis")).expanduser().resolve()

for _p in (URDF_BASE, _LOG_ROOT_ENV, _ANALYSIS_ROOT_ENV):
    _p.mkdir(parents=True, exist_ok=True)

# --- compat path "logs/" relativo (per moduli legacy) -----------------------
# Se nel CWD non esiste nulla chiamato "logs", facciamo un symlink → _LOG_ROOT_ENV.
# Questo permette a winged_drone_train.py / eval.py di funzionare senza patch forzata.
_compat_logs = Path("logs")
if not _compat_logs.exists():
    try:
        _compat_logs.symlink_to(_LOG_ROOT_ENV, target_is_directory=True)
    except Exception:
        pass

# ============================================================================
#  AUTO-DETECT PARALLEL ------------------------------------------------------
# ============================================================================
def want_parallel() -> bool:
    """True se ci sono ≥2 GPU **o** GA_PARALLEL=1. GA_PARALLEL=0 forza off."""
    flag = os.getenv("GA_PARALLEL", "auto").lower()
    if flag in ("0", "false", "no"):
        return False
    if flag in ("1", "true", "yes"):
        return True
    return torch.cuda.device_count() > 1

USE_PARALLEL = want_parallel()

# ============================================================================
#  SENTINELLE
# ============================================================================
INVALID_V = {0}
INVALID_E = {-100}
INVALID_P = {0}

CONST_GENES = {
    9: 1.00,   # density_scale
   11: 0.10,   # prop_radius
}

# Matplotlib headless
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from urdf_maker import UrdfMaker
import winged_drone_train as _wdt
import eval as eval_cmd
import winged_drone_eval as eval_free

def _select_eval(disable_vcmd: bool):
    """Sceglie il modulo eval in base alla presenza della commanded velocity."""
    return eval_free if disable_vcmd else eval_cmd

# ---------------------------------------------------------------------------
#  ENV utility
# ---------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        try:
            return int(float(v))
        except Exception:
            return default

def train_rl(exp, urdf_file, n_envs, train_iters, parent_exp=None, parent_ckpt=None,
             privileged=True, disable_vcmd: bool = False):    # passa log_root esplicito così *tutto* finisce in temp/logs
    return _wdt.train(
        exp, urdf_file, n_envs, train_iters,
        parent_exp=parent_exp,
        parent_ckpt=parent_ckpt,
        privileged=privileged,
        disable_commanded_velocity=disable_vcmd,
        log_root=_LOG_ROOT_ENV,
    )
# ---------------------------------------------------------------------------
#  FUNZIONE LOCALE (train+eval su *questa* GPU / processo)
#  - usa le dir persistenti definite sopra
# ---------------------------------------------------------------------------
def _train_and_eval_sync(chromo, parent_info, tag, cfg, return_arrays=True):
    parent_exp, parent_ckpt = parent_info

    URDF_BASE.mkdir(parents=True, exist_ok=True)
    urdf_file = UrdfMaker(chromo, out_dir=URDF_BASE).create_urdf()  # produce path in URDF_BASE
    exp = Path(urdf_file).stem

    # NB: train_iters può essere sovrascritto da cfg (vedi __init__)
    train_iters = cfg["TRAIN_ITERS_INHERIT"] if parent_exp and parent_ckpt else cfg["TRAIN_ITERS"]

    gpu = torch.cuda.current_device()
    print(f"[train] {exp}: start  on GPU {gpu}")
    disable = bool(cfg.get("DISABLE_VCMD", False))

    # Esegui training (winged_drone_train.py deve usare LOG_ROOT env; NO rmtree!)
    log_file = _LOG_ROOT_ENV / "ea" / exp / "train_capture.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[train] {exp}: logging PPO output both in console and in\n       {log_file}")
    log_file.parent.mkdir(exist_ok=True, parents=True)
    try:
        with open(log_file, "w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
            train_rl(
                exp, urdf_file, cfg["TRAIN_ENVS"], train_iters,
                parent_exp=parent_exp, parent_ckpt=parent_ckpt,
                privileged=True,
                disable_vcmd=disable,
            )
    except RuntimeError as e:
        msg = str(e)
        # riconosciamo solo l'errore di NaN
        if "NaN detected" in msg:
            print(f"[train] {exp}: rilevato NaN – interrompo training: {msg}")
            # Trova l’ultimo checkpoint disponibile nella cartella dei log
            ckpts = sorted(
                (p for p in (log_file.parent).glob("model_*.pt")),
                key=lambda p: int(p.stem.split("_")[1])
            )
            last_ckpt = int(ckpts[-1].stem.split("_")[1]) if ckpts else None
            if last_ckpt:
                print(f"[train] {exp}: uso last ckpt utilizzabile {last_ckpt}")
                ckpt_to_eval = last_ckpt + 1
                ev = _select_eval(disable)

                out = ev.evaluation(
                    exp_name=exp,
                    urdf_file=urdf_file,
                    ckpt=ckpt_to_eval,
                    envs=cfg["EVAL_ENVS"],
                    vmin=cfg["VMIN"],
                    vmax=cfg["VMAX"],
                    return_arrays=(return_arrays and not disable),
                )

                if return_arrays and not disable:
                    v_dict, e_dict, p_dict, _, extra = out
                    max_p = extra["max_p"]
                else:
                    v_dict, e_dict, p_dict, _, max_p = out

                meta = dict(
                    vel_v=(0.0 if cfg.get("DISABLE_VCMD", False) else v_dict["mean_v"]),
                    vel_E=(0.0 if cfg.get("DISABLE_VCMD", False) else -v_dict["mean_E"]),
                    vel_P=(0.0 if cfg.get("DISABLE_VCMD", False) else v_dict["mean_progress"]),
                    eff_v=(0.0 if cfg.get("DISABLE_VCMD", False) else e_dict["mean_v"]),
                    eff_E=(0.0 if cfg.get("DISABLE_VCMD", False) else -e_dict["mean_E"]),
                    eff_P=(0.0 if cfg.get("DISABLE_VCMD", False) else e_dict["mean_progress"]),
                    prog_v=p_dict["mean_v"], prog_E=-p_dict["mean_E"], prog_P=p_dict["mean_progress"],

                    train_it=ckpt_to_eval, exp_name=exp,
                    max_p=max_p
                )
                if cfg.get("DISABLE_VCMD", False):
                    ff = [p_dict["mean_v"], -p_dict["mean_E"], max_p]
                else:
                    ff = [v_dict["mean_v"], -e_dict["mean_E"], max_p]
                print(f"[GPU{gpu}] {exp}: V={ff[0]:.3f}  -E={ff[1]:.3f}  P={ff[2]:.3f}", flush=True)
                if cfg.get("DISABLE_VCMD", False):
                    extra = None
                return ff, meta, (extra if (return_arrays and not cfg.get("DISABLE_VCMD", False)) else None)

            else:
                print(f"[train] {exp}: nessun ckpt utilizzabile, restituisco valori di default")
                ff = [0.0, next(iter(INVALID_E)), 0.0]
                meta_tag, meta_it = exp, 0
                meta = dict(
                    vel_v=ff[0], vel_E=ff[1], vel_P=0.0,
                    eff_v=ff[0], eff_E=ff[1], eff_P=0.0,
                    prog_v=ff[0], prog_E=ff[1], prog_P=0.0,
                    train_it=meta_it, exp_name=meta_tag,
                    max_p = 0
                )
                extra = None
                print(f"[train] {exp}: fallback ff={ff}")
                return ff, meta, (extra if return_arrays else None) 
        else:
            raise

    ev = _select_eval(disable)

    out = ev.evaluation(
        exp_name=exp,
        urdf_file=urdf_file,
        ckpt=train_iters,
        envs=cfg["EVAL_ENVS"],
        vmin=cfg["VMIN"],
        vmax=cfg["VMAX"],
        return_arrays=(return_arrays and not disable),
    )

    if return_arrays and not disable:
        v_dict, e_dict, p_dict, _, extra = out
        max_p = extra["max_p"]
    else:
        v_dict, e_dict, p_dict, _, max_p = out

    meta = dict(
        vel_v=(0.0 if cfg.get("DISABLE_VCMD", False) else v_dict["mean_v"]),
        vel_E=(0.0 if cfg.get("DISABLE_VCMD", False) else -v_dict["mean_E"]),
        vel_P=(0.0 if cfg.get("DISABLE_VCMD", False) else v_dict["mean_progress"]),
        eff_v=(0.0 if cfg.get("DISABLE_VCMD", False) else e_dict["mean_v"]),
        eff_E=(0.0 if cfg.get("DISABLE_VCMD", False) else -e_dict["mean_E"]),
        eff_P=(0.0 if cfg.get("DISABLE_VCMD", False) else e_dict["mean_progress"]),
        prog_v=p_dict["mean_v"], prog_E=-p_dict["mean_E"], prog_P=p_dict["mean_progress"],

        train_it=train_iters, exp_name=exp,
        max_p=max_p
    )
    if cfg.get("DISABLE_VCMD", False):
        ff = [p_dict["mean_v"], -p_dict["mean_E"], max_p]
    else:
        ff = [v_dict["mean_v"], -e_dict["mean_E"], max_p]
    print(f"[GPU{gpu}] {exp}: V={ff[0]:.3f}  -E={ff[1]:.3f}  P={ff[2]:.3f}", flush=True)
    if cfg.get("DISABLE_VCMD", False):
        extra = None
    return ff, meta, (extra if (return_arrays and not cfg.get("DISABLE_VCMD", False)) else None)



def _eval_only_sync(exp_name, urdf_file, ckpt, cfg, return_arrays=True):
    gpu = torch.cuda.current_device()
    print(f"[eval] {exp_name}: start  on GPU {gpu}")

    disable = bool(cfg.get("DISABLE_VCMD", False))
    ev = _select_eval(disable)

    out = ev.evaluation(
        exp_name=exp_name,
        urdf_file=urdf_file,
        ckpt=ckpt,
        envs=cfg["EVAL_ENVS"],
        vmin=cfg["VMIN"],
        vmax=cfg["VMAX"],
        return_arrays=(return_arrays and not disable),
    )

    if return_arrays and not disable:
        v_dict, e_dict, p_dict, _, extra = out
        max_p = extra["max_p"]
    else:
        v_dict, e_dict, p_dict, _, max_p = out

    meta = dict(
        vel_v=(0.0 if disable else v_dict["mean_v"]),
        vel_E=(0.0 if disable else -v_dict["mean_E"]),
        vel_P=(0.0 if disable else v_dict["mean_progress"]),
        eff_v=(0.0 if disable else e_dict["mean_v"]),
        eff_E=(0.0 if disable else -e_dict["mean_E"]),
        eff_P=(0.0 if disable else e_dict["mean_progress"]),
        prog_v=p_dict["mean_v"], prog_E=-p_dict["mean_E"], prog_P=p_dict["mean_progress"],
        train_it=ckpt, exp_name=exp_name,
        max_p=max_p,
    )
    ff = [p_dict["mean_v"], -p_dict["mean_E"], max_p] if disable else [v_dict["mean_v"], -e_dict["mean_E"], max_p]
    print(f"[GPU{gpu}] {exp_name}: V={ff[0]:.3f}  -E={ff[1]:.3f}  P={ff[2]:.3f}", flush=True)
    return ff, meta, (extra if (return_arrays and not disable) else None)


# ============================================================================
#  RAY ACTOR-POOL placeholders (inizializzati lazy in _ensure_ray())
# ============================================================================
_RAY_READY = False
_ACTOR_CYCLE_TRAIN_EVAL = None   # pool per train + eval
_ACTOR_CYCLE_EVAL       = None   # pool per solo eval

def _ensure_ray(cfg_for_actors):
    """Inizializza Ray e crea actor pool se non già fatto."""
    global _RAY_READY, _ACTOR_CYCLE_TRAIN_EVAL, _ACTOR_CYCLE_EVAL
    if _RAY_READY or not USE_PARALLEL:
        return

    # —— se c’è già un cluster ————————————————————————————————
    address = os.getenv("RAY_ADDRESS")
    if address:
        ray.init(
            address=address,
            namespace="codesign",
            runtime_env={"working_dir": str(Path(__file__).parent)},
            ignore_reinit_error=True
        )
        # attende che tutti i nodi abbiano registrato le GPU previste
        expected_gpu = _env_int("RAY_GPUS", 1)
        for _ in range(30):                      # max ~60 s
            if ray.cluster_resources().get("GPU", 0) >= expected_gpu:
                break
            time.sleep(2)
    else:
        # cluster locale “stand-alone”
        total_cpus = _env_int("RAY_CPUS", 4)
        total_gpus = _env_int("RAY_GPUS", 1)
        ray.init(
            num_cpus=total_cpus,
            num_gpus=total_gpus,
            include_dashboard=False,
            runtime_env={"working_dir": str(Path(__file__).parent)}
        )

    # —— 2. risorse effettive -----------------------------------------------
    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    total_cpus = int(ray.cluster_resources().get("CPU", 0))

    if total_gpus == 0:
        print("[ray] WARNING: Ray sees 0 GPU → fallback serial.")
        return

    print("[ray] TOTAL RESOURCES: ", ray.cluster_resources())

    # metà CPU del nodo per actor, per lasciare spazio ai thread BLAS/PyTorch
    _cpus_per_actor = max(1, (total_cpus // total_gpus) // 2)

    @ray.remote(num_cpus=_cpus_per_actor, num_gpus=1)
    class Worker:                       # <‑‑ unico actor per GPU
        def __init__(self, cfg):
            self.cfg = cfg

        # train + eval
        def train_eval(self, chromo, parent_info, tag, return_arrays):
            return _train_and_eval_sync(
                chromo, parent_info, tag, self.cfg, return_arrays)

        # solo eval
        def eval_only(self, exp_name, urdf_file, ckpt, return_arrays):
            return _eval_only_sync(
                exp_name, urdf_file, ckpt, self.cfg, return_arrays)

    workers = [Worker.remote(cfg_for_actors) for _ in range(total_gpus)]
    import itertools
    _WORKER_CYCLE              = itertools.cycle(workers)
    _ACTOR_CYCLE_TRAIN_EVAL    = _WORKER_CYCLE   # stesso pool
    _ACTOR_CYCLE_EVAL          = _WORKER_CYCLE   # idem
    _RAY_READY                 = True

def _dispatch_train_eval(chromo, parent_info, tag, cfg):
    if USE_PARALLEL:
        _ensure_ray(cfg)
        if _RAY_READY:
            w = next(_ACTOR_CYCLE_TRAIN_EVAL)
            return w.train_eval.remote(chromo, parent_info, tag, True)

    return _train_and_eval_sync(chromo, parent_info, tag, cfg, True)


def _dispatch_eval(exp_name, urdf_file, ckpt, cfg, return_arrays):

    if USE_PARALLEL:
        _ensure_ray(cfg)
        if _RAY_READY:
            w = next(_ACTOR_CYCLE_EVAL)
            return w.eval_only.remote(exp_name, urdf_file, ckpt, return_arrays)

    return _eval_only_sync(exp_name, urdf_file, ckpt, cfg, return_arrays)


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
        cols = ["vel_v","vel_E","vel_P","eff_v","eff_E","eff_P","prog_v","prog_E","prog_P"]
        df = self.df.reindex(columns=cols)

        # elimina colonne completamente NaN e righe completamente NaN
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="all")

        if df.shape[1] < 2 or df.shape[0] == 0:
            print("No sufficient data for scatter-matrix – skip")
            return

        pd.plotting.scatter_matrix(df, figsize=(8,8), alpha=0.4, diagonal="hist")
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

    def fronts_progress_P(self, out="best_prog_per_gen.png"):
        # Se manca la colonna prog_P, esci
        if "prog_P" not in self.df.columns:
            print("No prog_P column – skip progress plot")
            return

        # 1) curva 1 – migliore progress per generazione
        best_prog = self.df.groupby("generation")["prog_P"].max()

        # 2) curva 2 – minimal_p salvato nel CSV (se presente)
        if "minimal_p" in self.df.columns:
            min_p = self.df.groupby("generation")["minimal_p"].first()
        else:
            min_p = None

        # Plot
        plt.figure()
        plt.plot(best_prog.index, best_prog.values, label="best progress ↑")
        if min_p is not None:
            plt.plot(min_p.index, min_p.values, "--", label="minimal p threshold")

        plt.xlabel("generation")
        plt.ylabel("meters")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
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
        # Se non ci sono righe, evita l’errore di indicizzazione su array vuoto
        if self.df.empty:
            print("No data for 3-D pareto – skip")
            return

        pts, labels = [], []
        for _, r in self.df.iterrows():
            pts.extend([
                (r.vel_v,  r.vel_E,  r.vel_P),
                (r.eff_v,  r.eff_E,  r.eff_P),
                (r.prog_v, r.prog_E, r.prog_P),
            ])
            labels.extend(["vel", "eff", "prog"])

        pts = np.asarray(pts, dtype=float)
        # Se per qualche ragione non ci sono punti validi, salta il plot
        if pts.size == 0 or pts.ndim != 2 or pts.shape[1] != 3:
            print("No points for 3-D pareto – skip")
            return

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        for tag, col, m in zip(["vel", "eff", "prog"],
                            ["tab:red", "tab:green", "tab:blue"],
                            ["o", "^", "s"]):
            mask = (np.array(labels) == tag)
            # Matplotlib ignora i NaN: non serve filtrare, non disegnerà quei punti
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
        [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],      
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4],      
        [-0.35, -0.31, -0.27, -0.23, -0.19, -0.15],
        [-0.35, -0.32, -0.29, -0.26, -0.23, -0.20],      
        [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],      
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
            self.df.to_csv(self.path, index=False)

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
    def P(self): return self.arr[2]
# ╭─────────────────────────────────────────────────────────────╮
# │  NSGA-II main class                                         │
# ╰─────────────────────────────────────────────────────────────╯
class CodesignDEAP:
    TRAIN_ITERS           = 1000
    TRAIN_ITERS_INHERIT   = 800
    TRAIN_ENVS            = 16384
    EVAL_ENVS             = 4096
    VMIN, VMAX            = 5.0, 25.0
    FAIL_VAL              = 1e4
    WEIGHTS               = (+1.0, +1.0, +1.0)   # maximize vel, -energy

    def __init__(self, n_pop=12, n_gen=20, cx_pb=0.90, mut_pb=0.3,
                 csv=None, inherit_policy=False,
                 use_dynamic_p=True, fixed_p=500.0, pct_above=50.0,
                 disable_vcmd: bool = False):
        assert n_pop % 4 == 0
        self.n_pop, self.n_gen = n_pop, n_gen
        self.cx_pb, self.mut_pb, self.inherit_policy = cx_pb, mut_pb, inherit_policy
        if csv is None:
            csv = str(_ANALYSIS_ROOT_ENV / "deap_temp")
        self.db    = FitnessDB(csv, 3)
        self.stats = Stats(n_pop, n_gen, 3)
        self.tag   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.use_dynamic_p = use_dynamic_p
        self.fixed_p       = fixed_p
        self.pct_above     = np.clip(pct_above, 0, 100)
        self.disable_vcmd  = bool(disable_vcmd)

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
    def _mutate(ind, indpb=0.1):
        for i, options in enumerate(Chromosome_Drone._CHOICES):
            if len(options) == 1:
                continue
            if random.random() < indpb:
                # evita di riestrarre lo stesso valore
                ind[i] = random.choice([v for v in options if v != ind[i]])
        return (ind,)


    # ─────── evaluate (seriale o remota) ──────────────────────────────────
    def _evaluate(self, indiv):
        chromo=list(indiv)
        print(f"[evaluate] gen={getattr(self,'_gen',0)}  chr={chromo}")
        cached_row = self.db.get_row(chromo)  # Series con meta già presente
        if cached_row is not None:
            print(f"   ↪ cache‑hit  exp={cached_row.exp_name}  ckpt={int(cached_row.train_it)}   (solo evaluation)")

            exp_name  = cached_row.exp_name
            urdf_path = (URDF_BASE / f"{exp_name}.urdf").resolve()

            # 2) crea il file solo se non esiste già
            if urdf_path.is_file():
                urdf_file = str(urdf_path)
            else:                           # prima volta: generiamo l’URDF
                urdf_file = UrdfMaker(chromo, out_dir=URDF_BASE).create_urdf()
                urdf_file = str(Path(urdf_file).resolve())
            exp_name  = cached_row.exp_name
            train_it  = int(cached_row.train_it)

            cfg_eval = dict(EVAL_ENVS=self.EVAL_ENVS,
                    VMIN=self.VMIN, VMAX=self.VMAX,
                    DISABLE_VCMD=self.disable_vcmd)

            if USE_PARALLEL:
                print("[eval]  Ray submit", chromo)
                fut = _dispatch_eval(
                    exp_name, urdf_file, train_it, cfg_eval, True)
                indiv._pending_future = fut
                return (0.0, 0.0, 0.0)   # placeholder finché Ray non finisce
            else:
                ff, meta, extra = _dispatch_eval(
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
            parent_info=(getattr(indiv,"parent_exp",None),
                        getattr(indiv,"parent_ckpt",None))
            cfg = dict(
                TRAIN_ITERS=self.TRAIN_ITERS,
                TRAIN_ITERS_INHERIT=self.TRAIN_ITERS_INHERIT,
                TRAIN_ENVS=self.TRAIN_ENVS,
                EVAL_ENVS=self.EVAL_ENVS,
                VMIN=self.VMIN, VMAX=self.VMAX,
                DISABLE_VCMD=self.disable_vcmd,   # <── aggiungi questo
            )

            if USE_PARALLEL:
                print("[train+eval]  Ray submit", chromo)
                fut = _dispatch_train_eval(chromo, parent_info, self.tag, cfg)
                indiv._pending_future = fut
                return (0.0, 0.0, 0.0)
            else:
                ff, meta, extra = _dispatch_train_eval(chromo, parent_info, self.tag, cfg)


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



    # ────────────────────────────────────────────────────────────────
    #  NSGA‑II – evolutionary loop  (select ▸ vary ▸ train/eval ▸ survive)
    # ────────────────────────────────────────────────────────────────
    def run(self):
        # ─── GEN‑0 ───
        pop = self.tb.pop(self.n_pop)
        self._gen = 0
        self._train_eval_population(pop)          # train + eval + persist
        m0 = self._compute_minimal_p(pop)                # soglia unica della gen-0
        for ind in pop:                                   # fitness finale + persist
            self._finalize_one(ind, m0, persist=True)
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

            minimal_p = self._compute_minimal_p(pop + offspring)

            peaks_all = [getattr(i, "max_p", np.nan) for i in (pop + offspring)]
            peaks_all = [p for p in peaks_all if not np.isnan(p)]
            if not peaks_all:
                print(f"[Gen {g}] minimal_p = {minimal_p:.2f} | no peaks available (N=0)")
            else:
                q25, q50, q75 = (np.percentile(peaks_all, q) for q in (25,50,75))
                print(f"[Gen {g}] minimal_p = {minimal_p:.2f} | max_p stats: "
                    f"min={min(peaks_all):.1f} q25={q25:.1f} med={q50:.1f} q75={q75:.1f} max={max(peaks_all):.1f}  "
                    f"(N={len(peaks_all)})")

            for ind in offspring:                        # figli: fitness + CSV
                self._finalize_one(ind, minimal_p, persist=True)
            for ind in pop:                              # padri: fitness ricalcolata
                self._finalize_one(ind, minimal_p, persist=False)

            def _is_feasible(ind):
                # feasible se la scelta non è la sentinella
                return hasattr(ind, "_p_s") and not (
                    np.isclose(ind.fitness.values[0], 0.0) and
                    np.isclose(ind.fitness.values[1], next(iter(INVALID_E)))
                )

            f_par = sum(_is_feasible(i) for i in pop)
            f_off = sum(_is_feasible(i) for i in offspring)
            print(f"[Gen {g}] feasible: parents {f_par}/{len(pop)} | offspring {f_off}/{len(offspring)}")

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
        Stage-1: allena/valuta chi non ha la fitness; riempie:
        - ind._meta_raw, ind.max_p
        - ind._p_s, ind._v_s, ind._E_s   (se disponibili)
        NON calcola minimal_p, NON persiste, NON fissa la fitness finale.
        """
        # 1) lancia training/eval (crea eventuali future Ray)
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
                    # placeholder (non usata per la selezione)
                    ind.fitness.values = tuple(ff)
                    del ind._pending_future
                    print(f"   ✅ Ray done chr={list(ind)} ff={ff} max_p={meta['max_p']:.2f}")

    # -----------------------------------------------------------------

    def _compute_minimal_p(self, inds):
        peaks = [getattr(i, "max_p", np.nan) for i in inds]
        peaks = [p for p in peaks if not np.isnan(p)]
        if not (self.use_dynamic_p and peaks):
            return self.fixed_p
        m = np.percentile(peaks, 100.0 - self.pct_above)
        if m > self.fixed_p:
            return self.fixed_p
        else:
            return m

    def _finalize_one(self, ind, minimal_p, persist=False):
        """
        Converte arrays → triple filtrate → fitness finale,
        e opzionalmente scrive nel CSV.
        """
        if self.disable_vcmd:
            # ramo NO-VCMD: non abbiamo il filtro; usiamo i meta già calcolati
            prog_d = dict(mean_v=getattr(ind, "_meta_raw", {}).get("prog_v", np.nan),
                        mean_E=-getattr(ind, "_meta_raw", {}).get("prog_E", np.nan),
                        mean_progress=getattr(ind, "_meta_raw", {}).get("prog_P", np.nan))
            vel_d  = dict(mean_v=0.0, mean_E=0.0, mean_progress=0.0)
            eff_d  = dict(mean_v=0.0, mean_E=0.0, mean_progress=0.0)
            ff_final = (prog_d["mean_v"], -prog_d["mean_E"], prog_d["mean_progress"])
            ind.fitness.values = ff_final
            if persist and not hasattr(ind, "_persisted"):
                meta = dict(getattr(ind, "_meta_raw", {}))
                meta.update(dict(
                    max_p=getattr(ind, "max_p", prog_d["mean_progress"]),
                    minimal_p=minimal_p,
                    vel_v=vel_d["mean_v"],  vel_E=vel_d["mean_E"],  vel_P=vel_d["mean_progress"],
                    eff_v=eff_d["mean_v"],  eff_E=eff_d["mean_E"],  eff_P=eff_d["mean_progress"],
                    prog_v=prog_d["mean_v"], prog_E=-prog_d["mean_E"], prog_P=prog_d["mean_progress"],
                    generation=self._gen,
                ))
                self.db.insert(list(ind), ff_final, meta)
                ind._persisted = True
            return

        # ramo standard (con commanded velocity) — servono gli arrays
        if not hasattr(ind, "_p_s"):
            # individuo fallito → fitness pessima coerente
            ind.fitness.values = (0.0, next(iter(INVALID_E)), 0.0)
            return

        vel_d, eff_d, prog_d = self._pick_triples(ind._p_s, ind._v_s, ind._E_s, minimal_p)

        # ⚠️ fitness DEAP = (+vel, +(-energy))
        ff_final = (vel_d["mean_v"], -eff_d["mean_E"], prog_d["mean_progress"])
        ind.fitness.values = ff_final

        if persist and not hasattr(ind, "_persisted"):
            meta = dict(getattr(ind, "_meta_raw", {}))
            meta.update(dict(
                max_p=getattr(ind, "max_p", prog_d["mean_progress"]),
                minimal_p=minimal_p,
                vel_v=vel_d["mean_v"],  vel_E=-vel_d["mean_E"],  vel_P=vel_d["mean_progress"],
                eff_v=eff_d["mean_v"],  eff_E=-eff_d["mean_E"],  eff_P=eff_d["mean_progress"],
                prog_v=prog_d["mean_v"], prog_E=-prog_d["mean_E"], prog_P=prog_d["mean_progress"],
                generation=self._gen,
            ))
            self.db.insert(list(ind), ff_final, meta)
            ind._persisted = True

        feasible = not (vel_d["mean_v"] == 0.0 and eff_d["mean_E"] == 100.0)
        status = "OK " if feasible else "FAIL"
        print(f"[finalize g={self._gen}] {status} "
            f"p*={getattr(ind,'max_p',float('nan')):.1f}  thr={minimal_p:.1f}  "
            f"vel(v={vel_d['mean_v']:.2f}, E={vel_d['mean_E']:.1f})  "
            f"eff(v={eff_d['mean_v']:.2f}, E={eff_d['mean_E']:.1f})  "
            f"ff=({ind.fitness.values[0]:.2f}, {ind.fitness.values[1]:.2f}, {ind.fitness.values[2]:.2f})")

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

            if self.inherit_policy:
                infos=[]
                for p in (parents[i], parents[i+1]):
                    if hasattr(p,"exp_name"):
                        ck_it=getattr(p,"train_it",self.TRAIN_ITERS)
                        LOG_ROOT = Path(os.getenv("LOG_ROOT", ".")).expanduser().resolve()
                        CK_DIRS = [
                            LOG_ROOT / "ea" / p.exp_name,      # training path effettivo
                            LOG_ROOT / p.exp_name,             # fallback compat
                        ]

                        ck = None
                        for d in CK_DIRS:
                            cand = d / f"model_{ck_it}.pt"
                            if cand.is_file():
                                ck = cand
                                break
                        if ck is None:
                            continue
                        if ck.is_file(): infos.append((p.exp_name,ck_it))
                if infos:
                    c1.parent_exp,c1.parent_ckpt=random.choice(infos)
                    c2.parent_exp,c2.parent_ckpt=random.choice(infos)
    # -----------------------------------------------------------------

    def _after_generation(self, pop):
        """Aggiorna stats, fa grafici e stampa riepilogo della generazione."""
        g = self._gen
        self.stats.record(g, pop)
        if g % 2 == 0 or g == self.n_gen:
            out_dir = Path(f"g{g:02d}"); out_dir.mkdir(exist_ok=True)
            PostAnalyzer(self.db.path, self.stats).analyze(prefix=f"{out_dir}/")
        best_v = np.nanmax(self.stats.V[g])
        best_e = -np.nanmin(self.stats.E[g])
        gen_df  = self.db.df[self.db.df.generation == g]
        best_p  = gen_df['prog_P'].max() if not gen_df.empty else float('nan')
        print(f"--- Gen {g} summary  best_vel={best_v:.2f} best_eff={best_e:.2f} best_prog={best_p:.2f}")


# ╭─────────────────────────────────────────────────────────────╮
# │  CLI                                                       │
# ╰─────────────────────────────────────────────────────────────╯
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--gen", type=int, default=10)
    ap.add_argument("--train_it", type=int, default=1000)
    ap.add_argument("--inherit", action="store_true", default=False,)
    ap.add_argument("--no_vcmd", action="store_true", default=False,
                    help="disabilita commanded velocity (usa evaluation da winged_drone_eval)")

    args = ap.parse_args()

    cd = CodesignDEAP(n_pop=args.pop, n_gen=args.gen, inherit_policy=args.inherit, disable_vcmd=args.no_vcmd)
    cd.TRAIN_ITERS = args.train_it
    # opzionale: scala anche quello ereditato
    # cd.TRAIN_ITERS_INHERIT = max(1, args.train_it // 5)
    cd.run()
