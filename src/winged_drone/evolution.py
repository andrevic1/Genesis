#!/usr/bin/env python3
# evolution.py – NSGA-II co-design with policy inheritance + checkpointing
# V-PAR 1.2 (patched per user requests)
# ---------------------------------------------------------------------------
# What changed vs. your version:
# 1) Removed all "DISABLE_VCMD" branches and CLI flag; a single evaluation path.
# 2) Cache-hit now triggers *re-training* (with policy inheritance from the cached ckpt).
# 3) Added reward-curve extraction from TensorBoard and plots (incl. reward/speed plot).
# 4) CSV writes rew_05pct .. rew_100pct.
# 5) Gene CHOICES replaced with the ones from the provided file.
# 6) Comments are in English.
# 7) Per-individual artifacts are organized under ANALYSIS_DIR/runs/<row_id>/ with
#    its URDF and a symlink to its logs; row_id starts from 1 and matches the CSV line.
#    Multi-GPU/Ray support is preserved.
# ---------------------------------------------------------------------------

from __future__ import annotations
import os, datetime, time, random, contextlib, shutil
from pathlib import Path
from typing import Optional, Tuple
import numpy as np, pandas as pd
from deap import base, creator, tools
from filelock import FileLock
import torch
import shutil
# ============================================================================
#  RUN ROOTS (persistent) -----------------------------------------------------
#  Provided by Slurm wrapper via env (falls back to local dirs if missing):
#    LOG_ROOT     -> /workspace/bind/temp/logs
#    URDF_DIR     -> /workspace/bind/temp/urdf_generated
#    ANALYSIS_DIR -> /workspace/bind/temp/analysis
# ============================================================================
URDF_BASE          = Path(os.getenv("URDF_DIR", "./urdf_generated")).expanduser().resolve()
_LOG_ROOT_ENV      = Path(os.getenv("LOG_ROOT", "./logs")).expanduser().resolve()
_ANALYSIS_ROOT_ENV = Path(os.getenv("ANALYSIS_DIR", "./analysis")).expanduser().resolve()
RUNS_BASE          = _ANALYSIS_ROOT_ENV / "runs"

for _p in (URDF_BASE, _LOG_ROOT_ENV, _ANALYSIS_ROOT_ENV, RUNS_BASE):
    _p.mkdir(parents=True, exist_ok=True)

# Legacy-compat symlink: create ./logs -> LOG_ROOT if missing so legacy modules work
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
    """True if there are ≥2 GPUs *or* GA_PARALLEL=1. GA_PARALLEL=0 forces off."""
    flag = os.getenv("GA_PARALLEL", "auto").lower()
    if flag in ("0", "false", "no"):
        return False
    if flag in ("1", "true", "yes"):
        return True
    return torch.cuda.device_count() > 1

USE_PARALLEL = want_parallel()

# ============================================================================
#  SENTINELS ------------------------------------------------------------------
# ============================================================================
INVALID_V = {0}
INVALID_E = {-100}
INVALID_P = {0}

# Matplotlib headless
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from urdf_maker import UrdfMaker
import winged_drone_train as _wdt
import eval as eval_cmd
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
#  ENV util ------------------------------------------------------------------
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
             privileged: bool = True):
    """Proxy to RL trainer, forcing log_root so *everything* lands in LOG_ROOT."""
    return _wdt.train(
        exp, urdf_file, n_envs, train_iters,
        parent_exp=parent_exp, parent_ckpt=parent_ckpt,
        privileged=privileged,
        log_root=_LOG_ROOT_ENV,
    )


def _tb_extract_reward_curve(exp: str, train_iters: int, n_points: int = 20, win_frac: float = 0.1) -> dict:
    """
    Read TensorBoard scalar "Train/mean_reward" from logs/ea/<exp>/ and compute
    smoothed reward at 5%, …, 100% of training. Uses a centered moving-average
    window equal to win_frac of total training steps; falls back to the nearest
    event if the window is empty.
    Returns keys: rew_05pct .. rew_100pct (floats). Missing TB → {}.
    """
    log_dir = _LOG_ROOT_ENV / "ea" / exp
    ea = event_accumulator.EventAccumulator(str(log_dir))
    try:
        ea.Reload()
    except Exception:
        return {}
    key = "Train/mean_reward"
    if key not in ea.Tags().get("scalars", []):
        return {}
    events = ea.Scalars(key)
    if not events:
        return {}
    steps = np.array([e.step for e in events], dtype=np.int64)
    vals  = np.array([e.value for e in events], dtype=np.float64)
    win_steps = max(1, int(round(train_iters * win_frac)))
    half = win_steps // 2
    out = {}
     for frac in range(1, n_points + 1):
         target = int(round(train_iters * frac / n_points))
         pct = int(round(100 * frac / n_points))  # 5,10,...,100
         lo, hi = target - half, target + half
         mask = (steps >= lo) & (steps <= hi)
         if mask.any():
             out[f"rew_{pct:02d}pct"] = float(np.nanmean(vals[mask]))
         else:
             idx = int(np.argmin(np.abs(steps - target)))
             out[f"rew_{pct:02d}pct"] = float(vals[idx])
    return out


# ---------------------------------------------------------------------------
#  LOCAL worker (train+eval on *this* GPU / process) -------------------------
#  - Uses the persistent dirs defined above
# ---------------------------------------------------------------------------

def _train_and_eval_sync(chromo, parent_info, tag, cfg, return_arrays=True):
    parent_exp, parent_ckpt = parent_info

    URDF_BASE.mkdir(parents=True, exist_ok=True)
    urdf_file = UrdfMaker(chromo, out_dir=URDF_BASE).create_urdf()  # path in URDF_BASE
    exp = Path(urdf_file).stem

    train_iters = cfg["TRAIN_ITERS_INHERIT"] if parent_exp and parent_ckpt else cfg["TRAIN_ITERS"]

    gpu = torch.cuda.current_device() if torch.cuda.is_available() else -1
    print(f"[train] {exp}: start on GPU {gpu}")

    # Capture trainer stdout/stderr in a persistent logfile
    log_file = _LOG_ROOT_ENV / "ea" / exp / "train_capture.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[train] {exp}: logging PPO output to {log_file}")

    try:
        with open(log_file, "w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
            train_rl(
                exp, urdf_file, cfg["TRAIN_ENVS"], train_iters,
                parent_exp=parent_exp, parent_ckpt=parent_ckpt,
                privileged=True,
            )
    except RuntimeError as e:
        msg = str(e)
        # Recognize NaN crash; evaluate the last usable checkpoint instead
        if "NaN detected" in msg:
            print(f"[train] {exp}: NaN detected – stopping training: {msg}")
            ckpts = sorted((p for p in (log_file.parent).glob("model_*.pt")),
                           key=lambda p: int(p.stem.split("_")[1]))
            last_ckpt = int(ckpts[-1].stem.split("_")[1]) if ckpts else None
            if last_ckpt:
                print(f"[train] {exp}: using last usable ckpt {last_ckpt}")
                ckpt_to_eval = last_ckpt + 1
                out = eval_cmd.evaluation(
                    exp_name=exp,
                    urdf_file=urdf_file,
                    ckpt=ckpt_to_eval,
                    envs=cfg["EVAL_ENVS"],
                    vmin=cfg["VMIN"],
                    vmax=cfg["VMAX"],
                    return_arrays=return_arrays,
                )
                reward_curve = _tb_extract_reward_curve(exp, ckpt_to_eval)
                if return_arrays:
                    v_dict, e_dict, p_dict, _, extra = out
                    max_p = extra["max_p"]
                else:
                    v_dict, e_dict, p_dict, _, max_p = out
                meta = dict(
                    vel_v=v_dict["mean_v"],  vel_E=-v_dict["mean_E"],  vel_P=v_dict["mean_progress"],
                    eff_v=e_dict["mean_v"],  eff_E=-e_dict["mean_E"],  eff_P=e_dict["mean_progress"],
                    prog_v=p_dict["mean_v"], prog_E=-p_dict["mean_E"], prog_P=p_dict["mean_progress"],
                    train_it=ckpt_to_eval, exp_name=exp, max_p=max_p,
                    **reward_curve,
                )
                ff = [v_dict["mean_v"], -e_dict["mean_E"], max_p]
                print(f"[GPU{gpu}] {exp}: V={ff[0]:.3f}  -E={ff[1]:.3f}  P={ff[2]:.3f}", flush=True)
                return ff, meta, (extra if return_arrays else None)
            else:
                print(f"[train] {exp}: no usable ckpt – returning default values")
                ff = [0.0, next(iter(INVALID_E)), 0.0]
                meta = dict(
                    vel_v=ff[0], vel_E=ff[1], vel_P=0.0,
                    eff_v=ff[0], eff_E=ff[1], eff_P=0.0,
                    prog_v=ff[0], prog_E=ff[1], prog_P=0.0,
                    train_it=0, exp_name=exp, max_p=0,
                )
                return ff, meta, None
        else:
            raise

    # Normal path: evaluate the produced checkpoint
    out = eval_cmd.evaluation(
        exp_name=exp,
        urdf_file=urdf_file,
        ckpt=train_iters,
        envs=cfg["EVAL_ENVS"],
        vmin=cfg["VMIN"],
        vmax=cfg["VMAX"],
        return_arrays=return_arrays,
    )
    reward_curve = _tb_extract_reward_curve(exp, train_iters)

    if return_arrays:
        v_dict, e_dict, p_dict, _, extra = out
        max_p = extra["max_p"]
    else:
        v_dict, e_dict, p_dict, _, max_p = out

    meta = dict(
        vel_v=v_dict["mean_v"],  vel_E=-v_dict["mean_E"],  vel_P=v_dict["mean_progress"],
        eff_v=e_dict["mean_v"],  eff_E=-e_dict["mean_E"],  eff_P=e_dict["mean_progress"],
        prog_v=p_dict["mean_v"], prog_E=-p_dict["mean_E"], prog_P=p_dict["mean_progress"],
        train_it=train_iters, exp_name=exp, max_p=max_p,
        **reward_curve,
    )
    ff = [v_dict["mean_v"], -e_dict["mean_E"], max_p]
    print(f"[GPU{gpu}] {exp}: V={ff[0]:.3f}  -E={ff[1]:.3f}  P={ff[2]:.3f}", flush=True)
    return ff, meta, (extra if return_arrays else None)


def _eval_only_sync(exp_name, urdf_file, ckpt, cfg, return_arrays=True):
    gpu = torch.cuda.current_device() if torch.cuda.is_available() else -1
    print(f"[eval] {exp_name}: start on GPU {gpu}")
    out = eval_cmd.evaluation(
        exp_name=exp_name,
        urdf_file=urdf_file,
        ckpt=ckpt,
        envs=cfg["EVAL_ENVS"],
        vmin=cfg["VMIN"], vmax=cfg["VMAX"],
        return_arrays=return_arrays,
    )
    if return_arrays:
        v_dict, e_dict, p_dict, _, extra = out
        max_p = extra["max_p"]
    else:
        v_dict, e_dict, p_dict, _, max_p = out
    meta = dict(
        vel_v=v_dict["mean_v"],  vel_E=-v_dict["mean_E"],  vel_P=v_dict["mean_progress"],
        eff_v=e_dict["mean_v"],  eff_E=-e_dict["mean_E"],  eff_P=e_dict["mean_progress"],
        prog_v=p_dict["mean_v"], prog_E=-p_dict["mean_E"], prog_P=p_dict["mean_progress"],
        train_it=ckpt, exp_name=exp_name, max_p=max_p,
    )
    ff = [v_dict["mean_v"], -e_dict["mean_E"], max_p]
    print(f"[GPU{gpu}] {exp_name}: V={ff[0]:.3f}  -E={ff[1]:.3f}  P={ff[2]:.3f}", flush=True)
    return ff, meta, (extra if return_arrays else None)


# ============================================================================
#  RAY ACTOR-POOL (lazy init in _ensure_ray()) -------------------------------
# ============================================================================
_RAY_READY = False
_ACTOR_CYCLE_TRAIN_EVAL = None  # pool for train+eval
_ACTOR_CYCLE_EVAL       = None  # pool for eval-only


def _ensure_ray(cfg_for_actors):
    """Initialize Ray and create actor pool if not already done."""
    global _RAY_READY, _ACTOR_CYCLE_TRAIN_EVAL, _ACTOR_CYCLE_EVAL
    if _RAY_READY or not USE_PARALLEL:
        return

    address = os.getenv("RAY_ADDRESS")
    if address:
        import ray
        ray.init(address=address, namespace="codesign",
                 runtime_env={"working_dir": str(Path(__file__).parent)},
                 ignore_reinit_error=True)
        expected_gpu = _env_int("RAY_GPUS", 1)
        for _ in range(30):  # ~60 s max
            if ray.cluster_resources().get("GPU", 0) >= expected_gpu:
                break
            time.sleep(2)
    else:
        import ray
        total_cpus = _env_int("RAY_CPUS", 4)
        total_gpus = _env_int("RAY_GPUS", 1)
        ray.init(num_cpus=total_cpus, num_gpus=total_gpus,
                 include_dashboard=False,
                 runtime_env={"working_dir": str(Path(__file__).parent)})

    import ray
    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    total_cpus = int(ray.cluster_resources().get("CPU", 0))

    if total_gpus == 0:
        print("[ray] WARNING: Ray sees 0 GPU → fallback serial.")
        return

    print("[ray] TOTAL RESOURCES:", ray.cluster_resources())
    _cpus_per_actor = max(1, (total_cpus // total_gpus) // 2)  # leave room for BLAS/torch threads

    @ray.remote(num_cpus=_cpus_per_actor, num_gpus=1)
    class Worker:  # one actor per GPU
        def __init__(self, cfg):
            self.cfg = cfg
        def train_eval(self, chromo, parent_info, tag, return_arrays):
            return _train_and_eval_sync(chromo, parent_info, tag, self.cfg, return_arrays)
        def eval_only(self, exp_name, urdf_file, ckpt, return_arrays):
            return _eval_only_sync(exp_name, urdf_file, ckpt, self.cfg, return_arrays)

    workers = [Worker.remote(cfg_for_actors) for _ in range(total_gpus)]
    import itertools
    _WORKER_CYCLE           = itertools.cycle(workers)
    _ACTOR_CYCLE_TRAIN_EVAL = _WORKER_CYCLE
    _ACTOR_CYCLE_EVAL       = _WORKER_CYCLE
    _RAY_READY              = True


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
        # Replace sentinels with NaN so Matplotlib ignores them
        for col in ("vel_v", "eff_v", "prog_v"):
            self.df.loc[self.df[col].isin(INVALID_V), col] = np.nan
        for col in ("vel_E", "eff_E", "prog_E"):
            self.df.loc[self.df[col].isin(INVALID_E), col] = np.nan
        for col in ("vel_P", "eff_P", "prog_P"):
            self.df.loc[self.df[col].isin(INVALID_P), col] = np.nan
        self.stats = stats_obj
        if self.stats is None and pkl_path and Path(pkl_path).is_file():
            import pickle
            with open(pkl_path, "rb") as f:
                self.stats = pickle.load(f)
        self.vel = self.df[["vel_v",  "vel_E",  "vel_P"]].rename(columns={"vel_v": "v", "vel_E": "E", "vel_P": "P"})
        self.eff = self.df[["eff_v",  "eff_E",  "eff_P"]].rename(columns={"eff_v": "v", "eff_E": "E", "eff_P": "P"})
        self.prog= self.df[["prog_v", "prog_E", "prog_P"]].rename(columns={"prog_v": "v", "prog_E": "E", "prog_P": "P"})

    def fronts_progress_V(self, out="best_vel_per_gen.png"):
        if self.stats is None:
            print("No stats – skip progress plot"); return
        V = np.where(np.isin(self.stats.V, list(INVALID_V)), np.nan, self.stats.V)
        gens = np.arange(V.shape[0])
        best = np.nanmax(V, axis=1)
        mean = np.nanmean(V, axis=1)
        std  = np.nanstd(V, axis=1)
        plt.figure()
        plt.plot(gens, best, label="best vel ↑", linewidth=2.0)
        plt.plot(gens, mean, "--", label="mean vel", linewidth=1.6)
        plt.fill_between(gens, mean - std, mean + std, alpha=0.18, label="mean ± std")
        plt.xlabel("generation"); plt.ylabel("velocity")
        plt.grid(True, alpha=0.25); plt.legend()
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        print("✓ progress plot →", out)

    def fronts_progress_P(self, out="best_prog_per_gen.png"):
        # Uses CSV (works even if stats.M isn't filled for progress)
        if "prog_P" not in self.df.columns:
            print("No prog_P column – skip progress plot"); return
        grp = self.df.groupby("generation")["prog_P"]
        best_prog = grp.max()
        mean_prog = grp.mean()
        std_prog  = grp.std(ddof=0).fillna(0.0)
        min_p = self.df.groupby("generation")["minimal_p"].first() if "minimal_p" in self.df.columns else None
        plt.figure()
        plt.plot(best_prog.index, best_prog.values, label="best progress ↑")
        plt.plot(mean_prog.index, mean_prog.values, "--", label="mean progress")
        plt.fill_between(mean_prog.index,
                         (mean_prog - std_prog).values,
                         (mean_prog + std_prog).values,
                         alpha=0.15, label="mean ± std")
        if min_p is not None:
            plt.plot(min_p.index, min_p.values, "--", label="minimal p threshold")
        plt.xlabel("generation"); plt.ylabel("meters"); plt.legend()
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        print("✓ progress plot →", out)

    def fronts_progress_E(self, out="best_eff_per_gen.png"):
        if self.stats is None:
            print("No stats – skip progress plot"); return
        E = np.where(np.isin(self.stats.E, list(INVALID_E)), np.nan, self.stats.E)
        gens = np.arange(E.shape[0])
        best = np.nanmax(E, axis=1)     # best -energy (greater is better)
        mean = np.nanmean(E, axis=1)
        std  = np.nanstd(E, axis=1)
        plt.figure()
        plt.plot(gens, best, label="best -energy ↑", linewidth=2.0)
        plt.plot(gens, mean, "--", label="mean -energy", linewidth=1.6)
        plt.fill_between(gens, mean - std, mean + std, alpha=0.18, label="mean ± std")
        plt.xlabel("generation"); plt.ylabel("-energy (efficiency)")
        plt.grid(True, alpha=0.25); plt.legend()
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        print("✓ progress plot →", out)

    def reward_and_speed(self, out="reward_speed.png"):
        """
        Plot per-generation MEAN ± STD of:
        • Final reward (rew_100pct) on the left y-axis
        • % steps to reach 90% of the final reward on the right y-axis

        Notes:
        - No moving average across generations.
        - Statistics are computed across all individuals within each generation.
        - Requires TensorBoard-derived columns: rew_10pct ... rew_100pct.
        """
        # Ensure required columns exist
        cols = [f"rew_{k:02d}pct" for k in range(5, 101, 5)]
        if not all(c in self.df.columns for c in cols):
            print("No rew_xxpct columns – skip reward/speed plot")
            return

        # Row-wise metrics
        df = self.df.copy()
        df["final_reward"] = df["rew_100pct"]

        # Earliest % where reward >= 90% of final reward (per row)
        def _steps90(row):
            tgt = 0.9 * row["final_reward"]
            for pct in range(5, 101, 5):
                val = row.get(f"rew_{pct}pct", np.nan)
                if np.isfinite(val) and val >= tgt:
                    return pct
            return np.nan

        df["steps90_pct"] = df.apply(_steps90, axis=1)

        # Keep only finite rows needed for grouping
        if "generation" not in df.columns:
            print("Missing 'generation' column – skip reward/speed plot")
            return
        df = df[np.isfinite(df["final_reward"]) & np.isfinite(df["generation"])]

        if df.empty:
            print("No valid rows – skip reward/speed plot")
            return

        # Per-generation aggregation (mean and std), NaN-safe
        gb = df.groupby("generation", sort=True)
        fr_mean = gb["final_reward"].mean()                 # mean final reward per gen
        fr_std  = gb["final_reward"].std(ddof=0).fillna(0)  # population std; fill NaNs to 0
        s90_mean = gb["steps90_pct"].mean()
        s90_std  = gb["steps90_pct"].std(ddof=0).fillna(0)

        if fr_mean.empty:
            print("No per-generation stats – skip reward/speed plot")
            return

        # Convert to numpy arrays for plotting
        gens   = fr_mean.index.to_numpy(dtype=float)
        fr_m   = fr_mean.to_numpy(dtype=float)
        fr_sd  = fr_std.to_numpy(dtype=float)
        s90_m  = s90_mean.to_numpy(dtype=float)
        s90_sd = s90_std.to_numpy(dtype=float)

        # Colors / styles
        cmap = plt.get_cmap("tab10")
        c_fr_line  = cmap(0)  # final reward
        c_fr_band  = cmap(0)
        c_s90_line = cmap(1)  # steps to 90%
        c_s90_band = cmap(1)

        # Figure and twin axes
        fig, ax1 = plt.subplots(figsize=(8.6, 4.8))
        ax2 = ax1.twinx()

        # Final reward: mean ± std (left axis)
        ax1.plot(gens, fr_m, label="Final Reward (mean)", linewidth=2.0,
                marker="o", ms=3.5, color=c_fr_line)
        ax1.fill_between(gens, fr_m - fr_sd, fr_m + fr_sd,
                        alpha=0.15, color=c_fr_band, label="Final Reward ±1σ")

        # Steps to 90%: mean ± std (right axis)
        ax2.plot(gens, s90_m, label="Steps to 90% (mean)", linewidth=2.0,
                linestyle="-", marker="s", ms=3.0, color=c_s90_line)
        ax2.fill_between(gens, s90_m - s90_sd, s90_m + s90_sd,
                        alpha=0.10, color=c_s90_band, label="Steps to 90% ±1σ")

        # Labels and appearance
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Final Reward (mean)", color=c_fr_line)
        ax2.set_ylabel("Steps to 90% Final Reward [%] (mean)", color=c_s90_line)

        ax1.grid(True, which="major", alpha=0.25)
        ax1.minorticks_on()
        ax1.grid(True, which="minor", alpha=0.10)

        ax1.tick_params(axis="y", colors=c_fr_line)
        ax2.tick_params(axis="y", colors=c_s90_line)

        # Unified legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)

        plt.title("Final Reward & Learning Speed per Generation (mean ± 1σ)")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print("✓ reward/speed plot →", out)


    def pairwise_scatter_fitness(self, out="pairwise_fitness_scatter.png"):
        """
        Three subplots with pairwise scatter of fitness metrics:
        1) Progress vs Velocity
        2) Progress vs Efficiency (-Energy)
        3) Velocity vs Efficiency

        Each point is an individual; colormap encodes the generation.
        Requirements:
        • Colorbar placed on the far right of the figure (outside the subplot grid).
        • Plot points ONLY for generations where minimal_p == fixed minimal p (300).
        • Points slightly larger than before.

        Note: sentinel values are already converted to NaN in __init__.
        """
        # Required columns in the analysis CSV
        needed = ["vel_v", "eff_E", "prog_P", "generation", "minimal_p"]
        if not all(c in self.df.columns for c in needed):
            print("Missing columns for pairwise scatter – skip")
            return

        # Filter to valid rows and only where minimal_p equals the fixed threshold (300)
        FIXED_MIN_P = 300.0  # must match CodesignDEAP.fixed_p default
        df = self.df.dropna(subset=needed).copy()
        df = df[np.isfinite(df["minimal_p"]) & np.isclose(df["minimal_p"], FIXED_MIN_P)]

        if df.empty:
            print("No valid rows for pairwise scatter (minimal_p != 300 or NaNs) – skip")
            return

        # Colormap over generations
        gens = df["generation"].astype(float).to_numpy()
        vmin = float(np.nanmin(gens))
        vmax = float(np.nanmax(gens))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("viridis")

        # Create figure and axes
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Slightly larger marker size
        point_size = 20

        # 1) Progress vs Velocity
        axes[0].scatter(
            df["prog_P"], df["vel_v"],
            c=gens, cmap=cmap, norm=norm,
            s=point_size, alpha=0.85, edgecolors="none"
        )
        axes[0].set_xlabel("Progress (m)")
        axes[0].set_ylabel("Velocity")
        axes[0].set_title("Progress vs Velocity")
        axes[0].grid(True, alpha=0.25)

        # 2) Progress vs Efficiency (-Energy)
        axes[1].scatter(
            df["prog_P"], df["eff_E"],
            c=gens, cmap=cmap, norm=norm,
            s=point_size, alpha=0.85, edgecolors="none"
        )
        axes[1].set_xlabel("Progress (m)")
        axes[1].set_ylabel("-Energy (↑ efficiency)")
        axes[1].set_title("Progress vs Efficiency")
        axes[1].grid(True, alpha=0.25)

        # 3) Velocity vs Efficiency
        axes[2].scatter(
            df["vel_v"], df["eff_E"],
            c=gens, cmap=cmap, norm=norm,
            s=point_size, alpha=0.85, edgecolors="none"
        )
        axes[2].set_xlabel("Velocity")
        axes[2].set_ylabel("-Energy (↑ efficiency)")
        axes[2].set_title("Velocity vs Efficiency")
        axes[2].grid(True, alpha=0.25)

        # Colorbar on the FAR RIGHT (outside subplot grid)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        # Reserve space on the right for the vertical colorbar
        plt.subplots_adjust(right=0.90)
        # [left, bottom, width, height] in figure coordinates (0..1)
        cax = fig.add_axes([0.915, 0.15, 0.02, 0.70])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Generation")

        # Compact layout after reserving right margin
        fig.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])

        plt.savefig(out, dpi=150)
        plt.close()
        print("✓ pairwise fitness scatter →", out)


    @staticmethod
    def plot_tb_reward_evolution(exp: str, out_png: Path, run_id: str | None = None, win_frac: float = 0.10) -> bool:
        """
        NUOVO: genera un plot dell'evoluzione di 'Train/mean_reward' per l'esperimento
        `exp` e lo salva in out_png. Ritorna True se i logs TB sono caricabili.
        """
        log_dir = _LOG_ROOT_ENV / "ea" / exp
        ea = event_accumulator.EventAccumulator(str(log_dir))
        try:
            ea.Reload()
        except Exception:
            return False
        key = "Train/mean_reward"
        if key not in ea.Tags().get("scalars", []):
            return False
        events = ea.Scalars(key)
        if not events:
            return False
        steps = np.asarray([e.step for e in events], dtype=float)
        vals  = np.asarray([e.value for e in events], dtype=float)
        # Moving average centrata + deviazione standard in dominio step
        win = max(11, int(round(len(vals) * win_frac))) | 1
        half = win // 2
        ma = np.empty_like(vals, dtype=float)
        sd = np.empty_like(vals, dtype=float)
        for i in range(len(vals)):
            lo = max(0, i - half); hi = min(len(vals), i + half + 1)
            seg = vals[lo:hi]
            ma[i] = float(np.mean(seg)); sd[i] = float(np.std(seg, ddof=0))
        plt.figure()
        plt.plot(steps, ma, label="moving avg reward", linewidth=2.0)
        plt.fill_between(steps, ma - sd, ma + sd, alpha=0.15, label="±1σ (moving)")
        plt.xlabel("training step"); plt.ylabel("mean reward")
        title = f"{run_id} - Reward evolution" if run_id else "Reward evolution"
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=150)
        plt.close()
        return True

    def analyze(self, prefix="analysis"):
        self.fronts_progress_V(f"{prefix}_progress_vel.png")
        self.fronts_progress_P(f"{prefix}_progress_prog.png")
        self.fronts_progress_E(f"{prefix}_progress_eff.png")
        self.reward_and_speed(f"{prefix}_reward_speed.png")
        self.pairwise_scatter_fitness(f"{prefix}_paretos.png")


# ╭─────────────────────────────────────────────────────────────╮
# │  Helpers – chromosome & DB                                  │
# ╰─────────────────────────────────────────────────────────────╯
class Chromosome_Drone:
    """Encodes the discrete design spaces for each gene."""

    @staticmethod
    def lspace(start: float, stop: float, delta: float) -> list[float]:
        """
        Inclusive linspace with fixed step `delta` (can be negative).
        Rounds values based on the number of decimals in `delta`
        to avoid floating artifacts in CSVs.
        """
        assert delta != 0.0, "delta must be non-zero"
        # Compute number of points (inclusive), robust to FP noise
        steps = abs((stop - start) / delta)
        n = int(round(steps)) + 1
        arr = np.linspace(start, stop, n)
        # Infer decimals from delta (e.g., 0.03 -> 2, 0.5 -> 1)
        from decimal import Decimal
        try:
            decimals = max(0, -Decimal(str(abs(delta))).normalize().as_tuple().exponent)
        except Exception:
            decimals = 6
        arr = np.round(arr, decimals)
        return arr.tolist()

    _CHOICES = [
        # 0: wing_span (m) → 0.40..0.60 step 0.02
        lspace.__func__(0.40, 0.70, 0.02),
        # 1: wing_aspect_ratio (span/chord) → 1.5..5.0 step 0.5
        lspace.__func__(1.5, 5.0, 0.25),
        # 2: fus_len
        lspace.__func__(0.40, 0.70, 0.02),
        # 3: cg_x_ratio (fus_cg_x / fus_length) → 0.60..0.30 step -0.05
        lspace.__func__(0.30, 0.70, 0.05),
        # 4: attach_x_ratio (wing_attach_x / fus_length) → 0.30..0.70 step 0.05
        lspace.__func__(0.30, 0.70, 0.05),
        # 5: elevator_span (m) → 0.10..0.30 step 0.02
        lspace.__func__(0.10, 0.30, 0.02),
        # 6: elevator_aspect_ratio → 1.0..3.0 step 0.5
        lspace.__func__(1.0, 3.0, 0.25),
        # 7: rudder_span (m) → 0.10..0.20 step 0.02
        lspace.__func__(0.10, 0.20, 0.02),
        # 8: rudder_aspect_ratio → 1.0..3.0 step 0.5
        lspace.__func__(1.0, 3.0, 0.25),
        # 9: dihedral_deg (degrees) → -20..20 step 5  (no duplicates)
        lspace.__func__(-20.0, 20.0, 5.0),
        # 10: hinge_le_ratio (fraction of chord) → 0.18..0.36 step 0.03
        [0.25],
        # 11: sweep_multiplier → 1.5..3.5 step 0.5
        lspace.__func__(1.0, 4.0, 0.5),
        # 12: twist_multiplier → 1.5..3.5 step 0.5
        lspace.__func__(1.0, 4.0, 0.5),
        # 13: cl_alpha_2d → single fixed value
        [2.0],
        # 14: alpha0_2d (degrees) → 0.0..-5.0 step -0.5
        lspace.__func__(0.0, -5.0, -0.5),
    ]


    def random(self):
        """Pick a random value *within the list* of each gene."""
        return [random.choice(options) for options in self._CHOICES]

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
        if not self.path.exists():
            self.df.to_csv(self.path, index=False)

    def lookup(self, chromo):
        # Used by older code paths; kept for compatibility
        row = self.df[self.df.chromosome == str(list(chromo))]
        if row.empty:
            return None
        return [row[f"ff_{i}"].min() for i in range(self.n_obj)]

    def insert(self, chromo, ff, meta):
        """Append a new row, then materialize a numbered run directory with URDF & logs."""
        lock = FileLock(str(self.path) + ".lock")
        with lock:
            # Re-read to compute the next 1-based row_id safely
            if self.path.exists():
                self.df = pd.read_csv(self.path)
            row_id = int(self.df.shape[0]) + 1

            print(f"   ↘ writing CSV  row_id={row_id}  gen={meta.get('generation')}  ff={ff}")
            row = self._blank().iloc[0]
            row.row_id = row_id
            row.timestamp = time.time(); row.chromosome = str(list(chromo))
            for i, v in enumerate(ff): row[f"ff_{i}"] = v
            for k, v in meta.items():   row[k] = v

            self.df = pd.concat([self.df, row.to_frame().T], ignore_index=True)
            self.df.to_csv(self.path, index=False)

            # --- per-run artifacts -------------------------------------------------
            run_dir = RUNS_BASE / f"{row_id:05d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            exp = str(row.get("exp_name", "exp"))
            # copy URDF
            urdf_src = URDF_BASE / f"{exp}.urdf"
            if urdf_src.exists():
                try:
                    shutil.copy2(urdf_src, run_dir / urdf_src.name)
                except Exception as e:
                    print(f"[runs] URDF copy failed: {e}")
            # symlink logs directory
            log_dir = _LOG_ROOT_ENV / "ea" / exp
            link = run_dir / "logs"
            try:
                if log_dir.exists():
                    if link.exists() or link.is_symlink():
                        link.unlink()
                    link.symlink_to(log_dir, target_is_directory=True)
            except Exception as e:
                print(f"[runs] log symlink failed: {e}")

            # NUOVO: plot evoluzione reward TB nella 0000x
            reward_png = run_dir / "reward_evolution.png"
            ok = PostAnalyzer.plot_tb_reward_evolution(exp, reward_png, run_id=f"{row_id:05d}")
            if not ok:
                # se TB non carica → togli i logs dalla 0000x
                if link.exists() or link.is_symlink():
                    try:
                        link.unlink()
                        print(f"[runs] logs not loadable → removed symlink in {run_dir.name}")
                    except Exception:
                        pass

            try:
                src_total = log_dir / "total_plot.png"
                dst_total = run_dir / "total_plot.png"
                if src_total.is_file():
                    shutil.copy2(src_total, dst_total)
                    print(f"✓ copied total_plot → {dst_total}")
            except Exception as e:
                print(f"[runs] total_plot copy failed: {e}")

            return row_id, run_dir

    def _blank(self):
        cols = [
            "row_id", "timestamp", "chromosome",
            *[f"ff_{i}" for i in range(self.n_obj)],
            "generation", "exp_name", "train_it", "max_p",
            "minimal_p",
            "parent_idx_a","parent_idx_b",
            "vel_v","vel_E","vel_P",
            "eff_v","eff_E","eff_P",
            "prog_v","prog_E","prog_P",
            # reward curve (10%..100%)
            *[f"rew_{k:02d}pct" for k in range(5, 101, 5)],
        ]
        return pd.DataFrame([{c: np.nan for c in cols}])

    def get_row(self, chromo):
        """Return the pandas Series for the chromosome, or None if not present."""
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
    TRAIN_ITERS_INHERIT   = 500
    TRAIN_ENVS            = 16384
    EVAL_ENVS             = 4096
    VMIN, VMAX            = 6.0, 24.0
    WEIGHTS               = (+1.0, +1.0, +1.0)  # maximize vel, -energy

    def __init__(self, n_pop=12, n_gen=20, cx_pb=0.85, mut_pb=0.3,
                 csv=None, inherit_policy=False,
                 use_dynamic_p=True, fixed_p=300.0, pct_above=50.0):
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
                ind[i] = random.choice([v for v in options if v != ind[i]])
        return (ind,)

    # ─────── evaluate (serial or remote) ─────────────────────────────────
    def _evaluate(self, indiv):
        chromo = list(indiv)
        print(f"[evaluate] gen={getattr(self,'_gen',0)}  chr={chromo}")
        cached_row = self.db.get_row(chromo)  # Series with existing meta
        if cached_row is not None:
            # CACHE-HIT → re-train from scratch (NO inheritance)
            parent_info = (None, None)
            print("   ↪ cache-hit: re-training from scratch (NO inheritance)")
        else:
            # NEW → train from scratch (or from selected parents if inheritance is on)
            parent_info = (getattr(indiv, "parent_exp", None), getattr(indiv, "parent_ckpt", None))
            print(f"   ↪ NEW chromosome → training for {self.TRAIN_ITERS} iters (or {self.TRAIN_ITERS_INHERIT} if inherited)")

        cfg = dict(
            TRAIN_ITERS=self.TRAIN_ITERS,
            TRAIN_ITERS_INHERIT=self.TRAIN_ITERS_INHERIT,
            TRAIN_ENVS=self.TRAIN_ENVS,
            EVAL_ENVS=self.EVAL_ENVS,
            VMIN=self.VMIN, VMAX=self.VMAX,
        )

        if USE_PARALLEL:
            print("[train+eval]  Ray submit", chromo)
            fut = _dispatch_train_eval(chromo, parent_info, self.tag, cfg)
            indiv._pending_future = fut
            return (0.0, 0.0, 0.0)  # placeholder until Ray completes
        else:
            ff, meta, extra = _dispatch_train_eval(chromo, parent_info, self.tag, cfg)

        print(f"   ✔ sync‑train+eval  ff={ff}  max_p={meta['max_p']:.2f}")
        indiv._meta_raw = meta
        indiv.max_p     = meta["max_p"]
        if extra:
            indiv._p_s = extra["p_s"];  indiv._v_s = extra["v_s"];  indiv._E_s = extra["E_s"]
        return tuple(ff)

    def _pick_triples(self, p_s, v_s, E_s, minimal_p):
        # Robustness: handle empty/None arrays → return sentinels
        if p_s is None or v_s is None or E_s is None:
            vel = dict(mean_v=0.0, mean_E=100.0, mean_progress=0.0)
            eff = dict(mean_v=0.0, mean_E=100.0, mean_progress=0.0)
            prog = dict(mean_v=0.0, mean_E=100.0, mean_progress=0.0)
            return vel, eff, prog
        p_s = np.asarray(p_s); v_s = np.asarray(v_s); E_s = np.asarray(E_s)
        if p_s.size == 0 or v_s.size == 0 or E_s.size == 0:
            vel = dict(mean_v=0.0, mean_E=100.0, mean_progress=0.0)
            eff = dict(mean_v=0.0, mean_E=100.0, mean_progress=0.0)
            prog = dict(mean_v=0.0, mean_E=100.0, mean_progress=0.0)
            return vel, eff, prog
        idx_p = int(np.nanargmax(p_s))  # max progress (unfiltered)
        mask  = np.where(p_s >= minimal_p)[0]
        if mask.size == 0:
            # sentinels
            vel = dict(mean_v=0.0, mean_E=100.0, mean_progress=0.0)
            eff = dict(mean_v=0.0, mean_E=100.0, mean_progress=0.0)
        else:
            idx_v = mask[np.argmax(v_s[mask])]  # max vel in valid region
            idx_e = mask[np.argmin(E_s[mask])]  # min energy in valid region
            vel = dict(mean_v=v_s[idx_v], mean_E=E_s[idx_v], mean_progress=p_s[idx_v])
            eff = dict(mean_v=v_s[idx_e], mean_E=E_s[idx_e], mean_progress=p_s[idx_e])
        prog = dict(mean_v=v_s[idx_p], mean_E=E_s[idx_p], mean_progress=p_s[idx_p])
        return vel, eff, prog

    def run(self):
        # ─── GEN-0 ───
        pop = self.tb.pop(self.n_pop)
        self._gen = 0
        self._train_eval_population(pop)  # train + eval
        # finalize + persist with shared minimal_p for gen-0
        m0 = self._compute_minimal_p(pop)
        for ind in pop:
            self._finalize_one(ind, m0, persist=True)
        pop = tools.selNSGA2(pop, self.n_pop)  # also assigns crowding distance
        self._after_generation(pop)

        # ─── GEN ≥ 1 ───
        for g in range(1, self.n_gen + 1):
            self._gen = g
            print(f"\n════════ Generation {g}/{self.n_gen} ════════")

            # 1) parent-selection
            # Tag each parent with its index in the current population (stable within generation)
            for _idx, _ind in enumerate(pop):
                _ind._pop_idx = _idx
            parents   = tools.selTournamentDCD(pop, len(pop))
            offspring = [self.tb.clone(p) for p in parents]

            # 2) variation (+ optional inheritance) before training
            self._apply_variation(offspring, parents)

            # 3) train + eval offspring
            self._train_eval_population(offspring)

            # 4) minimal_p on all (parents + offspring)
            minimal_p = self._compute_minimal_p(pop + offspring)
            peaks_all = [getattr(i, "max_p", np.nan) for i in (pop + offspring)]
            peaks_all = [p for p in peaks_all if not np.isnan(p)]
            if not peaks_all:
                print(f"[Gen {g}] minimal_p = {minimal_p:.2f} | no peaks available (N=0)")
            else:
                q25, q50, q75 = (np.percentile(peaks_all, q) for q in (25, 50, 75))
                print(f"[Gen {g}] minimal_p = {minimal_p:.2f} | max_p stats: "
                      f"min={min(peaks_all):.1f} q25={q25:.1f} med={q50:.1f} q75={q75:.1f} max={max(peaks_all):.1f}  "
                      f"(N={len(peaks_all)})")

            # 5) finalize (offspring persisted; parents re-evaluated only)
            for ind in offspring:
                self._finalize_one(ind, minimal_p, persist=True)
            for ind in pop:
                self._finalize_one(ind, minimal_p, persist=False)

            def _is_feasible(ind):
                return hasattr(ind, "_p_s") and not (
                    np.isclose(ind.fitness.values[0], 0.0) and
                    np.isclose(ind.fitness.values[1], next(iter(INVALID_E)))
                )
            f_par = sum(_is_feasible(i) for i in pop)
            f_off = sum(_is_feasible(i) for i in offspring)
            print(f"[Gen {g}] feasible: parents {f_par}/{len(pop)} | offspring {f_off}/{len(offspring)}")

            # 6) NSGA-II survivor selection → new population
            pop = tools.selNSGA2(pop + offspring, self.n_pop)

            # 7) logging / plots
            self._after_generation(pop)

        # Save global stats
        with open(f"stats_{self.tag}.pkl", "wb") as f:
            import pickle; pickle.dump(self.stats, f)
        print("Statistics saved ✔")
        return pop

    def _train_eval_population(self, population):
        """
        Stage-1: train/eval all individuals without fitness.
        Fills: ind._meta_raw, ind.max_p, ind._p_s/_v_s/_E_s (if available).
        Does NOT compute minimal_p, does NOT persist, does NOT fix final fitness.
        """
        for ind in population:
            if not ind.fitness.valid:
                ind.fitness.values = self.tb.evaluate(ind)

        if USE_PARALLEL:
            pend = [ind for ind in population if hasattr(ind, "_pending_future")]
            if pend:
                print(f"  ⏳ waiting {len(pend)} Ray jobs…")
                import ray
                results = ray.get([ind._pending_future for ind in pend])
                for ind, (ff, meta, extra) in zip(pend, results):
                    ind._meta_raw = meta
                    ind.max_p     = meta["max_p"]
                    if extra:
                        ind._p_s = extra["p_s"]; ind._v_s = extra["v_s"]; ind._E_s = extra["E_s"]
                    # placeholder (not used for selection yet)
                    ind.fitness.values = tuple(ff)
                    del ind._pending_future
                    print(f"   ✅ Ray done chr={list(ind)} ff={ff} max_p={meta['max_p']:.2f}")

    def _compute_minimal_p(self, inds):
        peaks = [getattr(i, "max_p", np.nan) for i in inds]
        peaks = [p for p in peaks if not np.isnan(p)]
        if not (self.use_dynamic_p and peaks):
            return self.fixed_p
        m = np.percentile(peaks, 100.0 - self.pct_above)
        return self.fixed_p if m > self.fixed_p else m

    def _finalize_one(self, ind, minimal_p, persist=False):
        """Convert arrays → filtered triples → final fitness; optionally persist to CSV."""
        # Guard: missing or empty arrays → mark as failed and skip persistence
        if (not hasattr(ind, "_p_s") or not hasattr(ind, "_v_s") or not hasattr(ind, "_E_s")
            or ind._p_s is None or ind._v_s is None or ind._E_s is None
            or len(np.atleast_1d(ind._p_s)) == 0
            or len(np.atleast_1d(ind._v_s)) == 0
            or len(np.atleast_1d(ind._E_s)) == 0):
            ind.fitness.values = (0.0, next(iter(INVALID_E)), 0.0)
            try:
                print(f"[finalize g={self._gen}] FAIL (empty arrays) thr={minimal_p:.1f}")
            except Exception:
                pass
            return
        vel_d, eff_d, prog_d = self._pick_triples(ind._p_s, ind._v_s, ind._E_s, minimal_p)
        ff_final = (vel_d["mean_v"], -eff_d["mean_E"], prog_d["mean_progress"])  # (+vel, +(-E), +prog)
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
                parent_idx_a=int(getattr(ind, "parent_idx_a", -1)),
                parent_idx_b=int(getattr(ind, "parent_idx_b", -1)),
            ))
            _row_id, _run_dir = self.db.insert(list(ind), ff_final, meta)
            # Keep row_id on the individual so future generations can reference it if needed
            try:
                ind.row_id = int(_row_id)
            except Exception:
                pass
            ind._persisted = True
        feasible = not (vel_d["mean_v"] == 0.0 and eff_d["mean_E"] == 100.0)
        status = "OK " if feasible else "FAIL"
        print(f"[finalize g={self._gen}] {status} "
              f"p*={getattr(ind,'max_p',float('nan')):.1f}  thr={minimal_p:.1f}  "
              f"vel(v={vel_d['mean_v']:.2f}, E={vel_d['mean_E']:.1f})  "
              f"eff(v={eff_d['mean_v']:.2f}, E={eff_d['mean_E']:.1f})  "
              f"ff=({ind.fitness.values[0]:.2f}, {ind.fitness.values[1]:.2f}, {ind.fitness.values[2]:.2f})")

    def _apply_variation(self, offspring, parents):
        """Crossover, mutation and optional inheritance **before** training."""
        # cleanup custom attributes
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
            # mutation
            if random.random() < self.mut_pb:
                self.tb.mutate(c1);  del c1.fitness.values
            if random.random() < self.mut_pb:
                self.tb.mutate(c2);  del c2.fitness.values
            # policy inheritance: pick a parent's checkpoint if available
            if self.inherit_policy:
                infos = []
                for p in (parents[i], parents[i+1]):
                    if hasattr(p, "exp_name"):
                        ck_it = getattr(p, "train_it", self.TRAIN_ITERS)
                        LOG_ROOT = _LOG_ROOT_ENV
                        CK_DIRS = [LOG_ROOT / "ea" / p.exp_name, LOG_ROOT / p.exp_name]
                        ck = None
                        for d in CK_DIRS:
                            cand = d / f"model_{ck_it}.pt"
                            if cand.is_file():
                                ck = cand; break
                        if ck is not None and ck.is_file():
                            infos.append((p.exp_name, ck_it))
                if infos:
                    c1.parent_exp, c1.parent_ckpt = random.choice(infos)
                    c2.parent_exp, c2.parent_ckpt = random.choice(infos)
            # Record parent indices for lineage tracking in CSV
            pA = parents[i]
            pB = parents[i+1]
            idxA = int(getattr(pA, "_pop_idx", -1))
            idxB = int(getattr(pB, "_pop_idx", -1))
            # Store indices on both children
            c1.parent_idx_a, c1.parent_idx_b = idxA, idxB
            c2.parent_idx_a, c2.parent_idx_b = idxA, idxB


    def _after_generation(self, pop):
        """Update stats, create plots, and print a generation summary."""
        g = self._gen
        self.stats.record(g, pop)
        if g % 2 == 0 or g == self.n_gen:
            out_dir = Path(f"g{g:02d}"); out_dir.mkdir(exist_ok=True)
            PostAnalyzer(self.db.path, self.stats).analyze(prefix=f"{out_dir}/")
        best_v = np.nanmax(self.stats.V[g])
        best_e = -np.nanmin(self.stats.E[g])
        gen_df = self.db.df[self.db.df.generation == g]
        best_p = gen_df['prog_P'].max() if not gen_df.empty else float('nan')
        print(f"--- Gen {g} summary  best_vel={best_v:.2f} best_eff={best_e:.2f} best_prog={best_p:.2f}")


# ╭─────────────────────────────────────────────────────────────╮
# │  CLI                                                        │
# ╰─────────────────────────────────────────────────────────────╯
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--gen", type=int, default=10)
    ap.add_argument("--train_it", type=int, default=1000)
    ap.add_argument("--inherit", action="store_true", default=False,
                    help="enable policy inheritance from selected parents")
    args = ap.parse_args()

    cd = CodesignDEAP(n_pop=args.pop, n_gen=args.gen, inherit_policy=args.inherit)
    cd.TRAIN_ITERS = args.train_it
    cd.run()
