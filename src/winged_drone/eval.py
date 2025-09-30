#!/usr/bin/env python3
# eval.py — evaluation utilities used by evolution.py
# ---------------------------------------------------------------------------
# What this file guarantees (aligned with evolution.py):
#  • Every evaluation call generates a total_plot and saves it in
#    LOG_ROOT/ea/<exp_name>/total_plot.png. If the numbered run directory
#    ANALYSIS_DIR/runs/<row_id>/ can be resolved from the CSV, a copy is also
#    saved there as ANALYSIS_DIR/runs/<row_id>/total_plot.png.
#  • Moving-average smoothing is performed *with respect to the commanded
#    velocity* (v_cmd). We sort by v_cmd and use a sliding window in that
#    coordinate; all smoothed quantities (progress, mean velocity, energy)
#    are computed on that v_cmd axis.
#  • Return signature is compatible with evolution.py: evaluation(...) returns
#    (top_vel, top_eff, top_prog, final_reason, extra) when return_arrays=True;
#    extra contains arrays p_s, v_s, E_s and max_p.
#  • Comments are in English.
# ---------------------------------------------------------------------------

import os, pickle, math, copy
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

# ActorCritic class gets injected as in the original code
from actor_critic_modified import ActorCriticTanh
import builtins
builtins.ActorCriticTanh = ActorCriticTanh

import genesis as gs
from winged_drone_env import WingedDroneEnv
from rsl_rl.runners import OnPolicyRunner

SUCCESS_TIME = 200.0  # minimum time [s] to mark an env as "success"

# ──────────────────────────────────────────────────────────────
#  1) EVAL ENVS WITH FIXED (LINEAR) COMMANDED SPEEDS
# ──────────────────────────────────────────────────────────────
class WingedDroneEvalLinSpeed(WingedDroneEnv):
    """Evaluation env that *locks* commanded speed to a precomputed linspace.
    This guarantees that any reset keeps the same v_cmd per env index.
    """
    def __init__(self, num_envs, v_min, v_max, *args, **kwargs):
        self._v_cmd = torch.linspace(v_min, v_max, num_envs)
        super().__init__(num_envs, *args, **kwargs)

    def _resample_commands(self, envs_idx):
        # yaw target = 0, altitude = 10 m, commanded speed = fixed grid
        self.commands[envs_idx, 0] = 0.0
        self.commands[envs_idx, 1] = 10.0
        self.commands[envs_idx, 2] = self._v_cmd[envs_idx]


class WingedDroneEvalFixedSpeed(WingedDroneEnv):
    """Evaluation env that accepts an arbitrary list of commanded speeds."""
    def __init__(self, v_list, *args, **kwargs):
        super().__init__(len(v_list), *args, **kwargs)
        self._v_cmd = torch.as_tensor(v_list, device=self.device, dtype=torch.float32)

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = 0.0
        self.commands[envs_idx, 1] = 10.0
        self.commands[envs_idx, 2] = self._v_cmd[envs_idx]


# ──────────────────────────────────────────────────────────────
#  2) ROLL-OUT (PER-ENV STATS + OPTIONAL TRACES)
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def run_eval(env: WingedDroneEnv, policy, watch_env_idxs: Iterable[int]=()) -> Tuple[np.ndarray, ...]:
    B, dt, dev = env.num_envs, env.dt, env.device
    watch_env_idxs = list(watch_env_idxs)

    done = torch.zeros(B, dtype=torch.bool, device=dev)
    t_acc  = torch.zeros(B, device=dev)
    dx_acc = torch.zeros(B, device=dev)
    E_acc  = torch.zeros(B, device=dev)
    final_reason = torch.full((B,), 3, dtype=torch.int8, device=dev)

    # Detailed traces only for watched envs
    traces = {i: {"s": [], "j_pos": [], "thr": [], "E": [], "alpha": [], "beta": []}
              for i in watch_env_idxs}
    E_int = torch.zeros(len(watch_env_idxs), device=dev)

    obs, _ = env.reset()
    x0 = env.base_pos[:, 0].clone()

    print(f"Running evaluation on {B} envs – watch={watch_env_idxs}")

    while not done.all():
        act = policy(obs)
        obs, _, term, _ = env.step(act)
        term = term.bool()
        nan_indices = env.nan_envs.to(torch.bool)  # envs with NaN in observations

        alive = (~done) & ~term & ~nan_indices
        t_acc[alive] += dt
        dx_acc[alive] = env.base_pos[alive, 0] - x0[alive]
        P = env.power_consumption()  # instantaneous power [W]
        E_acc[alive] += P[alive] * dt  # integrate power → energy [J]

        # Detailed logging only for watched envs
        for k, idx in enumerate(watch_env_idxs):
            if done[idx] or term[idx] or nan_indices[idx]:
                continue
            s = (env.base_pos[idx, 0] - x0[idx]).item()
            tr = traces[idx]
            tr["s"].append(s)
            tr["j_pos"].append(env.joint_position[idx].detach().cpu().numpy())
            # throttle/actuator trace (best-effort; attribute name may differ)
            thr_val = None
            for name in ("throttle", "thrust"):
                if hasattr(env, name):
                    try:
                        val = getattr(env, name)[idx]
                        thr_val = float(val.detach().cpu().mean().item()) if torch.is_tensor(val) else float(val)
                        break
                    except Exception:
                        pass
            if thr_val is not None:
                tr["thr"].append(thr_val)
            E_int[k] += P[idx] * dt
            tr["E"].append(E_int[k].item())
            for name in ("alpha", "beta"):
                if hasattr(env, name):
                    tr[name].append(float(getattr(env, name)[idx]))

        # finalize reasons for just-finished envs
        just_done = (~done) & term
        for j in just_done.nonzero(as_tuple=False).flatten().tolist():
            if t_acc[j] > SUCCESS_TIME - 0.1 or getattr(env, "pre_success", torch.zeros_like(t_acc))[j]:
                final_reason[j] = 0
            elif getattr(env, "pre_collision", torch.zeros_like(t_acc))[j]:
                final_reason[j] = 1
            elif getattr(env, "pre_crash_condition", torch.zeros_like(t_acc))[j]:
                final_reason[j] = 2
            else:
                final_reason[j] = 3
        done |= term | nan_indices

    # Aggregate per-env metrics
    safe = (~nan_indices) & (dx_acc>1)
    v_mean = (dx_acc[safe] / t_acc[safe].clamp_min(1e-6)).detach().cpu().numpy()  # mean vel along x
    E_tot  = (E_acc[safe] / dx_acc[safe].clamp_min(1e-6)).detach().cpu().numpy()  # J/m (cost of transport)
    v_cmd  = env.commands[safe, 2].detach().cpu().numpy()                         # commanded speed
    progress = dx_acc[safe].detach().cpu().numpy()                                # meters progressed
    final_reason = final_reason[safe].detach().cpu().numpy().astype(np.int8)

    return v_mean, E_tot, v_cmd, progress, final_reason


# ──────────────────────────────────────────────────────────────
#  3) MOVING AVERAGE ON THE v_cmd AXIS
# ──────────────────────────────────────────────────────────────
def _moving_avg_vs_vcmd(x_vcmd: np.ndarray, y: np.ndarray, win_frac: float = 0.03):
    """Return (x_sorted, y_smooth, y_std) using a centered sliding window
    *in the commanded-velocity domain*.

    We sort by v_cmd (x_vcmd) and, for each x point, average y over a window
    of size win = max(11, int(len(x)*win_frac)) (forced odd) around that sorted index.
    Edge windows are truncated without dropping points.
    """
    assert x_vcmd.shape == y.shape
    order = np.argsort(x_vcmd)
    x_ord, y_ord = x_vcmd[order], y[order]

    win = max(11, int(len(x_ord) * win_frac)) | 1
    half = win // 2

    y_smooth = np.empty_like(y_ord, dtype=float)
    y_std    = np.empty_like(y_ord, dtype=float)

    for i in range(len(x_ord)):
        lo = max(0, i - half)
        hi = min(len(x_ord), i + half + 1)
        seg = y_ord[lo:hi]
        y_smooth[i] = float(np.mean(seg))
        y_std[i]    = float(np.std(seg, ddof=0))

    return x_ord, y_smooth, y_std


# ──────────────────────────────────────────────────────────────
#  4) MAIN EVALUATION ENTRY POINT (used by evolution.py)
# ──────────────────────────────────────────────────────────────
def evaluation(exp_name: str,
               urdf_file: str,
               ckpt: int,
               envs: int,
               vmin: float,
               vmax: float,
               win_frac: float = 0.03,
               return_arrays: bool = True):
    """Run evaluation sweep and return three dicts (vel, eff, prog) computed
    from the *moving-averaged* curves vs v_cmd, plus final_reason and extra.

    The smoothing is done on the commanded velocity axis. We then take:
      • vel:   argmax of smoothed velocity subject to progress feasibility
      • eff:   argmin of smoothed energy per meter subject to progress feasibility
      • prog:  argmax of smoothed progress
    """
    gpu_id = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    # Genesis init is best-effort (no hard failure if already init'd)
    try:
        gs.init(logging_level="error", backend=gs.gpu)
    except Exception:
        pass

    LOG_ROOT = Path(os.getenv("LOG_ROOT", "logs")).expanduser().resolve()
    ANALYSIS_DIR = Path(os.getenv("ANALYSIS_DIR", "analysis")).expanduser().resolve()
    log_dir  = LOG_ROOT / "ea" / exp_name

    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, rew_cfg, cmd_cfg, train_cfg = pickle.load(f)
    # Clear reward shaping when evaluating
    rew_cfg["reward_scales"] = {}

    env_cfg.update(dict(
        visualize_camera=False, visualize_target=False,
        max_visualize_FPS=15, unique_forests_eval=True,
        growing_forest=True, episode_length_s=SUCCESS_TIME,
        x_upper=500, tree_radius=0.75,
        base_init_pos=[-50.0, 0.0, 10.0],
    ))

    env = WingedDroneEvalLinSpeed(
        num_envs=envs, v_min=vmin, v_max=vmax,
        env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=rew_cfg, command_cfg=cmd_cfg,
        urdf_file=urdf_file,
        show_viewer=False, eval=True, device=device,
    )

    # Deterministic eval
    for name in ("noise_sigma_mag", "noise_sigma_dir"):
        try:
            setattr(env.rigid_solver, name, 0.0)
        except Exception:
            pass

    # Load policy
    runner_cfg = copy.deepcopy(train_cfg)
    runner = OnPolicyRunner(env, runner_cfg, log_dir, device=gs.device)
    # evolution.py passes ckpt = train_iters  → we load model_{ckpt-1}.pt
    runner.load(str(log_dir / f"model_{ckpt-1}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    # Rollout
    v_mean, E_tot, v_cmd, progress, final_reason = run_eval(env, policy)
    try:
        gs.destroy()
    except Exception:
        pass

    # Filter valid data
    mask = np.isfinite(v_cmd) & np.isfinite(v_mean) & np.isfinite(E_tot) & np.isfinite(progress)
    x = np.asarray(v_cmd)[mask]      # v_cmd (smoothing axis)
    v = np.asarray(v_mean)[mask]     # achieved mean speed along x
    E = np.asarray(E_tot)[mask]      # energy per meter (J/m)
    P = np.asarray(progress)[mask]   # meters progressed

    # Moving-averages computed *vs v_cmd*
    x_for_v, v_ma, v_std = _moving_avg_vs_vcmd(x, v, win_frac)
    x_for_p, p_ma, p_std = _moving_avg_vs_vcmd(x, P, win_frac)
    x_for_E, E_ma, E_std = _moving_avg_vs_vcmd(x, E, win_frac)

    # Common x for the smoothed curves: use v_ma as abscissa ("black curve")
    x_line = v_ma  # MA(v_mean | v_cmd) used as the x-axis for plots/peaks

    # Peaks on smoothed curves
    max_p = float(np.nanmax(p_ma)) if len(p_ma) else float("nan")
    idx_p = int(np.nanargmax(p_ma)) if len(p_ma) else None

    # Feasible region (where progress exceeds a dynamic threshold is handled
    # upstream in evolution.py via minimal_p); here we compute the raw peaks
    idx_v = int(np.nanargmax(v_ma)) if len(v_ma) else None
    idx_e = int(np.nanargmin(E_ma)) if len(E_ma) else None

    def _pack(idx):
        if idx is None:
            return dict(mean_v=0.0, mean_E=100.0, mean_progress=0.0)
        return dict(mean_v=float(v_ma[idx]), mean_E=float(E_ma[idx]), mean_progress=float(p_ma[idx]))

    top_vel  = _pack(idx_v)
    top_eff  = _pack(idx_e)
    top_prog = _pack(idx_p)

    # Save total_plot in logs/ea/<exp_name>/ and, if possible, also in
    # ANALYSIS_DIR/runs/<row_id>/ (matching by exp_name in the CSV)
    try:
        plot_path = log_dir / "total_plot.png"
        total_plot(v_mean=v, E_tot=E, v_cmd=x, progress=P,
                   win_frac=win_frac, out=str(plot_path))
        # Best-effort copy to numbered run directory
        csv_path = ANALYSIS_DIR / "deap_temp.csv"
        if csv_path.is_file():
            import pandas as pd
            df = pd.read_csv(csv_path)
            cand = df[df.exp_name == exp_name]
            if not cand.empty and "row_id" in cand.columns:
                row_id = int(cand.iloc[-1].row_id)
                run_dir = ANALYSIS_DIR / "runs" / f"{row_id:05d}"
                run_dir.mkdir(parents=True, exist_ok=True)
                try:
                    # Rigenera il plot con titolo "0000x - …" direttamente nella cartella numerata
                    total_plot(v_mean=v, E_tot=E, v_cmd=x, progress=P,
                               win_frac=win_frac,
                               out=str(run_dir / "total_plot.png"),
                               title=f"{row_id:05d} - Performance (vs MA(v))")
                except Exception:
                    # fallback: copia il file già renderizzato
                    import shutil
                    shutil.copy2(plot_path, run_dir / "total_plot.png")
    except Exception as e:
        print(f"[eval] total_plot save failed: {e}")

    if return_arrays:
        extra = dict(
            p_s=p_ma, v_s=v_ma, E_s=E_ma,
            v_cmd_s=x_for_v, v_mean_s=v_ma, E_tot_s=E_ma,
            max_p=max_p,
        )
        return top_vel, top_eff, top_prog, final_reason, extra

    return top_vel, top_eff, top_prog, final_reason, max_p


# ──────────────────────────────────────────────────────────────
#  5) PLOTTING (TOTAL PLOT)
# ──────────────────────────────────────────────────────────────
def total_plot(v_mean: np.ndarray,
               E_tot: np.ndarray,
               v_cmd: np.ndarray,
               progress: np.ndarray,
               *,
               win_frac: float = 0.03,
               minimal_p: float = 300,
               out: str = "total_plot.png",
               title: str | None = None) -> None:
    """Create the 2-panel plot:
      • Top: progress vs MA(v_mean | v_cmd), colored by v_cmd
      • Bottom: energy-per-meter vs MA(v_mean | v_cmd), colored by v_cmd

    All moving averages are computed w.r.t. commanded velocity (v_cmd).
    """
    # Valid data
    mask = np.isfinite(v_cmd) & np.isfinite(v_mean) & np.isfinite(E_tot) & np.isfinite(progress)
    x = np.asarray(v_cmd)[mask]
    v = np.asarray(v_mean)[mask]
    E = np.asarray(E_tot)[mask]
    P = np.asarray(progress)[mask]

    # Moving averages vs v_cmd
    x_for_v, v_ma, v_std = _moving_avg_vs_vcmd(x, v, win_frac)
    x_for_p, p_ma, p_std = _moving_avg_vs_vcmd(x, P, win_frac)
    x_for_E, E_ma, E_std = _moving_avg_vs_vcmd(x, E, win_frac)

    # X-axis for the black curve: MA of achieved velocity
    x_line = v_ma
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(np.nanmin(x)) if len(x) else 0.0,
                         vmax=float(np.nanmax(x)) if len(x) else 1.0)

    # Minimal progress region shading (optional)
    intervals = []
    if minimal_p is not None and len(p_ma):
        below = p_ma < float(minimal_p)
        if np.any(below):
            i = 0
            while i < len(x_line):
                if below[i]:
                    j = i
                    while j + 1 < len(x_line) and below[j + 1]:
                        j += 1
                    intervals.append((float(x_line[i]), float(x_line[j])))
                    i = j + 1
                else:
                    i += 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 7.6), sharex=True,
                                   gridspec_kw=dict(hspace=0.10))
    if title:
        fig.suptitle(title)

    # Panel 1 — progress
    pts1 = np.array([x_line, p_ma]).T.reshape(-1, 1, 2)
    if len(pts1) > 1:
        segs1 = np.concatenate([pts1[:-1], pts1[1:]], axis=1)
        lc1 = LineCollection(segs1, cmap=cmap, norm=norm)
        lc1.set_array(0.5 * (x_for_v[:-1] + x_for_v[1:]))  # color by v_cmd mid-segment
        lc1.set_linewidth(2.2)
        ax1.add_collection(lc1)
        cb1 = plt.colorbar(lc1, ax=ax1, pad=0.01)
        cb1.set_label("Commanded Velocity [m/s]")
    ax1.fill_between(x_line, p_ma - p_std, p_ma + p_std, alpha=0.10)
    for a, b in intervals:
        ax1.axvspan(a, b, color="gray", alpha=0.30, lw=0)
    ax1.set_ylabel("Progress [m]")
    ax1.grid(alpha=0.25)

    # Panel 2 — energy per meter
    pts2 = np.array([x_line, E_ma]).T.reshape(-1, 1, 2)
    if len(pts2) > 1:
        segs2 = np.concatenate([pts2[:-1], pts2[1:]], axis=1)
        lc2 = LineCollection(segs2, cmap=cmap, norm=norm)
        lc2.set_array(0.5 * (x_for_v[:-1] + x_for_v[1:]))
        lc2.set_linewidth(2.2)
        ax2.add_collection(lc2)
        cb2 = plt.colorbar(lc2, ax=ax2, pad=0.01)
        cb2.set_label("Commanded Velocity [m/s]")
    ax2.fill_between(x_line, E_ma - E_std, E_ma + E_std, alpha=0.10)
    for a, b in intervals:
        ax2.axvspan(a, b, color="gray", alpha=0.30, lw=0)
    ax2.set_xlabel("MA of Achieved Velocity along X [m/s]")
    ax2.set_ylabel("Cost of Transport [J/m]")
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"✓ total_plot saved to {out}")
