#!/usr/bin/env python3
# eval.py
# ------------------------------------------------------------
# 1.  Valuta N env con velocità comandate distribuite linearmente
# 2.  Scatter:  v̄_x  vs  Energia totale   (colore = v_cmd)
# 3.  Timeseries (5 env):  joint-pos + throttle  |  E_int
# ------------------------------------------------------------
import os, pickle, math, copy, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from actor_critic_modified import ActorCriticTanh
import builtins
builtins.ActorCriticTanh = ActorCriticTanh
import genesis as gs
from winged_drone_env import WingedDroneEnv
from rsl_rl.runners import OnPolicyRunner
from pathlib import Path
import matplotlib.pyplot as plt

SUCCESS_TIME = 200.0  # tempo minimo per considerare un env come "successo"
# ──────────────────────────────────────────────────────────────
# 1) ENV DI VALUTAZIONE CON VELOCITÀ LINEARI FISSE
# ──────────────────────────────────────────────────────────────
class WingedDroneEvalLinSpeed(WingedDroneEnv):
    """
    Sottoclasse che **blinda** il comando di velocità su valori pre-calcolati
    in modo che qualunque reset mantenga gli stessi v_cmd.
    """
    def __init__(self, num_envs, v_min, v_max, *args, **kwargs):
        self._v_cmd = torch.linspace(v_min, v_max, num_envs)
        super().__init__(num_envs, *args, **kwargs)

    def _resample_commands(self, envs_idx):
        # azzera yaw-target, quota a 10 m e imposta la velocità desiderata
        self.commands[envs_idx, 0] = 0.0           # direzione
        self.commands[envs_idx, 1] = 10.0          # quota
        self.commands[envs_idx, 2] = self._v_cmd[envs_idx]

class WingedDroneEvalFixedSpeed(WingedDroneEnv):
    """Env che accetta una lista arbitraria di velocità comandate."""
    def __init__(self, v_list, *args, **kwargs):
        super().__init__(len(v_list), *args, **kwargs)
        self._v_cmd = torch.as_tensor(v_list, device=self.device, dtype=torch.float32)

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = 0.0           # direzione
        self.commands[envs_idx, 1] = 10.0          # quota
        self.commands[envs_idx, 2] = self._v_cmd[envs_idx]

# ──────────────────────────────────────────────────────────────
# 2) ROLL-OUT CHE RITORNA STATISTICHE **PER-ENV**
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def run_eval(env, policy, watch_env_idxs=()):
    B, dt, dev = env.num_envs, env.dt, env.device
    watch_env_idxs = list(watch_env_idxs)              # cast a lista

    done = torch.zeros(B, dtype=torch.bool, device=dev)
    t_acc  = torch.zeros(B, device=dev)
    dx_acc = torch.zeros(B, device=dev)
    E_acc  = torch.zeros(B, device=dev)
    final_reason = torch.full((B,), 3, dtype=torch.int8, device=dev)

    # Traces SOLO per gli env “watch”
    traces = {
        i: {"s": [], "j_pos": [], "thr": [], "E": [], "alpha": [], "beta": []}
        for i in watch_env_idxs
    }
    E_int = torch.zeros(len(watch_env_idxs), device=dev)  # accumulatore locale

    obs, _ = env.reset()
    x0 = env.base_pos[:, 0].clone()
    traces_all = {
        "s":   [[] for _ in range(B)],
        "j_pos": [[] for _ in range(B)],
        "v_cmd": env.commands[:, 2].cpu().numpy(),   # costante per env
    }
    print(f"Running evaluation on {B} envs – logging {watch_env_idxs}")

    # ---------- loop simulazione -------------------------------------------
    while not done.all():
        act = policy(obs)
        obs, _, term, _ = env.step(act)
        term = term.bool()
        nan_indices = env.nan_envs.to(torch.bool)  # env con NaN in osservazioni

        alive = (~done) & ~term & ~nan_indices
        t_acc[alive] += dt
        if alive.any():
            print(f"Elapsed time: {t_acc[alive].max().item():.2f} s")
        dx_acc[alive] = env.base_pos[alive, 0] - x0[alive]
        P = env.power_consumption()
        E_acc[alive] += P[alive] * dt

        with torch.no_grad():  # esplicito: niente grad
            base_x = env.base_pos[:, 0] - x0          # Δx per tutti gli env
            jp      = env.joint_position.cpu()        # (B, n_joints)

        for idx in range(B):
            if done[idx] or term[idx] or nan_indices[idx]:
                continue  # env già terminato a step precedente
            s_val = base_x[idx].item()
            traces_all["s"][idx].append(s_val)
            traces_all["j_pos"][idx].append(jp[idx].numpy())

        # ---- logging dettagliato solo per gli env di interesse -------------
        for k, idx in enumerate(watch_env_idxs):
            if done[idx] or term[idx]:
                continue
            s = (env.base_pos[idx, 0] - x0[idx]).item()
            tr = traces[idx]
            tr["s"].append(s)
            tr["j_pos"].append(env.joint_position[idx].cpu().numpy())
            tr["thr"].append(env.thurst[idx])
            E_int[k] += P[idx] * dt
            tr["E"].append(E_int[k].item())
            tr["alpha"].append(env.alpha[idx])
            tr["beta"].append(env.beta[idx])

        # ---- aggiorna stato done / final_reason ----------------------------
        just_done = (~done) & term
        for j in just_done.nonzero(as_tuple=False).flatten().tolist():
            if t_acc[j] > SUCCESS_TIME-0.1 or env.pre_success[j]:
                final_reason[j] = 0
            elif env.pre_collision[j]:
                final_reason[j] = 1
            elif env.pre_crash_condition[j]:
                final_reason[j] = 2
            else:
                final_reason[j] = 3
        done |= term | nan_indices

    # ---------- metriche globali -------------------------------------------
    v_mean = (dx_acc[~nan_indices] / t_acc[~nan_indices].clamp_min(1e-6)).cpu().numpy()
    E_tot  = (E_acc[~nan_indices] / dx_acc[~nan_indices].clamp_min(1e-6)).cpu().numpy()
    v_cmd  = env.commands[~nan_indices, 2].cpu().numpy()
    progress = dx_acc.cpu().numpy()
    final_reason = final_reason.cpu().numpy()

    # compatta j_pos in array (serve per il plot)
    for i in traces.keys():
        traces[i]["j_pos"] = np.vstack(traces[i]["j_pos"])

    for i in range(B):
        traces_all["s"][i]      = np.asarray(traces_all["s"][i])
        traces_all["j_pos"][i]  = (np.vstack(traces_all["j_pos"][i])
                                    if traces_all["j_pos"][i]
                                    else np.empty((0, env.joint_position.shape[1])))

    return v_mean, E_tot, v_cmd, progress, final_reason, traces, traces_all

def _moving_avg(x, y, win_frac=0.03):
    """
    Restituisce (x, y_smooth, y_std) a lunghezza intera.
    La finestra è pari a win_frac*len(x) (almeno 11, obblig. dispari).
    Agli estremi la finestra viene troncata senza eliminare i punti.
    """
    order = np.argsort(x)
    x_ord, y_ord = x[order], y[order]

    win = max(11, int(len(x_ord)*win_frac)) | 1      # forziamo dispari
    half = win // 2

    y_smooth = np.empty_like(y_ord, dtype=float)
    y_std    = np.empty_like(y_ord, dtype=float)

    for i in range(len(x_ord)):
        lo = max(0, i-half)
        hi = min(len(x_ord), i+half+1)
        seg = y_ord[lo:hi]
        y_smooth[i] = seg.mean()
        y_std[i]    = seg.std(ddof=0)

    return x_ord, y_smooth, y_std


# ──────────────────────────────────────────────────────────────
# 3) GRAFICI
# ──────────────────────────────────────────────────────────────
def plot_joint_diff_heatmap(traces_all, dof, out="joint_behaviour_heatmap.png",
                            s_bins=20, v_bins=16):
    """Colormap  s × v_cmd  → mean( j0 − j1 ).
    * *traces_all* è il dizionario prodotto da run_eval.
    * L'asse x sono i bin di s (distance covered).
    * L'asse y le velocità comandate (ordinate dal min al max).
    * Il colore è la media della differenza fra i primi due giunti.
    """
    v_cmd = traces_all["v_cmd"]
    v_min, v_max = v_cmd.min(), v_cmd.max()
    env_n = len(v_cmd)
    uniq_v = np.unique(v_cmd)
    uniq_v.sort()

    # massimo s complessivo per dimensionare i bin
    max_s = 0.0
    for i in range(env_n):
        if traces_all["s"][i].size:
            max_s = max(max_s, traces_all["s"][i].max())
    if max_s <= 0:
        print("⚠️  Nessun dato valido per la heatmap – salto plot.")
        return

    max_s = max( (s.max() if len(s) else 0) for s in traces_all["s"] )
    if max_s <= 0:
        print("⚠️  Nessun dato valido per la heat-map – salto plot.");  return
    s_edges = np.linspace(0, max_s, s_bins + 1)

    v_edges   = np.linspace(v_min, v_max, v_bins + 1)      # limiti dei bin
    v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])        # facoltativo: per tick

    heat  = np.full((v_bins, s_bins), np.nan)
    count = np.zeros_like(heat, dtype=int)

    # --- popola la matrice --------------------------------------------------
    for env_idx in range(env_n):
        s_arr  = traces_all["s"][env_idx]
        jp_arr = traces_all["j_pos"][env_idx]
        if s_arr.size == 0:
            continue
        if dof == "sweep":
            behaviour = np.rad2deg((jp_arr[:, 0] - jp_arr[:, 1])/2)          # (T,)
        elif dof == "twist":
            behaviour = np.rad2deg(-(jp_arr[:, 2] + jp_arr[:, 3])/2)
        else:
            raise ValueError(f"dof must be 'sweep' or 'twist', got '{dof}'")
        bins = np.searchsorted(s_edges, s_arr, side="right") - 1
        bins[bins == s_bins] = s_bins - 1
        v_val = v_cmd[env_idx]
        v_i = np.searchsorted(v_edges, v_val, side="right") - 1
        v_i = min(v_i, v_bins - 1)
        # scatter update
        for b in range(s_bins):
            mask = bins == b
            if not mask.any():
                continue
            m = behaviour[mask].mean()
            if np.isnan(heat[v_i, b]):
                heat[v_i, b] = m
            else:
                heat[v_i, b] = (heat[v_i, b] * count[v_i, b] + m) / (count[v_i, b] + 1)
            count[v_i, b] += 1

    # --- plot ---------------------------------------------------------------
    heat_masked = np.ma.masked_invalid(heat)          # ignora celle vuote
    vmin, vmax  = np.nanmin(heat), np.nanmax(heat)    # scala colori sui soli dati

    plt.figure(figsize=(9, 5))
    extent = [s_edges[0], s_edges[-1], v_edges[0], v_edges[-1]]
    im = plt.imshow(heat_masked, origin="lower", aspect="auto",
                    extent=extent, interpolation="nearest",
                    cmap="turbo", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="mean joint position [deg]")
    plt.xlabel("Distance covered s  [m]")
    plt.ylabel("Commanded velocity  [m/s]")
    plt.yticks(v_centers[::max(1, v_bins//8)])
    if dof == "sweep":
        plt.title("Joint‑behaviour heatmap (+ is forward sweep)")
    else:
        plt.title("Joint‑behaviour heatmap (+ is upward twist)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✅  heatmap salvato in {out}")



def _plot_scatter_with_band(x, y, c, xlabel, ylabel, out, vlines=None):
    """Generic scatter with moving‑avg curve, ±1σ band and optional vlines."""
    plt.figure(figsize=(6, 4))
    sc = plt.scatter(x, y, c=c, cmap="viridis", s=25, alpha=0.7)
    x_ma, y_ma, y_std = _moving_avg(x, y)
    plt.plot(x_ma, y_ma, color="black", lw=2, label="moving avg")
    plt.fill_between(x_ma, y_ma - y_std, y_ma + y_std,
                     color="black", alpha=0.15, label="±1 σ")

    # vertical guide‑lines if requested
    if vlines is not None:
        vlines = np.atleast_1d(vlines)
        for i, xv in enumerate(vlines):
            if i==0:
                lab = "Highest Vel"
            elif i == 1:
                lab = "Highest Eff"
            plt.axvline(xv, color="red", ls="--", lw=1.4, label=lab)

    plt.colorbar(sc).set_label("Commanded Velocity [m/s]")
    plt.xlabel(xlabel);  plt.ylabel(ylabel)
    plt.xlim(5, 21)
    plt.ylim(-0, +20)
    plt.legend();  plt.tight_layout();  plt.savefig(out, dpi=150)
    plt.close();  print(f"✅  scatter salvato in {out}")

# ──────────────────────────────────────────────────────────────
# 3‑D scatter interattivo (senza superficie)
# ──────────────────────────────────────────────────────────────
def plot_3d_speed_energy_agility(
        v_mean, E_tot, progress, v_cmd,
        html_out="speed_energy_progress_3D.html",
        cmap="Viridis",
):
    """
    Crea e salva un grafico 3‑D interattivo (Plotly) con:
      • x  = v_mean   [m/s]
      • y  = E_tot    [J/m]
      • z  = progress [m]
      • colore = v_cmd [m/s]
    Il file HTML risultante si apre in un browser e si può ruotare/zoomare.
    """
    import numpy as np
    import plotly.graph_objects as go

    # ─── filtra eventuali valori non finiti ────────────────────────────────
    mask = np.isfinite(v_mean) & np.isfinite(E_tot) & np.isfinite(progress)
    x, y, z, c = v_mean[mask], E_tot[mask], progress[mask], v_cmd[mask]

    # ─── scatter plotly ────────────────────────────────────────────────────
    fig = go.Figure(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(
                size=4,
                color=c,
                colorscale=cmap,
                colorbar=dict(title="v_cmd [m/s]"),
                opacity=0.8
            ),
            name="data‑points"
        )
    )

    fig.update_layout(
        title="Speed–Energy–Progress (3‑D)",
        scene=dict(
            xaxis=dict(
                title="v_mean [m/s]",
                range=[5, 20]             # <- limiti x
            ),
            yaxis=dict(
                title="E_tot [J/m]",
                range=[0, 7]          # <- limiti y
            ),
            zaxis=dict(
                title="Progress [m]",
                range=[0, 1000]           # <- limiti z
            ),
        ),
        margin=dict(l=0, r=0, t=35, b=0),
    )

    # ─── salva in HTML interattivo ─────────────────────────────────────────
    fig.write_html(html_out, include_plotlyjs="cdn")
    print(f"✅  grafico interattivo salvato in →  {html_out}")



def scatter_speed_energy(v_mean, E_tot, v_cmd, progress, out="speed_vs_energy.png", *, vline=None):
    mask = progress > 0
    _plot_scatter_with_band(v_mean[mask], E_tot[mask], v_cmd[mask],
                            "Mean Velocity x̄ [m/s]", "Energy per Meter [J/m]",
                            out, vlines=vline)



def scatter_speed_agility(v_mean, E_tot, v_cmd, progress, percentage=0.2,
                          out="speed_vs_agility.png"):
    mask = progress > 0
    x = v_mean[mask]
    y = progress[mask]
    c = v_cmd[mask]

    # === scatter + banda ±σ ===============================================
    plt.figure(figsize=(6, 4))
    sc = plt.scatter(x, y, c=c, cmap="viridis", s=25, alpha=0.7)
    x_ma, y_ma, y_std = _moving_avg(x, y)
    plt.plot(x_ma, y_ma, color="black", lw=2, label="moving avg")
    plt.fill_between(x_ma, y_ma - y_std, y_ma + y_std,
                     color="black", alpha=0.15, label="±1 σ")

    # === 3 linee verticali =================================================
    idx_peak = np.argmax(y_ma)          # indice del massimo
    peak_x   = y_ma[idx_peak]           # ascissa del picco
    lo_x     = peak_x * (1 - percentage)  # -20 % del picco

    plt.axhline(peak_x, color="red",  ls="--", lw=1.4,
                label=f"peak ({peak_x:.2f})")
    plt.axhline(lo_x,  color="gray", ls="--", lw=1.0,
                label=f"-20 % ({lo_x:.2f})")

    # === resto del layout ==================================================
    plt.colorbar(sc).set_label("Commanded Velocity [m/s]")
    plt.xlabel("Mean Velocity x̄ [m/s]")
    plt.ylabel("Progress [m]")
    plt.xlim(5, 21)
    plt.ylim(-0, +1100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✅  scatter salvato in {out}")



def plot_timeseries(traces, env, out=f"thurst_angles.png"):
    if not traces:
        return
    n = len(traces)
    fig, axs = plt.subplots(n, 1, figsize=(9, 2.8 * n), sharex=True)
    if n == 1:
        axs = [axs]

    # -------- limiti globali ----------
    left_min, left_max = np.inf, -np.inf
    right_min, right_max = np.inf, -np.inf
    for tr in traces.values():
        thr_arr = np.asarray(tr["thr"])
        yL = np.concatenate([tr["j_pos"].flatten(), thr_arr, tr["alpha"], tr["beta"]])
        yR = np.asarray(tr["E"])
        left_min, left_max = min(left_min, yL.min()), max(left_max, yL.max())
        right_min, right_max = min(right_min, yR.min()), max(right_max, yR.max())

    for ax, (idx, tr) in zip(axs, traces.items()):
        s   = np.array(tr["s"])
        jp  = tr["j_pos"]
        thr = np.array(tr["thr"])
        E   = np.array(tr["E"])
        alp = np.array(tr["alpha"])
        bet = np.array(tr["beta"])

        # -- left axis
        μt, σt = thr.mean(), thr.std()
        ax.plot(s, thr, "k--", lw=1, label="thr")
        ax.axhline(μt, color="gray", ls=":",  lw=2, label="μ thr")
        ax.axhspan(μt-σt, μt+σt, color="gray", alpha=0.15, label="±1 σ thr")
        #for d in range(jp.shape[1]):
        #    ax.plot(s, jp[:, d], label=f"j{d}")
        # ---------- ALPHA ------------------------------------------------------
        μa, σa = alp.mean(), alp.std()
        ax.plot(s, alp, "g", lw=1, label="α fus [°]")
        ax.axhline(μa, color="g", ls=":",  lw=2, label="μ α")
        ax.axhspan(μa-σa, μa+σa, color="g", alpha=0.15, label="±1 σ α")

        # ---------- BETA -------------------------------------------------------
        #μb, σb = bet.mean(), bet.std()
        #ax.plot(s, bet, "b", lw=1, label="β fus [°]")
        #ax.axhline(μb, color="b", ls=":",  lw=2, label="μ β")
        #ax.axhspan(μb-σb, μb+σb, color="b", alpha=0.15, label="±1 σ β")

        ax.set_ylabel("thr / α")
        ax.set_ylim(left_min, left_max)

        # -- right axis
        ax2 = ax.twinx()
        ax2.plot(s, E, "r", lw=1.2, label="E_int")
        ax2.set_ylabel("∫P dt [J]")
        ax2.set_ylim(right_min, right_max)
        ax2.tick_params(axis="y", colors="r")

        ax.set_title(f"env {idx}   v_cmd = {env.commands[idx,2].item():.1f} m/s")
        ax.grid(alpha=0.3)

    axs[-1].set_xlabel("Distance Covered Δx [m]")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=min(len(labels), 9))
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✅  timeseries salvato in {out}")


def evaluation(exp_name, urdf_file, ckpt, envs, vmin, vmax, win_frac=0.05, return_arrays=True):
    """Restituisce tre dizionari (vel, eff, prog) basati sul picco della media
    mobile lungo la v_cmd.  win_frac è la frazione di dati usata per la finestra
    (default 10 %)."""
    gpu_id = os.getenv("CUDA_VISIBLE_DEVICES","0").split(",")[0]
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    try:
        gs.init(
            logging_level    = "error",
            backend=gs.gpu,        # INFO o DEBUG per vedere tutti i messaggi
        )
    except:
        pass

    LOG_ROOT = Path(os.getenv("LOG_ROOT", "logs")).expanduser().resolve()
    log_dir  = LOG_ROOT / "ea" / exp_name
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, rew_cfg, cmd_cfg, train_cfg = pickle.load(f)
    rew_cfg["reward_scales"] = {}

    env_cfg.update(dict(visualize_camera=False, visualize_target=False,
                        max_visualize_FPS=15, unique_forests_eval=True,
                        growing_forest=True, episode_length_s=SUCCESS_TIME,
                        x_upper=1000, tree_radius=0.75))
    env = WingedDroneEvalLinSpeed(num_envs=envs, v_min=vmin, v_max=vmax,
                                  env_cfg=env_cfg, obs_cfg=obs_cfg,
                                  reward_cfg=rew_cfg, command_cfg=cmd_cfg,
                                  urdf_file = urdf_file,
                                  show_viewer=False, eval=True, device=device)
    runner_cfg = copy.deepcopy(train_cfg)
    runner = OnPolicyRunner(env, runner_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt-1}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    v_mean, E_tot, v_cmd, progress, final_reason, _, _ = run_eval(env, policy)

    gs.destroy()

    v_cmd_m, v_mean_m = v_cmd, v_mean
    E_tot_m, prog_m   = E_tot, progress

    xm_s, p_s,  _  = _moving_avg(v_mean_m, prog_m, win_frac)
    idx_p = np.argmax(p_s)        # progress max
    max_p = p_s[idx_p]

    idxs = np.arange(len(p_s))

    xc_s, v_s, _ = _moving_avg(v_cmd_m, v_mean_m, win_frac)
    xm_s, E_s,  _  = _moving_avg(v_mean_m, E_tot_m, win_frac)
    prog = p_s[idxs]
    vel = v_s[idxs]
    Energy = E_s[idxs]
    idx_p = np.argmax(prog)        # progress max
    # --- picchi ------------------------------------------------------------
    idx_v = np.argmax(vel)        # velocità massima
    idx_e = np.argmin(Energy)        # energia minima ⇒ efficienza max

    top_vel = {
        "mean_v":      vel[idx_v],
        "mean_E":      Energy[idx_v],
        "mean_progress": prog[idx_v],
    }
    top_eff = {
        "mean_v":      vel[idx_e],
        "mean_E":      Energy[idx_e],
        "mean_progress": prog[idx_e],
    }
    top_prog = {
        "mean_v":      vel[idx_p],
        "mean_E":      Energy[idx_p],
        "mean_progress": prog[idx_p],
    }
    if return_arrays:
        extra = dict(p_s=p_s, v_s=v_s, E_s=E_s,
                     v_cmd=v_cmd_m, v_mean=v_mean_m, E_tot=E_tot_m,
                     max_p=max_p)
        return top_vel, top_eff, top_prog, final_reason, extra
    return top_vel, top_eff, top_prog, final_reason, max_p


def plot_forest(env):
    cyl = env.cylinders_array[0].cpu().numpy()        # (N,3) – x,y,zCenter
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(cyl[:,0], cyl[:,1], s=4, c='forestgreen')
    #ax.set_aspect('equal')
    ax.set_xlim(-10, 1010)
    ax.set_ylim(-60, +60)
    ax.set_xlabel('x [m]');  ax.set_ylabel('y [m]')
    ax.set_title('First forest: tree positions')
    plt.tight_layout()
    plt.savefig("forest_layout.png", dpi=150)
    plt.close()
    print("✅  immagine salvata in forest_layout.png")

# ──────────────────────────────────────────────────────────────
# 4) MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-e","--exp_name", default="drone-forest")
    p.add_argument("--ckpt",  type=int, default=300)
    p.add_argument("--envs",  type=int, default=2048)
    p.add_argument("--vmin",  type=float, default=6.0)
    p.add_argument("--vmax",  type=float, default=18.0)
    p.add_argument("--gpu",   default="cuda")
    args = p.parse_args()

    # ───── load cfgs + policy ─────
    log_dir = f"logs/{args.exp_name}"
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, rew_cfg, cmd_cfg, train_cfg = pickle.load(f)

    rew_cfg["reward_scales"] = {}          # azzera reward ⇒ nessuna influenza

    gs.init(logging_level="error")

    env_cfg.update(dict(
        visualize_camera=False,
        visualize_target=False,
        max_visualize_FPS=100,
        unique_forests_eval=True,
        growing_forest=True,
        episode_length_s= SUCCESS_TIME,
        x_upper=1000,
        tree_radius=0.75,
    ))

    env = WingedDroneEvalLinSpeed(
        num_envs=args.envs,
        v_min=args.vmin, v_max=args.vmax,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg, reward_cfg=rew_cfg,
        command_cfg=cmd_cfg,
        show_viewer=False, eval=True
    )

    plot_forest(env)

    runner_cfg = copy.deepcopy(train_cfg)
    runner = OnPolicyRunner(env, runner_cfg,
                            log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    # velocità comandate generate da WingedDroneEvalLinSpeed
    v_cmd_all = np.linspace(args.vmin, args.vmax, args.envs)
    targets = np.array([9.0, 12.0, 15.0])
    tol = 1e-2

    watch_idxs = []
    for t in targets:
        matches = np.where(np.abs(v_cmd_all - t) < tol)[0]
        if len(matches) == 0:
            raise ValueError(f"Nessun env con v_cmd == {t} m/s")
        watch_idxs.append(int(matches[0]))
    print("Env da tracciare:", watch_idxs)

    # --- 1. chiamata unica -------------------------------------------------
    v_mean, E_tot, v_cmd, progress, final_reason, traces, traces_all = run_eval(
        env, policy,                       # <--  policy del primo env
        watch_env_idxs=watch_idxs  # <--  salva i dati di TUTTI gli env
    )

    # --- 2. conteggi -------------------------------------------------------
    n_success = int((final_reason == 0).sum())
    n_obst    = int((final_reason == 1).sum())
    n_walls   = int((final_reason == 2).sum())
    print(f"\n►  SUCCESSI: {n_success}/{args.envs}   |  "
        f"OBSTACLES: {n_obst}   |  WALLS: {n_walls}")
    
    v_cmd_m, v_mean_m = v_cmd, v_mean
    E_tot_m, prog_m   = E_tot, progress

    # curve smussate
    win_frac = 0.03

    xm_s, p_s,  _  = _moving_avg(v_mean_m, prog_m, win_frac)
    idx_p = np.argmax(p_s)        # progress max
    max_p = p_s[idx_p]
    percentage = 0.2
    minimum = max_p * (1-percentage)

    if max_p > minimum:

        # get the where p_s is greater than or equal to 0.80 * max_p
        idxs = np.where(p_s >= minimum)[0]

        xc_s, v_s, _ = _moving_avg(v_cmd_m, v_mean_m, win_frac)
        xe_s, E_s,  _  = _moving_avg(v_mean_m, E_tot_m, win_frac)
        prog = p_s[idxs]
        vel = v_s[idxs]
        Energy = E_s[idxs]
        idx_p = np.argmax(prog)        # progress max
        # --- picchi ------------------------------------------------------------
        idx_v = np.argmax(vel)        # velocità massima
        idx_e = np.argmin(Energy)        # energia minima ⇒ efficienza max

        top_vel = {
            "mean_v":      vel[idx_v],
            "mean_E":      Energy[idx_v],
            "mean_progress": prog[idx_v],
        }
        top_eff = {
            "mean_v":      vel[idx_e],
            "mean_E":      Energy[idx_e],
            "mean_progress": prog[idx_e],
        }
        top_prog = {
            "mean_v":      vel[idx_p],
            "mean_E":      Energy[idx_p],
            "mean_progress": prog[idx_p],
        }

    # --- risultati --------------------------------------------------------
    print(f"\nVelocità massima: {vel[idx_v]:.2f} m/s   |  Energia: {Energy[idx_e]:.2f} J/m   |  "
          f"Progress: {prog[idx_p]:.2f} m")

    scatter_speed_energy(v_mean, E_tot, v_cmd, progress, vline=[vel[idx_v], vel[idx_e]],
                         out=f"{args.exp_name}_{args.ckpt}_speed_vs_energy.png")

    scatter_speed_agility(v_mean, E_tot, v_cmd, progress, percentage,
                          out=f"{args.exp_name}_{args.ckpt}_speed_vs_agility.png")

    plot_timeseries(traces, env, 
                    out=f"{args.exp_name}_{args.ckpt}_thurst_angles.png")
    
    plot_joint_diff_heatmap(traces_all, "sweep", 
                            out=f"{args.exp_name}_{args.ckpt}_joint_behaviour_heatmap_sweep.png")
    
    plot_joint_diff_heatmap(traces_all, "twist", 
                            out=f"{args.exp_name}_{args.ckpt}_joint_behaviour_heatmap_twist.png")


    plot_3d_speed_energy_agility(
        v_mean, E_tot, progress, v_cmd,
        html_out=f"{args.exp_name}_{args.ckpt}_3D_speed_energy_progress.html"
    )