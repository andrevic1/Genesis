#!/usr/bin/env python3
# hover_eval_lin_speed.py
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
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
        super().__init__(num_envs, *args, **kwargs)
        self._v_cmd = torch.linspace(v_min, v_max, num_envs,
                                     device=self.device)

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
        "vel_x": [[] for _ in range(B)],
        "energy": [[] for _ in range(B)],
        "v_cmd": env.commands[:, 2].cpu().numpy(),   # costante per env
    }
    # ───────── buffer circolare: velocità ultimi 10 s ─────────────────────
    buf_size   = int(5.0 / dt) or 1          # passi simulazione in 10 s
    vel_ring   = torch.zeros((B, buf_size), device=dev)
    vel_ptr    = torch.zeros(B, dtype=torch.int32, device=dev)
    vel_count  = torch.zeros(B, dtype=torch.int32, device=dev)
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
            vel_x_tensor = env.base_lin_vel[:, 0]          # (B,) – GPU
            vel_x = vel_x_tensor.cpu()
    
            # ------- aggiorna ring-buffer della velocità -----------------------
            if alive.any():
                ids = alive.nonzero(as_tuple=False).flatten()
                p   = vel_ptr[ids]
                vel_ring[ids, p] = vel_x_tensor[ids]
                vel_ptr[ids] = (p + 1) % buf_size
                vel_count[ids] = torch.clamp(vel_count[ids] + 1, max=buf_size)
            energy = (P * dt).cpu()

        for idx in range(B):
            if done[idx] or term[idx] or nan_indices[idx]:
                continue  # env già terminato a step precedente
            s_val = base_x[idx].item()
            traces_all["s"][idx].append(s_val)
            traces_all["j_pos"][idx].append(jp[idx].numpy())
            traces_all["vel_x"][idx].append(vel_x[idx].numpy())
            traces_all["energy"][idx].append(energy.numpy())

        # ---- logging dettagliato solo per gli env di interesse -------------
        for k, idx in enumerate(watch_env_idxs):
            if done[idx] or term[idx]:
                continue
            s = (env.base_pos[idx, 0] - x0[idx]).item()
            tr = traces[idx]
            tr["s"].append(s)
            tr["j_pos"].append(env.joint_position[idx].cpu().numpy())
            tr["thr"].append(env.thurst[idx].detach().cpu().numpy().squeeze())
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
    t_acc = t_acc.cpu().numpy()

    # -------- velocità media sugli ultimi 10 s ----------------------------
    v_last10_mean = torch.full((B,), float("nan"), device=dev)
    full_buf  = vel_count == buf_size
    if full_buf.any():
        v_last10_mean[full_buf] = vel_ring[full_buf].mean(dim=1)
    part_buf = (~full_buf) & (vel_count > 0)
    for j in part_buf.nonzero(as_tuple=False).flatten():
        v_last10_mean[j] = vel_ring[j, :vel_count[j]].mean()
    v_last10_mean = v_last10_mean[~nan_indices].cpu().numpy()

    # compatta j_pos in array (serve per il plot)
    for i in traces.keys():
        traces[i]["j_pos"] = np.vstack(traces[i]["j_pos"])

    for i in range(B):
        traces_all["s"][i]      = np.asarray(traces_all["s"][i])
        traces_all["j_pos"][i]  = (np.vstack(traces_all["j_pos"][i])
                                    if traces_all["j_pos"][i]
                                    else np.empty((0, env.joint_position.shape[1])))
        traces_all["vel_x"][i]      = np.asarray(traces_all["s"][i])
        traces_all["energy"][i]  = np.vstack(traces_all["j_pos"][i])        

    return v_mean, v_last10_mean, E_tot, v_cmd, progress, t_acc, final_reason, traces, traces_all

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
# Helpers: formattazione e annotazioni per le triple di picco
# ──────────────────────────────────────────────────────────────
def _fmt_triplet(tag, T):
    label = {"prog": "MAX PROGRESS", "vel": "MAX VELOCITY", "eff": "MAX EFFICIENCY"}[tag]
    return f"{label} — (v={T['mean_v']:.2f} m/s, E={T['mean_E']:.2f} J/m, P={T['mean_progress']:.2f} m)"

def _draw_vlines_for_peaks(peaks, *, colors=None):
    """
    Ritorna lista di (key, x, kwargs, label) per axvline delle tre triple.
    colors: dict opzionale per personalizzare i colori.
    """
    if not peaks:
        return []
    default_colors = {"prog": "tab:orange", "vel": "tab:red", "eff": "tab:green"}
    if colors:
        default_colors.update(colors)
    spec = []
    for key in ("prog", "vel", "eff"):
        if key in peaks and peaks[key] is not None:
            v = float(peaks[key]["mean_v"])
            lab = _fmt_triplet(key, peaks[key])
            spec.append((key, v, dict(color=default_colors[key], ls="--", lw=1.6), lab))
    return spec


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

def _plot_scatter_with_band(
    x, y, c, xlabel, ylabel, out, 
    vlines=None, peaks=None, ylim=None,
    legacy_labels=True, legend_fontsize=9
):
    """Generic scatter con moving-avg, ±1σ, picchi e opzioni legenda."""
    plt.figure(figsize=(6, 4))
    sc = plt.scatter(x, y, c=c, cmap="viridis", s=25, alpha=0.7)
    x_ma, y_ma, y_std = _moving_avg(x, y)
    plt.plot(x_ma, y_ma, color="black", lw=2, label="moving avg")
    plt.fill_between(x_ma, y_ma - y_std, y_ma + y_std, color="black", alpha=0.15, label="±1 σ")

    # --- vecchie vlines (opzionali)
    if vlines is not None:
        vlines = np.atleast_1d(vlines)
        for i, xv in enumerate(vlines):
            lab = "Highest Vel" if (i == 0 and legacy_labels) else ("Highest Eff" if (i == 1 and legacy_labels) else None)
            plt.axvline(xv, color="red", ls="--", lw=1.4, label=lab)

    # --- vlines per le triple
    for _, xv, kwargs, lab in _draw_vlines_for_peaks(peaks):
        plt.axvline(xv, label=lab, **kwargs)

    cb = plt.colorbar(sc); cb.set_label("Commanded Velocity [m/s]")
    plt.xlabel(xlabel);  plt.ylabel(ylabel)
    plt.xlim(5, 25)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✅  scatter salvato in {out}")


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
                range=[4, 10]          # <- limiti y
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

# ──────────────────────────────────────────────────────────────
# 3‑D scatter + curva moving‑average
# ──────────────────────────────────────────────────────────────
def plot_3d_speed_energy_progress_ma(
        v_mean, E_tot, progress, v_cmd,
        win_frac=0.03,
        html_out="speed_energy_progress_ma_3D.html",
        cmap="Viridis",
):
    """
    Come lo scatter 3‑D classico, ma in più una linea che unisce
    (v_mean, 𝑚𝑎Energy, 𝑚𝑎Progress).

      • punti      →  x=v_mean,  y=E_tot,  z=progress
      • curva MA   →  x=x_ma,    y=E_ma,   z=P_ma
    """
    import numpy as np, plotly.graph_objects as go

    # ── filtra dati validi ───────────────────────────────────────────────
    mask = np.isfinite(v_mean) & np.isfinite(E_tot) & np.isfinite(progress)
    x, y, z, c = v_mean[mask], E_tot[mask], progress[mask], v_cmd[mask]

    # ── scatter 3‑D ──────────────────────────────────────────────────────
    fig = go.Figure(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=4, color=c, colorscale=cmap,
                        colorbar=dict(title="v_cmd [m/s]"),
                        opacity=0.8),
            name="data‑points"
        )
    )

    # ── curva moving‑average ─────────────────────────────────────────────
    #   1.  Energy vs v_mean
    x_ma, E_ma, _ = _moving_avg(x, y,  win_frac)
    #   2.  Progress vs v_mean (stessa x_ma garantita dall’ordinamento interno)
    _,   P_ma, _ = _moving_avg(x, z,  win_frac)

    fig.add_trace(
        go.Scatter3d(
            x=x_ma, y=E_ma, z=P_ma,
            mode="lines",
            line=dict(color="black", width=5),
            name=f"moving‑avg (win={win_frac:.0%})"
        )
    )

    # ── layout ───────────────────────────────────────────────────────────
    fig.update_layout(
        title="Speed–Energy–Progress (3‑D) + moving average",
        scene=dict(
            xaxis=dict(title="v_mean [m/s]", range=[5, 20]),
            yaxis=dict(title="E_tot [J/m]", range=[4, 10]),
            zaxis=dict(title="Progress [m]", range=[0, 1000]),
        ),
        margin=dict(l=0, r=0, t=35, b=0),
        legend=dict(x=0.02, y=0.98),
    )

    fig.write_html(html_out, include_plotlyjs="cdn")
    print(f"✅  grafico interattivo salvato in →  {html_out}")

def scatter_speed_energy(v_mean, E_tot, v_cmd, progress, out="speed_vs_energy.png", *, peaks=None, vline=None):
    mask = progress > 0
    _plot_scatter_with_band(
        v_mean[mask], E_tot[mask], v_cmd[mask],
        "Mean Velocity x̄ [m/s]", "Energy per Meter [J/m]",
        out,
        vlines=vline,          # rimane per compatibilità ma senza label
        peaks=peaks,
        ylim=(0, 20),
        legacy_labels=False,   # ⟵ niente "Highest Vel/Eff" in legenda
        legend_fontsize=9      # ⟵ legenda più piccola
    )

def scatter_speed_agility(v_mean, E_tot, v_cmd, progress, percentage=0.2, out="speed_vs_agility.png", *, peaks=None):
    mask = progress > 0
    x = v_mean[mask]
    y = progress[mask]
    c = v_cmd[mask]

    plt.figure(figsize=(6, 4))
    sc = plt.scatter(x, y, c=c, cmap="viridis", s=25, alpha=0.7)
    x_ma, y_ma, y_std = _moving_avg(x, y)
    plt.plot(x_ma, y_ma, color="black", lw=2, label="moving avg")
    plt.fill_between(x_ma, y_ma - y_std, y_ma + y_std, color="black", alpha=0.15, label="±1 σ")

    # --- PROGRESS MAX: orizzontale + verticale; la label completa la metto sulla orizzontale
    if peaks and "prog" in peaks:
        Pmax = float(peaks["prog"]["mean_progress"])
        V_at_Pmax = float(peaks["prog"]["mean_v"])
        plt.axhline(Pmax, color="tab:orange", ls="--", lw=1.6, label=_fmt_triplet("prog", peaks["prog"]))
        plt.axvline(V_at_Pmax, color="tab:orange", ls="--", lw=1.2)  # senza label per evitare doppione

    # --- VELOCITY & EFFICIENCY: verticali con label (così compaiono tutte le triple in legenda)
    for key, xv, kwargs, lab in _draw_vlines_for_peaks(peaks, colors={"prog": "tab:orange"}):
        if key == "prog":
            continue  # già etichettato con la hline sopra
        plt.axvline(xv, **kwargs, label=lab)

    cb = plt.colorbar(sc); cb.set_label("Commanded Velocity [m/s]")
    plt.xlabel("Mean Velocity x̄ [m/s]")
    plt.ylabel("Progress [m]")
    plt.xlim(5, 25)
    plt.ylim(0, 1100)

    # soglia -20% del max progress (facoltativa, mantiene la tua visual)
    if peaks and "prog" in peaks and percentage is not None:
        lo = float(peaks["prog"]["mean_progress"]) * (1 - percentage)
        plt.axhline(lo, color="gray", ls="--", lw=1.0, label=f"-{int(percentage*100)} % ({lo:.2f})")

    plt.legend(fontsize=9)  # ⟵ legenda più piccola
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✅  scatter salvato in {out}")



def scatter_speed_time(v_mean, v_cmd, time, out="speed_vs_time.png", *, peaks=None):
    x = v_mean
    y = time
    c = v_cmd

    plt.figure(figsize=(6, 4))
    sc = plt.scatter(x, y, c=c, cmap="viridis", s=25, alpha=0.7)
    x_ma, y_ma, y_std = _moving_avg(x, y)
    plt.plot(x_ma, y_ma, color="black", lw=2, label="moving avg")
    plt.fill_between(x_ma, y_ma - y_std, y_ma + y_std, color="black", alpha=0.15, label="±1 σ")

    # verticali per le tre triple (qui il valore di y non è usato; l'informazione è in legenda)
    for _, xv, kwargs, lab in _draw_vlines_for_peaks(peaks):
        plt.axvline(xv, label=lab, **kwargs)

    cb = plt.colorbar(sc); cb.set_label("Commanded Velocity [m/s]")
    plt.xlabel("Mean Velocity x̄ [m/s]")
    plt.ylabel("Time [s]")
    plt.xlim(5, 25)
    plt.ylim(0, 120)
    plt.legend(fontsize=9)
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


def evaluation(exp_name, urdf_file, ckpt, envs, vmin, vmax, win_frac=0.03, return_arrays=False):
    """Restituisce tre dizionari (vel, eff, prog) basati sul picco della media
    mobile lungo la v_cmd.  win_frac è la frazione di dati usata per la finestra
    (default 10 %)."""
    gpu_id = os.getenv("CUDA_VISIBLE_DEVICES","0").split(",")[0]
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    gs.init(
        logging_level    = "error",
        backend=gs.gpu,        # INFO o DEBUG per vedere tutti i messaggi
    )
    log_dir = f"logs/ea/{exp_name}"
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

    v_mean, v_mean_last_10s, E_tot, v_cmd, progress, t_acc, final_reason, _, _ = run_eval(env, policy)

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
    fig, ax = plt.subplots(figsize=(6,2.5))

    # Alberi
    ax.scatter(cyl[:,0], cyl[:,1], s=2, c='forestgreen', label="Trees")

    # Starting point
    ax.scatter(-30, 0, s=20, c='blue', marker='o', label="Starting point")

    # --- Rettangolo manuale ---
    x0, y0 = 0, -50
    width, height = 1000, 100

    # Lati lunghi e alto/basso -> continui
    ax.plot([x0, x0+width], [y0, y0], color="red", linewidth=2)           # lato basso
    ax.plot([x0, x0+width], [y0+height, y0+height], color="red", linewidth=2)  # lato alto
    ax.plot([x0+width, x0+width], [y0, y0+height], color="red", linewidth=2)   # lato destro

    # Lato sinistro -> tratteggiato
    ax.plot([x0, x0], [y0, y0+height], color="red", linewidth=2, linestyle="--")

    ax.annotate(
        '', xy=(1000, 55), xytext=(0, 55),
        arrowprops=dict(arrowstyle='->', color='black', linewidth=1.5)
    )

    ax.plot([], [], color='black', linewidth=1.5, label="Progress direction")
    # Legenda: aggiungo un handle fittizio per "wall"
    ax.plot([], [], color='red', linewidth=2, label="Wall")

    ax.set_xlim(-40, 1010)
    ax.set_ylim(-60, +60)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Evaluation Forest Example')

    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig("forest_layout.png", dpi=150)
    plt.close()
    print("✅ immagine salvata in forest_layout.png")


def total_plot_points_instead_of_ma(
    v_mean, E_tot, v_cmd, progress,
    *, win_frac=0.03,
    percentage=0.10,
    minimal_p=None,
    custom_point=None,          # (v, p, eff) — usato per disegnare la palla verde
    out="total_plot_points_instead_of_ma.png",
    data_marker_size=22,        # punti “tutti i campioni”
    op_marker_size=80           # palla verde (dimensione costante)
):
    """
    Stesso frame/limiti/griglia/legenda di total_plot, MA:
      • Subplot 1: scatter di tutti i punti (x=v_mean, y=progress), colormap = v_cmd.
      • Subplot 2: scatter di tutti i punti (x=v_mean, y=E_tot), colormap = v_cmd.
      • Aggiunge l'OPERATIONAL POINT come palla verde (dimensione costante), in primo piano.
      • Mostra le zone NON ammissibili (dove MA(progress) < minimal_p) come bande opache grigie.
      • Mostra la 'minimal progress threshold' nel primo subplot.

    NOTA: niente moving average e niente linee blu/annotazioni dei picchi.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # ---- helper per il punto custom (v, p, eff) ---------------------------
    def _coerce_custom_point(cp):
        if cp is None:
            return (np.nan, np.nan, np.nan)
        if isinstance(cp, dict):
            return (float(cp.get("v", np.nan)),
                    float(cp.get("p", np.nan)),
                    float(cp.get("eff", np.nan)))
        if isinstance(cp, (tuple, list)) and len(cp) == 3:
            return tuple(map(float, cp))
        raise ValueError("custom_point deve essere (v, p, eff) o dict con chiavi v,p,eff")

    v_c, p_c, eff_c = _coerce_custom_point(custom_point)

    # ---- filtra dati validi ----------------------------------------------
    mask = (np.isfinite(v_mean) & np.isfinite(E_tot) &
            np.isfinite(progress) & np.isfinite(v_cmd))
    x = np.asarray(v_mean)[mask]
    E = np.asarray(E_tot)[mask]
    P = np.asarray(progress)[mask]
    C = np.asarray(v_cmd)[mask]

    # ---- ricava bande 'inadmissible' dalla MA del progress ----------------
    x_ma, p_ma, _ = _moving_avg(x, P, win_frac)
    max_p = float(np.nanmax(p_ma)) if len(p_ma) else np.nan
    if minimal_p is None and np.isfinite(max_p):
        minimal_p = max_p * (1.0 - float(percentage))

    def _find_intervals_below(xv, yv, thr):
        if len(xv) == 0 or not np.isfinite(thr): return []
        below = yv < thr
        if not np.any(below): return []
        out, i, n = [], 0, len(xv)
        while i < n:
            if below[i]:
                xs, j = xv[i], i
                while j + 1 < n and below[j + 1]:
                    j += 1
                out.append((float(xs), float(xv[j])))
                i = j + 1
            else:
                i += 1
        return out

    intervals = _find_intervals_below(x_ma, p_ma, minimal_p) if np.isfinite(minimal_p) else []

    # ---- figura: stessa area dati del total_plot + colonna separata per le colorbar
    fig = plt.figure(figsize=(8.2, 7.6))
    gs  = fig.add_gridspec(
        2, 2,
        width_ratios=[1.0, 0.08],   # ← colonna stretta per le colorbar (spostata a destra)
        hspace=0.10, wspace=0.25
    )
    ax1  = fig.add_subplot(gs[0, 0])
    ax2  = fig.add_subplot(gs[1, 0], sharex=ax1)
    cax1 = fig.add_subplot(gs[0, 1])   # ← sede colorbar top
    cax2 = fig.add_subplot(gs[1, 1])   # ← sede colorbar bottom
    ax2.set_xlim(5, 25)
    xmin, xmax = ax2.get_xlim()

    # =======================
    #   SUBPLOT 1 — PROGRESS
    # =======================
    sc1 = ax1.scatter(x, P, c=C, cmap="viridis", s=data_marker_size, alpha=0.75, zorder=3)
    if np.isfinite(minimal_p):
        ax1.axhline(minimal_p, ls="--", lw=1.4, color="tab:gray",
                    label="minimal progress threshold", zorder=4)

    cb1 = plt.colorbar(sc1, cax=cax1)
    cb1.set_label("Commanded Velocity [m/s]")

    if np.isfinite(v_c) and np.isfinite(p_c):
        ax1.scatter([v_c], [p_c], s=op_marker_size, c=["tab:red"],
                    edgecolors="none", label="progress only -- Operational Point",
                    zorder=12)
        
    ax1.set_ylim(0, 1000)
    ax1.set_ylabel("Progress [m]")
    ax1.grid(alpha=0.25)

    ax1.legend(fontsize=9, loc="best")

    # ==========================
    #   SUBPLOT 2 — ENERGIA/met
    # ==========================
    sc2 = ax2.scatter(x, E, c=C, cmap="viridis", s=data_marker_size, alpha=0.75, zorder=3)

    cb2 = plt.colorbar(sc2, cax=cax2)
    cb2.set_label("Commanded Velocity [m/s]")

    # Operational point anche qui, stessa dimensione e foreground
    if np.isfinite(v_c) and np.isfinite(eff_c):
        ax2.scatter([v_c], [eff_c], s=op_marker_size, c=["tab:red"],
                    edgecolors="none", zorder=12)

    ax2.set_ylim(3, 9)
    ax2.set_xlabel("Mean Velocity x̄ along progress direction [m/s]")
    ax2.set_ylabel("Cost of Transport [J/m]")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9, loc="best")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"✅  total_plot_points_instead_of_ma salvato in {out}")


def total_plot_progress_only_point(
    v_mean, E_tot, v_cmd, progress,
    *, win_frac=0.03,
    percentage=0.10,
    minimal_p=None,
    custom_point=None,   # (v, p, eff) — come in total_plot
    out="total_plot_progress_only_point.png"
):
    """
    Stesso layout (2×1, sharex, limiti, griglia, font, posizioni legenda) di total_plot,
    ma disegna SOLO:
      • progress only operational point (verde) nei due subplot,
      • le sue linee tratteggiate verso gli assi (orizzontali + verticali condivise),
      • la 'minimal progress threshold' (hline grigia nel subplot del progress).

    Nessuna curva MA, niente bande, niente picchi BLU.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- parse punto custom (come total_plot) ------------------------------
    def _coerce_custom_point(cp):
        if cp is None: return (np.nan, np.nan, np.nan)
        if isinstance(cp, dict):
            return (float(cp.get("v", np.nan)),
                    float(cp.get("p", np.nan)),
                    float(cp.get("eff", np.nan)))
        if isinstance(cp, (tuple, list)) and len(cp) == 3:
            return tuple(map(float, cp))
        raise ValueError("custom_point deve essere (v, p, eff) o dict con chiavi v,p,eff")

    v_c, p_c, eff_c = _coerce_custom_point(custom_point)

    # --- calcolo minimal_p coerente a total_plot ---------------------------
    mask = np.isfinite(v_mean) & np.isfinite(E_tot) & np.isfinite(progress)
    x = np.asarray(v_mean)[mask]
    E = np.asarray(E_tot)[mask]
    P = np.asarray(progress)[mask]

    # usa la stessa logica del total_plot per derivare minimal_p, ma non si plottano le MA
    x_ma, p_ma, _ = _moving_avg(x, P, win_frac)
    max_p = float(np.nanmax(p_ma)) if len(p_ma) else np.nan
    if minimal_p is None and np.isfinite(max_p):
        minimal_p = max_p * (1.0 - float(percentage))

    # --- figura identica per frame/limiti/griglie/legende ------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 7.6), sharex=True,
                                   gridspec_kw=dict(hspace=0.10))
    ax2.set_xlim(5, 25)
    xmin, xmax = ax2.get_xlim()

    Z_LINE, Z_MARK = 6, 10

    # =======================
    #   SUBPLOT 1 — PROGRESS
    # =======================
    if np.isfinite(v_c) and np.isfinite(p_c):
        ax1.plot(v_c, p_c, "o", ms=6, color="tab:red",
                 label="progress only -- Operational Point", zorder=Z_MARK)
        ax1.plot([xmin, v_c], [p_c, p_c], ls="--", lw=1.4, color="tab:red",
                 label="progress only -- Fitness", zorder=Z_LINE)
        ax1.plot([xmin], [p_c], marker="*", ms=11, color="tab:red",
                 zorder=Z_MARK, clip_on=False)
        ax1.axvline(v_c, ls="--", lw=1.4, color="tab:red", zorder=Z_LINE)

    if np.isfinite(minimal_p):
        ax1.axhline(minimal_p, ls="--", lw=1.4, color="tab:gray",
                    label="minimal progress threshold", zorder=Z_LINE)

    ax1.set_ylim(0, 1000)
    ax1.set_ylabel("Progress [m]")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9, loc="best")

    # ==========================
    #   SUBPLOT 2 — ENERGIA/met
    # ==========================
    if np.isfinite(v_c) and np.isfinite(eff_c):
        ax2.plot(v_c, eff_c, "o", ms=6, color="tab:red", zorder=Z_MARK)
        ax2.plot([xmin, v_c], [eff_c, eff_c], ls="--", lw=1.4, color="tab:red", zorder=Z_LINE)
        ax2.plot([xmin], [eff_c], marker="*", ms=11, color="tab:red", zorder=Z_MARK, clip_on=False)
        ax2.axvline(v_c, ls="--", lw=1.4, color="tab:red", zorder=Z_LINE)
        # stellina sull'asse X in basso, stessa estetica del total_plot
        ax2.plot([v_c], [3.0], marker="*", ms=11, color="tab:red", zorder=Z_MARK, clip_on=False)

    ax2.set_ylim(3, 9)
    ax2.set_xlabel("Mean Velocity x̄ along progress direction [m/s]")
    ax2.set_ylabel("Cost of Transport [J/m]")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9, loc="best")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"✅  total_plot_progress_only_point salvato in {out}")


def total_plot(
    v_mean, E_tot, v_cmd, progress,
    *, win_frac=0.03,
    percentage=0.10,           # usato solo se minimal_p è None
    minimal_p=None,            # soglia assoluta; se None → max_p*(1-percentage)
    custom_point=None,         # (v, p, eff)  — eff = Energia per metro
    out="total_plot.png"
):
    """
    Figura 2×1 (sharex=True)

    [1] Progress (MA) vs Mean Velocity
        - hline a minimal_p (grigio)
        - bande verticali dove MA(progress) < minimal_p
        - custom point VERDE (v,p): orizzontale → asse Y (stellina, sopra l’asse)
          + verticale condivisa fino in basso (passa anche nel 2° subplot)
        - max progress: orizzontale BLU → asse Y (quadratino, sopra l’asse)
        - max velocity (tra punti con progress > minimal_p): verticale BLU
          su entrambi i subplot fino alla scala di velocità in basso (quadratino)

    [2] Energia per metro (MA) vs Mean Velocity
        - stesse bande oscurate
        - custom point VERDE (v,eff): orizzontale → asse Y (stellina, sopra l’asse)
          + verticale condivisa (stellina su asse X in basso, sopra l’asse)
        - **min energy** calcolato SOLO dove progress_MA > minimal_p:
          orizzontale BLU → asse Y (quadratino, sopra l’asse)
        - ylim energia fissato a [0, 9]
    """
    # --- util: parser tripla custom
    def _coerce_custom_point(cp):
        if cp is None: return None
        if isinstance(cp, dict):
            return (float(cp.get("v", np.nan)),
                    float(cp.get("p", np.nan)),
                    float(cp.get("eff", np.nan)))
        if isinstance(cp, (tuple, list)) and len(cp) == 3:
            return tuple(map(float, cp))
        raise ValueError("custom_point deve essere (v, p, eff) o dict con chiavi v,p,eff")

    # --- filtra dati validi
    mask = np.isfinite(v_mean) & np.isfinite(E_tot) & np.isfinite(progress)
    x = np.asarray(v_mean)[mask]
    E = np.asarray(E_tot)[mask]      # Energia per metro
    P = np.asarray(progress)[mask]

    # --- curve MA (stesso x ordinato per entrambe)
    x_ma, p_ma, p_std = _moving_avg(x, P, win_frac)
    _,    E_ma, E_std = _moving_avg(x, E, win_frac)

    # --- soglia minimal_p e intervalli da oscurare
    max_p = float(np.nanmax(p_ma)) if len(p_ma) else np.nan
    if minimal_p is None and np.isfinite(max_p):
        minimal_p = max_p * (1.0 - float(percentage))

    def _find_intervals_below(xv, yv, thr):
        if len(xv) == 0 or not np.isfinite(thr): return []
        below = yv < thr
        if not np.any(below): return []
        out, i, n = [], 0, len(xv)
        while i < n:
            if below[i]:
                xs, j = xv[i], i
                while j + 1 < n and below[j + 1]:
                    j += 1
                out.append((float(xs), float(xv[j])))
                i = j + 1
            else:
                i += 1
        return out

    intervals = _find_intervals_below(x_ma, p_ma, minimal_p) if np.isfinite(minimal_p) else []

    # --- picchi BLU
    # max progress (orizzontale blu nel subplot progress)
    idx_p = int(np.nanargmax(p_ma)) if len(p_ma) else None
    x_at_pmax = float(x_ma[idx_p]) if idx_p is not None else None
    y_pmax    = float(p_ma[idx_p]) if idx_p is not None else None

    # max velocity nella zona con progress > minimal_p
    x_maxvel_zone = None
    y_at_xmax_zone = None
    if np.isfinite(minimal_p):
        ok = p_ma > minimal_p
        if np.any(ok):
            x_candidates = x_ma[ok]
            y_candidates = p_ma[ok]
            idx_loc = int(np.argmax(x_candidates))
            x_maxvel_zone = float(x_candidates[idx_loc])
            y_at_xmax_zone = float(y_candidates[idx_loc])

    # **min energy solo dove progress_MA > minimal_p**
    x_at_emin = None
    y_emin    = None
    if np.isfinite(minimal_p):
        okE = p_ma > minimal_p
        if np.any(okE):
            Emin_segment = E_ma[okE]
            x_segment    = x_ma[okE]
            j = int(np.nanargmin(Emin_segment))
            y_emin = float(Emin_segment[j])
            x_at_emin = float(x_segment[j])
    # fallback (se soglia non valida/nessun punto sopra soglia)
    if x_at_emin is None or not np.isfinite(x_at_emin):
        if len(E_ma):
            j = int(np.nanargmin(E_ma))
            x_at_emin = float(x_ma[j])
            y_emin    = float(E_ma[j])

    # --- custom point (v,p,eff=energia per metro)
    v_c = p_c = eff_c = np.nan
    cp = _coerce_custom_point(custom_point)
    if cp is not None:
        v_c, p_c, eff_c = cp

    # --- figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 7.6), sharex=True,
                                   gridspec_kw=dict(hspace=0.10))

    # limiti X coerenti
    ax2.set_xlim(5, 25)
    xmin, xmax = ax2.get_xlim()

    # zorders per stare sopra assi/spine
    Z_LINE  = 6
    Z_MARK  = 10

    # =======================
    #   SUBPLOT 1 — PROGRESS
    # =======================
    if np.isfinite(v_c) and np.isfinite(p_c):
        ax1.plot(v_c, p_c, "o", ms=6, color="tab:red", label="progress only -- Operational Point",
                 zorder=Z_MARK)
        # orizzontale verso asse Y (stellina)
        ax1.plot([xmin, v_c], [p_c, p_c], ls="--", lw=1.4, color="tab:red", label="progress only -- Fitness",
                 zorder=Z_LINE)
        ax1.plot([xmin], [p_c], marker="*", ms=11, color="tab:red",
                 zorder=Z_MARK, clip_on=False)
        # verticale condivisa
        ax1.axvline(v_c, ls="--", lw=1.4, color="tab:red",
                    zorder=Z_LINE)

    ax1.plot(x_ma, p_ma, color="black", lw=2.2, label="velocity tracking -- Moving Average")
    ax1.fill_between(x_ma, p_ma - p_std, p_ma + p_std, color="purple", label="±1 σ",
                    alpha=0.1)

    for a, b in intervals:
        ax1.axvspan(a, b, color="gray", alpha=0.30, lw=0)

    # orizzontale BLU dal max progress → asse Y (quadratino sopra l'asse)
    if y_pmax is not None and np.isfinite(y_pmax):
        ax1.plot([xmin, x_at_pmax], [y_pmax, y_pmax], ls="--", lw=1.4,
                 color="tab:blue", zorder=Z_LINE, label="velocity tracking -- Fitness")
        ax1.plot([xmin], [y_pmax], marker="s", ms=9, color="tab:blue",
                 zorder=Z_MARK, clip_on=False)

    if np.isfinite(minimal_p):
        ax1.axhline(minimal_p, ls="--", lw=1.4, color="tab:gray",
                    label=f"minimal progress threshold", zorder=Z_LINE)

    # verticale BLU max velocity (zona ammessa) — continua nel 2° subplot
    if x_maxvel_zone is not None and np.isfinite(y_at_xmax_zone):
        ax1.axvline(x_maxvel_zone, ls="--", lw=1.4, color="tab:blue",
                    zorder=Z_LINE)  # niente label: la regione è rappresentata dal rettangolo grigio in legenda


    # custom point VERDE + linee + stelline (sopra assi)


    # assi Y da 0
    y1min, y1max = ax1.get_ylim()
    ax1.set_ylim(0, 1000)
    ax1.set_ylabel("Progress [m]")
    ax1.grid(alpha=0.25)
    # aggiungi rettangolino grigio alla legenda per indicare la regione
    _handles1, _labels1 = ax1.get_legend_handles_labels()
    _handles1.append(Patch(facecolor="gray", alpha=0.30, label="NOT admissible region"))
    ax1.legend(handles=_handles1, fontsize=9, loc="best")

    # ==========================
    #   SUBPLOT 2 — ENERGIA/met
    # ==========================
    ax2.plot(x_ma, E_ma, color="black", lw=2.2)
    ax2.fill_between(x_ma, E_ma - E_std, E_ma + E_std, color="purple",
                    alpha=0.1)

    for a, b in intervals:
        ax2.axvspan(a, b, color="gray", alpha=0.30, lw=0)

    # orizzontale BLU dal min energy (solo zona > minimal_p) → asse Y
    if y_emin is not None and np.isfinite(y_emin):
        ax2.plot([xmin, x_at_emin], [y_emin, y_emin], ls="--", lw=1.4,
                 color="tab:blue", zorder=Z_LINE)
        ax2.plot([xmin], [y_emin], marker="s", ms=9, color="tab:blue",
                 zorder=Z_MARK, clip_on=False)

    # verticale BLU max velocity (continuazione) + quadratino su asse X
    if x_maxvel_zone is not None and np.isfinite(y_at_xmax_zone):
        ax2.axvline(x_maxvel_zone, ls="--", lw=1.4, color="tab:blue",
                    zorder=Z_LINE)
        ax2.plot([x_maxvel_zone], [3.0], marker="s", ms=9, color="tab:blue",
                 zorder=Z_MARK, clip_on=False)

    # custom point VERDE su energia + linee verdi e stelline (sopra assi)
    if np.isfinite(v_c) and np.isfinite(eff_c):
        ax2.plot(v_c, eff_c, "o", ms=6, color="tab:red",
                 zorder=Z_MARK)
        # orizzontale verso asse Y (stellina)
        ax2.plot([xmin, v_c], [eff_c, eff_c], ls="--", lw=1.4, color="tab:red",
                 zorder=Z_LINE)
        ax2.plot([xmin], [eff_c], marker="*", ms=11, color="tab:red",
                 zorder=Z_MARK, clip_on=False)
        # verticale condivisa + stellina su asse X in basso
        ax2.axvline(v_c, ls="--", lw=1.4, color="tab:red",
                    zorder=Z_LINE)
        ax2.plot([v_c], [3.0], marker="*", ms=11, color="tab:red",
                 zorder=Z_MARK, clip_on=False)

    # assi Y: energia 0..9 fisso
    ax2.set_ylim(3, 9)

    ax2.set_xlabel("Mean Velocity x̄ along progress direction [m/s]")
    ax2.set_ylabel("Cost of Transport [J/m]")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9, loc="best")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"✅  total_plot salvato in {out}")

def total_plot_efficiency(
    v_mean, E_tot, v_cmd, progress,
    *, win_frac=0.03,
    percentage=0.10,           # usato solo se minimal_p è None
    minimal_p=None,            # soglia assoluta; se None → max_p*(1-percentage)
    custom_point=None,         # (v, p, eff)  — eff = Energia per metro (J/m) ⇢ verrà invertito
    out="total_plot_efficiency.png"
):
    """
    Come total_plot, ma nel [2] mostra l'inverso dell'energia per metro:
    meters per Joule [m/J]. La linea BLU indica il punto con m/J massimo
    nella 'admissible region' (progress_MA > minimal_p).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- util: parser tripla custom
    def _coerce_custom_point(cp):
        if cp is None: return None
        if isinstance(cp, dict):
            return (float(cp.get("v", np.nan)),
                    float(cp.get("p", np.nan)),
                    float(cp.get("eff", np.nan)))
        if isinstance(cp, (tuple, list)) and len(cp) == 3:
            return tuple(map(float, cp))
        raise ValueError("custom_point deve essere (v, p, eff) o dict con chiavi v,p,eff")

    # --- filtra dati validi
    mask = np.isfinite(v_mean) & np.isfinite(E_tot) & np.isfinite(progress)
    x = np.asarray(v_mean)[mask]
    E = np.asarray(E_tot)[mask]      # Energia per metro [J/m]
    P = np.asarray(progress)[mask]

    # --- curve MA (stesso x ordinato per entrambe)
    x_ma, p_ma, p_std = _moving_avg(x, P, win_frac)
    _,    E_ma, E_std = _moving_avg(x, E, win_frac)

    # --- soglia minimal_p e intervalli da oscurare (stessa logica dell’originale)
    max_p = float(np.nanmax(p_ma)) if len(p_ma) else np.nan
    if minimal_p is None and np.isfinite(max_p):
        minimal_p = max_p * (1.0 - float(percentage))

    def _find_intervals_below(xv, yv, thr):
        if len(xv) == 0 or not np.isfinite(thr): return []
        below = yv < thr
        if not np.any(below): return []
        out, i, n = [], 0, len(xv)
        while i < n:
            if below[i]:
                xs, j = xv[i], i
                while j + 1 < n and below[j + 1]:
                    j += 1
                out.append((float(xs), float(xv[j])))
                i = j + 1
            else:
                i += 1
        return out

    intervals = _find_intervals_below(x_ma, p_ma, minimal_p) if np.isfinite(minimal_p) else []

    # --- picchi su progress (come original)
    idx_p = int(np.nanargmax(p_ma)) if len(p_ma) else None
    x_at_pmax = float(x_ma[idx_p]) if idx_p is not None else None
    y_pmax    = float(p_ma[idx_p]) if idx_p is not None else None

    # --- punto di massima efficienza nella zona ammessa:
    #     min(E) ⇢ max(η), con η = 1/E
    x_at_eta_max = None
    eta_max      = None
    if np.isfinite(minimal_p):
        ok = p_ma > minimal_p
        if np.any(ok):
            E_seg = E_ma[ok]
            x_seg = x_ma[ok]
            j = int(np.nanargmin(E_seg))
            Emin = float(E_seg[j])
            x_at_eta_max = float(x_seg[j])
            if np.isfinite(Emin) and Emin > 0:
                eta_max = 1.0 / Emin
    # fallback se non c’è zona ammessa
    if (x_at_eta_max is None or eta_max is None) and len(E_ma):
        j = int(np.nanargmin(E_ma))
        Emin = float(E_ma[j])
        x_at_eta_max = float(x_ma[j])
        eta_max = 1.0 / max(Emin, 1e-9)

    # --- custom point (v,p,eff=J/m) → eff_inv per il 2° subplot
    v_c = p_c = eff_c = np.nan
    cp = _coerce_custom_point(custom_point)
    if cp is not None:
        v_c, p_c, eff_c = cp
    eta_c = (1.0 / eff_c) if np.isfinite(eff_c) and eff_c > 0 else np.nan

    # --- figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 7.6), sharex=True,
                                   gridspec_kw=dict(hspace=0.10))
    ax2.set_xlim(5, 25)
    xmin, xmax = ax2.get_xlim()

    Z_LINE, Z_MARK = 6, 10

    # =======================
    #   SUBPLOT 1 — PROGRESS
    # =======================
    ax1.plot(x_ma, p_ma, color="black", lw=2.2, label="velocity tracking RL setting")
    ax1.fill_between(x_ma, p_ma - p_std, p_ma + p_std, color="red", alpha=0.1)
    if np.isfinite(minimal_p):
        ax1.axhline(minimal_p, ls="--", lw=1.4, color="tab:gray", label="minimal progress", zorder=Z_LINE)

    for a, b in intervals:
        ax1.axvspan(a, b, color="gray", alpha=0.30, lw=0)

    # orizzontale BLU dal max progress → asse Y (quadratino)
    if y_pmax is not None and np.isfinite(y_pmax):
        ax1.plot([xmin, x_at_pmax], [y_pmax, y_pmax], ls="--", lw=1.4, color="tab:blue", zorder=Z_LINE, label="velocity tracking -- Fitness")
        ax1.plot([xmin], [y_pmax], marker="s", ms=9, color="tab:blue", zorder=Z_MARK, clip_on=False)

    # verticale BLU “admissible region” (velocità massima entro la zona ammessa)
    if np.isfinite(minimal_p):
        ok = p_ma > minimal_p
        if np.any(ok):
            x_candidates = x_ma[ok]
            y_candidates = p_ma[ok]
            idx_loc = int(np.argmax(x_candidates))
            x_maxvel_zone = float(x_candidates[idx_loc])
            ax1.axvline(x_maxvel_zone, ls="--", lw=1.4, color="tab:blue",
                        zorder=Z_LINE)

    if np.isfinite(v_c) and np.isfinite(p_c):
        ax1.plot(v_c, p_c, "o", ms=6, color="tab:green", label="best progress RL setting", zorder=Z_MARK)
        ax1.plot([xmin, v_c], [p_c, p_c], ls="--", lw=1.4, color="tab:green", zorder=Z_LINE)
        ax1.plot([xmin], [p_c], marker="*", ms=11, color="tab:green", zorder=Z_MARK, clip_on=False)
        ax1.axvline(v_c, ls="--", lw=1.4, color="tab:green", zorder=Z_LINE)

    ax1.set_ylim(0, 1000)
    ax1.set_ylabel("Progress — moving avg [m]")
    ax1.grid(alpha=0.25)
    # aggiungi rettangolino grigio alla legenda SOLO nel primo subplot
    _h1, _l1 = ax1.get_legend_handles_labels()
    _h1.append(Patch(facecolor="gray", alpha=0.30, label="admissible region"))
    ax1.legend(handles=_h1, fontsize=9, loc="best")

    # ==================================
    #   SUBPLOT 2 — METERS per JOULE
    # ==================================
    # Trasformo MA e banda ±1σ per inversione (stessi 'risultati', solo invertiti)
    eps = 1e-9
    E_lo = np.clip(E_ma - E_std, eps, None)
    E_hi = np.clip(E_ma + E_std, eps, None)
    eta_ma = 1.0 / np.clip(E_ma, eps, None)     # m/J
    eta_lo = 1.0 / E_hi                         # banda inferiore in m/J
    eta_hi = 1.0 / E_lo                         # banda superiore in m/J

    ax2.plot(x_ma, eta_ma, color="black", lw=2.2)
    ax2.fill_between(x_ma, np.minimum(eta_lo, eta_hi), np.maximum(eta_lo, eta_hi),
                     color="black", alpha=0.15)

    for a, b in intervals:
        ax2.axvspan(a, b, color="gray", alpha=0.30, lw=0)

    # BLU: evidenzia il punto di massima efficienza (m/J) nella zona ammessa
    if (x_at_eta_max is not None) and (eta_max is not None) and np.isfinite(eta_max):
        ax2.plot([xmin, x_at_eta_max], [eta_max, eta_max], ls="--", lw=1.4, color="tab:blue", zorder=Z_LINE)
        ax2.plot([xmin], [eta_max], marker="s", ms=9, color="tab:blue", zorder=Z_MARK, clip_on=False)
        #ax2.axvline(x_at_eta_max, ls="--", lw=1.4, color="tab:blue", zorder=Z_LINE)
        # quadratino sull'asse X in basso
        y_min, y_max = ax2.get_ylim()
        ax2.plot([x_at_eta_max], [y_min], marker="s", ms=9, color="tab:blue", zorder=Z_MARK, clip_on=False)

    # punto custom in efficienza (verde)
    if np.isfinite(v_c) and np.isfinite(eta_c):
        ax2.plot(v_c, eta_c, "o", ms=6, color="tab:green", zorder=Z_MARK)
        ax2.plot([xmin, v_c], [eta_c, eta_c], ls="--", lw=1.4, color="tab:green", zorder=Z_LINE)
        ax2.plot([xmin], [eta_c], marker="*", ms=11, color="tab:green", zorder=Z_MARK, clip_on=False)
        ax2.axvline(v_c, ls="--", lw=1.4, color="tab:green", zorder=Z_LINE)
        y_min, y_max = ax2.get_ylim()
        ax2.plot([v_c], [y_min], marker="*", ms=11, color="tab:green", zorder=Z_MARK, clip_on=False)

    # limiti Y: inverso dell'intervallo 3..9 J/m → ~0.111..0.333 m/J
    ax2.set_ylim(1/10.0, 1/4.0)

    ax2.set_xlabel("Mean Velocity x̄ [m/s]")
    ax2.set_ylabel("Meters per Joule  [m/J]")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9, loc="best")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"✅  total_plot_efficiency salvato in {out}")

# ──────────────────────────────────────────────────────────────
# 4) MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-e","--exp_name", default="drone-forest")
    p.add_argument("--ckpt",  type=int, default=300)
    p.add_argument("--envs",  type=int, default=8192)
    p.add_argument("--vmin",  type=float, default=5.0)
    p.add_argument("--vmax",  type=float, default=24.0)
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
    targets = np.array([])
    tol = 1e-2

    watch_idxs = []
    for t in targets:
        matches = np.where(np.abs(v_cmd_all - t) < tol)[0]
        if len(matches) == 0:
            raise ValueError(f"Nessun env con v_cmd == {t} m/s")
        watch_idxs.append(int(matches[0]))
    print("Env da tracciare:", watch_idxs)

    # --- 1. chiamata unica -------------------------------------------------
    v_mean, v_last10_mean, E_tot, v_cmd, progress, t_acc, final_reason, traces, traces_all = run_eval(
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
    percentage = 0.1
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

    peaks = {"vel": top_vel, "eff": top_eff, "prog": top_prog}

    # --- risultati --------------------------------------------------------
    print(f"\nVelocità massima: {vel[idx_v]:.2f} m/s   |  Energia: {Energy[idx_e]:.2f} J/m   |  "
          f"Progress: {prog[idx_p]:.2f} m")

    scatter_speed_energy(
        v_mean, E_tot, v_cmd, progress,
        peaks=peaks,
        out=f"{args.exp_name}_{args.ckpt}_speed_vs_energy.png"
    )

    scatter_speed_time(
        v_mean, v_cmd, t_acc,
        peaks=peaks,
        out=f"{args.exp_name}_{args.ckpt}_speed_vs_time.png"
    )

    scatter_speed_agility(
        v_mean, E_tot, v_cmd, progress, percentage,
        peaks=peaks,
        out=f"{args.exp_name}_{args.ckpt}_speed_vs_agility.png"
    )

    scatter_speed_agility(
        v_last10_mean, E_tot, v_cmd, progress, percentage,
        peaks=peaks,
        out=f"{args.exp_name}_{args.ckpt}_speed_vs_agility_5s.png"
    )

    scatter_speed_agility(
        v_cmd, E_tot, v_cmd, progress, percentage,
        peaks=peaks,
        out=f"{args.exp_name}_{args.ckpt}_speed_vs_agility_commanded.png"
    )

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

    plot_3d_speed_energy_progress_ma(
        v_mean, E_tot, progress, v_cmd,
        win_frac=0.03,                                 # o ciò che preferisci
        html_out=f"{args.exp_name}_{args.ckpt}_3D_speed_energy_progress_MA.html"
    )

    total_plot(
        v_mean, E_tot, v_cmd, progress,
        win_frac=0.03,
        minimal_p=500.0,                 # oppure ometti e usa percentage=0.10
        #percentage=0.10,
        custom_point=(14.9, 640.0, 6.5),
        out=f"{args.exp_name}_{args.ckpt}_total_plot.png"
    )

    total_plot_efficiency(
        v_mean, E_tot, v_cmd, progress,
        win_frac=0.03,
        minimal_p=500.0,                 # come nel total_plot per coerenza
        #percentage=0.10,                # oppure usa la percentuale
        custom_point=(14.9, 640.0, 6.5), # stesso punto: qui l’eff verrà mostrata come 1/6.5 m/J
        out=f"{args.exp_name}_{args.ckpt}_total_plot_efficiency.png"
    )

    total_plot_progress_only_point(
        v_mean, E_tot, v_cmd, progress,
        win_frac=0.03,
        minimal_p=500.0,                     # o lascia None e usa percentage
        # percentage=0.10,
        custom_point=(14.9, 640.0, 6.5),     # stesso formato di total_plot
        out=f"{args.exp_name}_{args.ckpt}_total_plot_progress_only_point.png"
    )

    total_plot_points_instead_of_ma(
        v_mean, E_tot, v_cmd, progress,
        win_frac=0.03,
        minimal_p=500.0,                     # o percentage=0.10
        custom_point=(14.9, 640.0, 6.5),
        out=f"{args.exp_name}_{args.ckpt}_total_plot_points_instead_of_ma.png"
    )
