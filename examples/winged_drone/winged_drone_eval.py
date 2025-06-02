#!/usr/bin/env python3
# ------------------------------------------------------------
# hover_eval.py – evaluation + top-down video + parallel stats
# ------------------------------------------------------------
import argparse, os, pickle, math, copy
import numpy as np, torch

# Matplotlib non-GUI backend (per headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FFMpegWriter

import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from winged_drone_env import WingedDroneEnv
# ---------------------------------------------------------------------------


def compute_fov(pos_xy, yaw, roll, fov_angle_nom=30.0, fov_x_max=50.0):
    """
    Restituisce i 3 vertici del cono di visione a terra.
    yaw, roll in radianti.
    """
    alpha_eff = max(fov_angle_nom * abs(math.cos(roll)), 2.0)
    a = math.radians(alpha_eff)
    dx1, dy1 = fov_x_max * math.cos(yaw + a), fov_x_max * math.sin(yaw + a)
    dx2, dy2 = fov_x_max * math.cos(yaw - a), fov_x_max * math.sin(yaw - a)
    return np.vstack((pos_xy,
                      pos_xy + np.array([dx1, dy1]),
                      pos_xy + np.array([dx2, dy2])))


# ---------------------------------------------------------------------------
def create_topdown_video_multi(env, trajectories, save_path):
    """
    Renderizza N traiettorie simultanee con FOV e mette in risalto (alpha alto)
    gli ostacoli che ricadono in uno qualsiasi dei FOV.
    """
    # ── setup grafico ────────────────────────────────────────────────────────
    x0, x1 = env.env_cfg.get("x_lower", 0), env.env_cfg.get("x_upper", 200)
    y0, y1 = env.env_cfg.get("y_lower", -50), env.env_cfg.get("y_upper", 50)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal", "box")
    ax.set_xlim(x0 - 50, x1 + 10)
    ax.set_ylim(y0 - 10, y1 + 10)
    ax.set_title("Top-Down – Traiettorie Multiple con FOV")

    # foresta (bordo)
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                               lw=2, edgecolor="red", facecolor="none"))

    # alberi: creiamo una lista di (patch, coord)
    cyl = env.cylinders_array.cpu().numpy()
    obstacles = cyl[0] if cyl.ndim == 3 else cyl
    tree_r = env.env_cfg.get("tree_radius", 1.0)
    obstacle_patches = []
    for cx, cy, _ in obstacles:
        circ = Circle((cx, cy), tree_r, color="green", alpha=0.1)
        ax.add_patch(circ)
        obstacle_patches.append((circ, cx, cy))

    # ── traiettorie & FOV ───────────────────────────────────────────────────
    N = len(trajectories)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(N)]

    lines, markers, polys = [], [], []
    for col in colors:
        lines.append( ax.plot([], [], lw=2, color=col)[0] )
        markers.append( ax.plot([], [], "o", ms=4, color=col)[0] )
        poly = Polygon(np.empty((0, 2)), closed=True,
                       edgecolor=col, facecolor=col, alpha=0.15)
        ax.add_patch(poly)
        polys.append(poly)

    # timeline globale
    t_max = max(tr["time_steps"][-1] for tr in trajectories)
    t_vals = np.arange(0.0, t_max, env.dt)

    writer = FFMpegWriter(fps=int(1/env.dt), metadata=dict(artist="winged-drone"))
    with writer.saving(fig, save_path, dpi=200):
        for t in t_vals:
            # reset alpha di tutti gli alberi a base
            for circ, _, _ in obstacle_patches:
                circ.set_alpha(0.2)

            # per ogni episodio aggiorno path, marker, FOV
            for i, tr in enumerate(trajectories):
                ts = tr["time_steps"]
                idx = len(ts)-1 if t > ts[-1] else max(np.searchsorted(ts, t)-1, 0)
                pos  = tr["positions"]
                yaw  = tr["yaw"][idx]
                roll = tr["roll"][idx]

                # linea e marker
                lines[i].set_data(pos[:idx+1, 0], pos[:idx+1, 1])
                markers[i].set_data([pos[idx, 0]], [pos[idx, 1]])

                # FOV polygon
                fov_verts = compute_fov(pos[idx], yaw, roll)
                polys[i].set_xy(fov_verts)

                # evidenzio alberi nel FOV i-esimo
                # trasformo ogni albero in body-frame yaw
                cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
                tan_eff = math.tan(math.radians(max(2.0, 30.0*abs(math.cos(roll)))))
                for circ, cx, cy in obstacle_patches:
                    dx, dy = cx - pos[idx,0], cy - pos[idx,1]
                    x_loc = dx * cos_y - dy * sin_y
                    y_loc = dx * sin_y + dy * cos_y
                    if (x_loc >= 0) and (x_loc <= 50.0) and (abs(y_loc) <= tree_r + x_loc*tan_eff):
                        # se è nel FOV di questo drone aumento opacità
                        old = circ.get_alpha()
                        circ.set_alpha(min(1.0, max(old, 0.6)))

            writer.grab_frame()
    plt.close(fig)


# ---------------------------------------------------------------------------
def run_and_record(env, policy, collect_video=False):
    """
    Esegue fino a termine; se collect_video=True salva pos,yaw,roll,t step.
    Al ritorno tronca l'ultimo sample (post-reset).
    """
    device, B = env.device, env.num_envs
    done = torch.zeros(B, dtype=torch.bool, device=device)
    time_acc   = torch.zeros(B, device=device)
    energy_acc = torch.zeros(B, device=device)
    success_acc= torch.zeros(B, dtype=torch.bool, device=device)

    if collect_video and B==1:
        pos_b, yaw_b, roll_b, t_b = [], [], [], []

    obs, _ = env.reset()
    t = torch.zeros(B, device=device)

    while not done.all():
        with torch.no_grad():
            act = policy(obs)
        obs, _, term, _ = env.step(act)

        active = ~done
        time_acc  [active] += env.dt
        energy_acc[active] += env.power_consumption()[active] * env.dt

        just_done = active & term.to(torch.bool)
        success_acc[just_done] = env.pre_success[just_done].to(torch.bool)
        done |= term.to(torch.bool)

        if collect_video and B==1:
            pos_b .append(env.base_pos [0,:2].cpu().numpy())
            yaw_b .append(env.base_euler[0,2].cpu().numpy())
            roll_b.append(env.base_euler[0,0].cpu().numpy())
            t_b  .append(t[0].item())
        t += env.dt

    succ = success_acc.sum().item()
    mt   = time_acc[success_acc].mean().item()   if succ else float('nan')
    me   = energy_acc[success_acc].mean().item() if succ else float('nan')
    stats = dict(n_success=succ, mean_time=mt, mean_energy=me)

    if collect_video and B==1:
        traj = dict(
            positions  = np.vstack(pos_b)[:-1],
            yaw        = np.array(yaw_b)[:-1],
            roll       = np.array(roll_b)[:-1],
            time_steps = np.array(t_b)[:-1],
        )
        return stats, traj
    return stats, None


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--exp_name",  type=str, default="drone-forest")
    parser.add_argument("--ckpt",      type=int, default=300)
    parser.add_argument("--visual",    action="store_true")
    parser.add_argument("--video",     type=str, default="eval_topdown.mp4")
    parser.add_argument("--stats_envs",type=int, default=1000)
    parser.add_argument("--episodes",  type=int, default=3,
                        help="numero di episodi da filmare")
    args = parser.parse_args()

    # ─── init & config ─────────────────────────────────────────────────────
    gs.init(logging_level="error")
    log_dir = f"logs/{args.exp_name}"
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, rew_cfg, cmd_cfg, train_cfg = pickle.load(f)

    rew_cfg["reward_scales"] = {}
    env_cfg.update(dict(
        visualize_camera=True,
        visualize_target=False,
        max_visualize_FPS=60,
        unique_forests_eval=False
    ))
    env_cfg["num_trees_eval"] = 100
    if args.episodes>0:
        # ─── env (1 drone) ──────────────────────────────────────────────────────
        env1 = WingedDroneEnv(
            num_envs=1, env_cfg=env_cfg,
            obs_cfg=obs_cfg, reward_cfg=rew_cfg,
            command_cfg=cmd_cfg,
            show_viewer=args.visual, eval=True
        )
        runner_cfg = copy.deepcopy(train_cfg)
        runner = OnPolicyRunner(env1, runner_cfg, log_dir, device=gs.device)
        runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
        policy = runner.get_inference_policy(device=gs.device)

        # ── loop multi-episodi con start centrati ──────────────────────────────
        N  = args.episodes
        y0 = env_cfg["base_init_pos"][1]
        dy = 10.0

        trajectories, successes = [], 0
        for k in range(N):
            offset_y = (k - (N-1)/2) * dy
            env_cfg["base_init_pos"][1] = y0 + offset_y

            stats, traj = run_and_record(env1, policy, collect_video=True)
            trajectories.append(traj)
            successes += stats["n_success"]

            print(f"Ep{k+1}/{N}  y0={y0+offset_y:+.1f}m  "
                f"succ={bool(stats['n_success'])}  "
                f"dur={stats['mean_time']:.1f}s  E={stats['mean_energy']:.1f}J")

        # ─── render multi-traj + FOV dinamico ─────────────────────────────────
        print("\nRendering multi-trajectory video …")
        create_topdown_video_multi(env1, trajectories, args.video)
        print(f"✅  Video salvato in {args.video}   –   successi {successes}/{N}")

    # ─── statistiche parallele (opzionale) ─────────────────────────────────
    print(f"\n>>> Computing statistics on {args.stats_envs} envs")
    env_cfg_stats = {**env_cfg, **{"unique_forests_eval": True}}
    env2 = WingedDroneEnv(
        num_envs=args.stats_envs,
        env_cfg=env_cfg_stats,
        obs_cfg=obs_cfg, reward_cfg=rew_cfg,
        command_cfg=cmd_cfg,
        show_viewer=False, eval=True
    )
    runner2_cfg = copy.deepcopy(train_cfg)
    runner2 = OnPolicyRunner(env2, runner2_cfg, log_dir, device=gs.device)
    runner2.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy2 = runner2.get_inference_policy(device=gs.device)

    stats2, _ = run_and_record(env2, policy2, collect_video=False)
    print(f"Success rate: {stats2['n_success']/args.stats_envs*100:.2f}%   "
          f"Mean time: {stats2['mean_time']:.2f}s   "
          f"Mean energy: {stats2['mean_energy']:.2f}J")
