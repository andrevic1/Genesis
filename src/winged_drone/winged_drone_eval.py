#!/usr/bin/env python3
# ------------------------------------------------------------
# hover_eval.py – evaluation + top-down video + parallel stats
# ------------------------------------------------------------
import argparse, os, pickle, math, copy
import numpy as np, torch

# Matplotlib non-GUI backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FFMpegWriter

import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from winged_drone_env import WingedDroneEnv
from actor_critic_modified import ActorCriticTanh
import builtins
builtins.ActorCriticTanh = ActorCriticTanh
# ---------------------------------------------------------------------------

SUCCESS_TIME_SEC = 300.0          # volo minimo richiesto

import math
import numpy as np

def compute_fov(pos_xy, yaw, roll, fov_angle_nom=30.0, fov_x_max=60.0, n_points=30):
    """
    Restituisce un array di punti che descrivono un settore circolare (cono) 
    di raggio fov_x_max e apertura angolare 2*alpha_eff, posizionato in pos_xy,
    orientato lungo 'yaw'. Il settore è approssimato con n_points lungo l'arco.
    """
    # angolo effettivo, come prima
    alpha_eff = max(fov_angle_nom * abs(math.cos(roll)), 2.0)
    a = math.radians(alpha_eff)
    
    # genera n_points angoli equispaziati tra -a e +a, ruotati di yaw
    thetas = np.linspace(-a, a, n_points) + yaw
    
    # calcola i punti sull'arco di raggio fov_x_max
    arc_points = np.stack([
        fov_x_max * np.cos(thetas),
        fov_x_max * np.sin(thetas)
    ], axis=1)
    
    # sposta tutto sul vertice pos_xy
    arc_points += pos_xy

    # costruisce il settore: prima il vertice, poi tutti i punti dell'arco
    return np.vstack((pos_xy, arc_points))


# ===========================================================
# ===========           PATCH ‑‑ BEGIN           ============
# ===========================================================
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FFMpegWriter

# 1) TOP‑DOWN VIDEO ---------------------------------------------------------
def create_topdown_video_multi(env, trajectories, save_path,
                               dpi=300, scale_px_per_m=30.0):
    """Render video top‑down senza titoli né margini inutili,
    con risoluzione = bounding‑box reale della foresta."""
    x0, x1 = -40, env.env_cfg.get("x_upper", 400)        # parte da −40 m
    y0, y1 = env.env_cfg.get("y_lower", -50), env.env_cfg.get("y_upper", 50)
    W, H   = (x1-x0), (y1-y0)
    fig_w, fig_h = W/scale_px_per_m, H/scale_px_per_m    # “inch”
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_title("Top‑Down Trajectory", fontsize=30, pad=6)
    ax.set_xlabel("Distance Covered [m]", fontsize=12)
    ax.tick_params(axis="x", labelsize=8, bottom=True)  # mostra l’asse x
    ax.tick_params(axis="y", left=False, labelleft=False)  # nasconde la y
    ax.set_aspect("equal", "box")
    ax.set_xlim(x0, x1);  ax.set_ylim(y0, y1)
    ax.axis("off")                                # niente assi / titolo

    # ---------- oggetti scena (alberi, rettangolo rosso) ------------------
    cyl = env.cylinders_array.cpu().numpy()
    obstacles = cyl[0] if cyl.ndim == 3 else cyl
    tree_r = env.env_cfg.get("tree_radius", 1.0)
    obstacle_patches = []
    for cx, cy, _ in obstacles:
        circ = Circle((cx, cy), tree_r, color="green", alpha=0.15, linewidth=4)
        ax.add_patch(circ)
        obstacle_patches.append((circ, cx, cy))

    ax.add_patch(plt.Rectangle((x0, y0), W, H,
                               lw=8, edgecolor="red", facecolor="none"))
    # ---------- impostazione traiettorie ----------------------------------
    N = len(trajectories)
    cmap = plt.get_cmap("tab10")
    lines, markers, polys = [], [], []
    for i in range(N):
        col = cmap(i % 10)
        lines .append(ax.plot([], [], lw=5,  color=col)[0])
        markers.append(ax.plot([], [], "o", ms=5, color=col)[0])
        poly = Polygon(np.empty((0, 2)), closed=True,
                       edgecolor=col, facecolor=col, alpha=0.2)
        ax.add_patch(poly);  polys.append(poly)

    # ---------- rendering --------------------------------------------------
    t_max = max(tr["time_steps"][-1] for tr in trajectories)
    t_vals = np.arange(0.0, t_max, env.dt)

    writer = FFMpegWriter(fps=int(1/env.dt),
                          metadata=dict(artist="winged‑drone"))
    with writer.saving(fig, save_path, dpi=dpi):
        for t in t_vals:
            for circ, _, _ in obstacle_patches:
                circ.set_alpha(0.3)              # reset alpha

            for i, tr in enumerate(trajectories):
                ts = tr["time_steps"]
                if t > ts[-1]:
                    continue
                idx   = max(np.searchsorted(ts, t) - 1, 0)
                pos   = tr["positions"]
                yaw   = tr["yaw"][idx]
                roll  = tr["roll"][idx]
                lines[i].set_data(pos[:idx+1,0], pos[:idx+1,1])
                markers[i].set_data([pos[idx,0]], [pos[idx,1]])
                polys[i].set_xy(compute_fov(pos[idx], yaw, roll))

            writer.grab_frame()
    plt.close(fig)

# 2) OVERLAY  --------------------------------------------------------------

def create_overlay_video(cam_mp4: str,
                         td_mp4 : str,
                         traj    : dict,
                         out_mp4 : str = "camera_overlay.mp4",
                         v_commanded: float = 12.0,
                         dpi: int = 240):
    """
    • SINISTRA : video‑camera
    • DESTRA   : 5 subplot compatti
                 (Σ thrust, J0‑1, J2‑3, J4‑5, v‑lin xyz)
    • IN BASSO : video top‑down (tutta la larghezza)
    """
    import cv2, matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.gridspec import GridSpec

    # ---------- sorgenti video -------------------------------------------
    cap_cam = cv2.VideoCapture(cam_mp4)
    cap_td  = cv2.VideoCapture(td_mp4)

    fps = cap_cam.get(cv2.CAP_PROP_FPS)
    nF  = int(min(cap_cam.get(cv2.CAP_PROP_FRAME_COUNT),
                  cap_td .get(cv2.CAP_PROP_FRAME_COUNT)))
    dt  = 1.0 / fps

    # ---------- layout ----------------------------------------------------
    #  ▸ 3 colonne   [ videoCam | subplot | gap ]
    #  ▸ 6 righe     [5 subplot + 1 riga video top‑down]
    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    gs  = GridSpec(nrows=5, ncols=3,
                   width_ratios =[1.8, 0.05, 1.0],
                   height_ratios=[1.1, 1.1, 1.1, 1.1, 1.1],
                   wspace=0.08, hspace=0.34)

    # --- assi video -------------------------------------------------------
    ax_cam = fig.add_subplot(gs[:3, 0]);  ax_cam.axis("off")
    ax_td  = fig.add_subplot(gs[3:5 , 0]);  ax_td .axis("off")

    # --- assi serie temporali --------------------------------------------
    ax_T    = fig.add_subplot(gs[0, 2])   # Σ thrust
    ax_J01  = fig.add_subplot(gs[1, 2])   # J0‑1
    ax_J23  = fig.add_subplot(gs[2, 2])   # J2‑3
    ax_J45  = fig.add_subplot(gs[3, 2])   # J4‑5
    ax_VLIN = fig.add_subplot(gs[4, 2])   # vel. lineari
    ts_axes = [ax_T, ax_J01, ax_J23, ax_J45, ax_VLIN]

    # ---------- dati ------------------------------------------------------
    t_all   = traj["time_steps"]
    thr_sum = traj["thrust"].sum(axis=1)          # (N,)
    jp      = traj["joint_positions"]             # (N,6)
    vlin    = traj["lin_vel"]                     # (N,3)

    # ---------- limiti + griglie -----------------------------------------
    ax_T.set_ylim(0, 1)
    for ax in (ax_J01, ax_J23, ax_J45):
        ax.set_ylim(-0.3, 0.3)

    ax_VLIN.set_ylim(-2, v_commanded + 3)
    ax_VLIN.grid(True, lw=.3, alpha=.4)
    ax_VLIN.set_xlabel("t [s]")
    for ax in ts_axes:
        ax.grid(True, lw=.3, alpha=.4)

    # make vel commanded an array with same lenght of the others and always 13
    vel_commanded = np.full_like(t_all, v_commanded, dtype=np.float32)

    # ---------- linee + legende ------------------------------------------
    lnT , = ax_T   .plot([], [], c="tab:orange", lw=2.0, label="Σ thr")

    lnJ0, = ax_J01 .plot([], [], c="tab:blue" , lw=1.6, label="Sweep Mean")
    lnJ1, = ax_J01 .plot([], [], c="tab:cyan" , lw=1.6, label="Sweep Diff")

    lnJ2, = ax_J23 .plot([], [], c="tab:green", lw=1.6, label="Twist Mean")
    lnJ3, = ax_J23 .plot([], [], c="tab:olive", lw=1.6, label="Twist Diff")

    lnJ4, = ax_J45 .plot([], [], c="tab:red"  , lw=1.6, label="J4")
    lnJ5, = ax_J45 .plot([], [], c="tab:pink" , lw=1.6, label="J5")

    lnVx, = ax_VLIN.plot([], [], c="tab:purple" , lw=1.6, label="vx")
    lnVy, = ax_VLIN.plot([], [], c="tab:brown"  , lw=1.6, label="vy")
    lnVz, = ax_VLIN.plot([], [], c="tab:gray"   , lw=1.6, label="vz")
    lnVCOM, = ax_VLIN.plot([], [], c="k", lw=1.6, label="vCOM")

    # una sola legenda per asse (più compatta)
    ax_T   .legend(fontsize=9, frameon=False, loc="upper right")
    for ax in (ax_J01, ax_J23, ax_J45, ax_VLIN):
        ax.legend(fontsize=9, frameon=False, loc="upper right", ncol=2)

    # ---------- primi frame ----------------------------------------------
    _, frm_cam = cap_cam.read();  _, frm_td = cap_td.read()
    im_cam = ax_cam.imshow(cv2.cvtColor(frm_cam, cv2.COLOR_BGR2RGB))
    im_td  = ax_td .imshow(cv2.cvtColor(frm_td , cv2.COLOR_BGR2RGB))

    # ---------- writer ----------------------------------------------------
    writer = FFMpegWriter(fps=fps, metadata=dict(artist="winged‑drone"))
    with writer.saving(fig, out_mp4, dpi=dpi):
        for k in range(nF):
            if k:                       # aggiorna frame video
                rC, frm_cam = cap_cam.read()
                rT, frm_td  = cap_td .read()
                if not (rC and rT):
                    break
                im_cam.set_data(cv2.cvtColor(frm_cam, cv2.COLOR_BGR2RGB))
                im_td .set_data(cv2.cvtColor(frm_td , cv2.COLOR_BGR2RGB))

            t_now = k * dt
            idx   = max(np.searchsorted(t_all, t_now) - 1, 0)

            # --- aggiorna serie ------------------------------------------
            lnT .set_data(t_all[:idx+1], thr_sum[:idx+1])

            lnJ0.set_data(t_all[:idx+1], (jp[:idx+1, 0]-jp[:idx+1, 1])/2)
            lnJ1.set_data(t_all[:idx+1], (jp[:idx+1, 0]+jp[:idx+1, 1]))
            lnJ2.set_data(t_all[:idx+1], (jp[:idx+1, 2]+jp[:idx+1, 3])/2)
            lnJ3.set_data(t_all[:idx+1], (jp[:idx+1, 2]-jp[:idx+1, 3]))
            lnJ4.set_data(t_all[:idx+1], jp[:idx+1, 4])
            lnJ5.set_data(t_all[:idx+1], jp[:idx+1, 5])

            lnVx.set_data(t_all[:idx+1], vlin[:idx+1, 0])
            lnVy.set_data(t_all[:idx+1], vlin[:idx+1, 1])
            lnVz.set_data(t_all[:idx+1], vlin[:idx+1, 2])
            lnVCOM.set_data(t_all[:idx+1], vel_commanded[:idx+1])

            # scorrimento asse x
            for ax in ts_axes:
                ax.set_xlim(0, t_all[idx])

            writer.grab_frame()

    cap_cam.release(); cap_td.release()
    plt.close(fig)




def run_and_record(env, policy, show_video=False,
                   collect_video=False, video_cam_path="camera_view.mp4"):
    device, B = env.device, env.num_envs
    done = torch.zeros(B, dtype=torch.bool, device=device)
    time_acc = torch.zeros(B, device=device)
    energy_acc = torch.zeros(B, device=device)
    energy_propulsion = torch.zeros(B, device=device)
    energy_joints = torch.zeros(B, device=device)
    x_init = torch.zeros(B, device=device)
    x_progress = torch.zeros(B, device=device)
    straight = torch.zeros(B, device=device)
    final_reason = [""] * B

    if B == 1:
        pos_b, yaw_b, roll_b, t_b = [], [], [], []
        thrust_b = []
        joint_positions_b = []
        lin_vel_b = []

    if collect_video and env.rec_cam is not None and env.num_envs == 1:
        env.start_video(filename=video_cam_path, fps=int(1 / env.dt))
        cam_recording = True
    else:
        cam_recording = False

    obs, _ = env.reset()
    x_init[:] = env.base_pos[:, 0]
    t = torch.zeros(B, device=device)

    while not done.all():
        # salvo pre-step
        if B == 1:
            pos_b.append(env.base_pos[0, :2].cpu().numpy())
            yaw_b.append(env.base_euler[0, 2].cpu().numpy())
            roll_b.append(env.base_euler[0, 0].cpu().numpy())
            t_b.append(t[0].item())
            thrust_b.append(env.thurst[0].detach().cpu().numpy())
            joint_positions_b.append(env.joint_position[0].cpu().numpy())
            lin_vel_b.append(env.base_lin_vel[0].cpu().numpy())

        with torch.no_grad():
            act = policy(obs)
        obs, _, term, _ = env.step(act)

        terminated = term.to(torch.bool)
        Nan_mask = torch.isnan(env.base_pos[:, 0])
        still_flying = (~done) & (~terminated) & (~Nan_mask)

        time_acc[still_flying] += env.dt
        energy_acc[still_flying] += env.power_consumption()[still_flying] * env.dt
        energy_propulsion[still_flying] += env.cons_prop[still_flying] * env.dt
        energy_joints[still_flying] += env.cons_joint[still_flying] * env.dt
        x_progress[still_flying] = env.base_pos[still_flying, 0] - x_init[still_flying]
        straight[still_flying] += torch.norm(env.base_lin_vel[still_flying, 1:], dim=1) / env.base_lin_vel[still_flying, 0]
        just_done = (~done) & terminated
        for j in just_done.nonzero(as_tuple=False).flatten().tolist():
            if env.pre_collision[j]:
                final_reason[j] = "obstacle"
            elif env.pre_crash_condition[j]:
                final_reason[j] = "walls"
            else:
                final_reason[j] = "timeout"

        done |= terminated

        t += env.dt

    if cam_recording:
        env.stop_video()

    success_mask = time_acc >= SUCCESS_TIME_SEC-0.1
    n_completed_20s = int(success_mask.sum().item())

    mean_x = x_progress.mean().item()
    mean_survival_time = time_acc.mean().item()
    mean_energy_total = energy_acc.mean().item()
    mean_energy_per_m_x = (energy_acc / x_progress.clamp_min(1e-6)).mean().item()
    mean_en_propulsion = (energy_propulsion).mean().item()
    mean_en_joints = (energy_joints).mean().item()
    prop_to_en = mean_en_propulsion / mean_energy_total if mean_energy_total > 0 else float('nan')
    joints_to_en = mean_en_joints / mean_energy_total if mean_energy_total > 0 else float('nan')
    straightness = (straight/(time_acc/env.dt)).mean().item()

    # media X solo timeout
    timeout_mask = torch.tensor(
        [fr == "timeout" for fr in final_reason],
        dtype=torch.bool, device=device
    )
    if timeout_mask.any():
        mean_x_timeout = x_progress[timeout_mask].mean().item()
    else:
        mean_x_timeout = float('nan')

    # how many obstalces were hit?
    n_obstacles_hit = sum(fr == "obstacle" for fr in final_reason)
    if n_obstacles_hit > 0:
        print(f"⚠️  {(n_obstacles_hit/B*100):.2f}% ostacoli colpiti durante il volo")
    n_walls_hit = sum(fr == "walls" for fr in final_reason)
    if n_walls_hit > 0:
        print(f"⚠️  {(n_walls_hit/B*100):.2f}% collisioni con le pareti")
    if n_completed_20s > 0:
        print(f"✅  {(n_completed_20s/B*100):.2f}% episodi completati in {SUCCESS_TIME_SEC} s")

    stats = dict(
        n_completed_20s=n_completed_20s,
        mean_x=mean_x,
        mean_survival_time = mean_survival_time,
        mean_energy_total=mean_energy_total,
        mean_energy_per_m_x=mean_energy_per_m_x,
        prop_to_en=prop_to_en,
        joints_to_en=joints_to_en,
        mean_x_timeout=mean_x_timeout,
        straightness=straightness,
    )

    if B == 1:
        traj = dict(
            positions=np.vstack(pos_b),
            yaw=np.array(yaw_b),
            roll=np.array(roll_b),
            thrust=np.vstack(thrust_b),
            joint_positions=np.vstack(joint_positions_b),
            lin_vel=np.vstack(lin_vel_b),
            time_steps=np.array(t_b),
            end_reason=final_reason[0],
        )
        return stats, traj, cam_recording
    return stats, None, cam_recording

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-forest")
    parser.add_argument("--ckpt", type=int, default=300)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video", type=str, default="eval_topdown.mp4")
    parser.add_argument("--video_cam", type=str, default="camera_view.mp4",
                        help="percorso output MP4 della camera del drone")
    parser.add_argument("--stats_envs", type=int, default=2048)
    parser.add_argument("--episodes", type=int, default=3,
                        help="numero di episodi da filmare")
    parser.add_argument("--growing_forest", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(logging_level="error")
    log_dir = f"logs/{args.exp_name}"
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, rew_cfg, cmd_cfg, train_cfg = pickle.load(f)

    rew_cfg["reward_scales"] = {}

    env_cfg.update(dict(
    visualize_camera=False,
    visualize_target=False,
    max_visualize_FPS=15,
    unique_forests_eval=False,
    growing_forest=args.growing_forest,
    episode_length_s= SUCCESS_TIME_SEC,
    x_upper=500,
    tree_radius=0.75,
    base_init_pos=[-100, 0, 10.0],
    ))


    if args.episodes > 0:
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

        N = args.episodes
        y0 = env_cfg["base_init_pos"][1]
        dy = 10.0

        trajectories, completed = [], 0
        for k in range(N):
            offset_y = (k - (N - 1) / 2) * dy
            env_cfg["base_init_pos"][1] = y0 + offset_y

            stats, traj, cam_saved = run_and_record(
                env1, policy,
                show_video=args.visual,
                collect_video=(args.save_video and k == 0),
                video_cam_path=args.video_cam
            )
            trajectories.append(traj)
            completed += stats["n_completed_20s"]

            print(
                f"Ep{k + 1}/{N}  y0={y0 + offset_y:+.1f} m  "
                f"20 s ok={bool(stats['n_completed_20s'])}  "
                f"x̄={stats['mean_x']:.1f} m  "
                f"Ē={stats['mean_energy_total']:.1f} J  "
                f"Straightness={stats['straightness']:.2f}  "
            )
            if cam_saved and k == 0:
                print(f"✅  Video camera salvato in {args.video_cam}")

        print("\nRendering multi-trajectory video …")
        create_topdown_video_multi(env1, trajectories, args.video)
        print(
            f"✅  Video top-down salvato in {args.video}   –   "
            f"completati 20 s {completed}/{N}"
        )

        if args.save_video and cam_saved:
            print("Rendering overlay (camera + HUD) …")
            create_overlay_video(
                cam_mp4 = args.video_cam,
                td_mp4  = args.video,
                traj    = trajectories[0],
                v_commanded = env1.commands[0,2].cpu().item(),
                out_mp4 = "camera_overlay.mp4"
            )
            print("✅  Video finale con HUD in camera_overlay.mp4")

    print(f"\n>>> Computing statistics on {args.stats_envs} envs")
    
    env_cfg.update(dict(
    visualize_camera=False,
    visualize_target=False,
    max_visualize_FPS=15,
    unique_forests_eval=True,
    growing_forest=args.growing_forest,
    episode_length_s= SUCCESS_TIME_SEC,
    x_upper=500,
    tree_radius=0.75,
    base_init_pos=[-100, 0, 10.0],
    ))

    env2 = WingedDroneEnv(
        num_envs=args.stats_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg, reward_cfg=rew_cfg,
        command_cfg=cmd_cfg,
        show_viewer=False, eval=True
    )
    runner2_cfg = copy.deepcopy(train_cfg)
    runner2 = OnPolicyRunner(env2, runner2_cfg, log_dir, device=gs.device)
    runner2.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy2 = runner2.get_inference_policy(device=gs.device)

    stats2, _, _ = run_and_record(env2, policy2, collect_video=False)
    print(
        f"\n Completati 20 s: {stats2['n_completed_20s']}/{args.stats_envs}  "
        f"\n Mean X (tutti): {stats2['mean_x']:.1f} m  "
        f"\n Mean Survival Time: {stats2['mean_survival_time']:.1f}"
        f"\n Mean X (timeout): {stats2['mean_x_timeout']:.1f} m  "
        f"\n Energy: {stats2['mean_energy_total']:.1f} J  "
        f"\n Propulsion Energy Cut: {stats2['prop_to_en']:.3f}  "
        f"\n Joints Energy Cut: {stats2['joints_to_en']:.3f}  "
        f"\n Energy/X: {stats2['mean_energy_per_m_x']:.3f} J/m"
        f"\n Straightness: {stats2['straightness']:.3f}"
    )
