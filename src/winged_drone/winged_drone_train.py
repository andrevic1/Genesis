import argparse
import os
import pickle
import shutil
import torch
from importlib import metadata
from actor_critic_modified import ActorCriticTanh
import builtins
builtins.ActorCriticTanh = ActorCriticTanh
# Verifica della versione dei pacchetti rsl_rl e rsl-rl-lib
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner  # Runner per il training on-policy (ad es., PPO)
import genesis as gs
from pathlib import Path
from winged_drone_env import WingedDroneEnv

# -----------------------------------------------------------------------------
# Log root configurabile
# -----------------------------------------------------------------------------
def _get_log_root() -> Path:
    """
    Radice *persistente* per log & checkpoint.
    Usa $LOG_ROOT se presente, altrimenti fallback a 'logs' relativa.
    """
    root = Path(os.getenv("LOG_ROOT", "logs")).expanduser().resolve()
    # crea subito la sottodir 'ea' (esperimenti) per convenzione GA
    (root / "ea").mkdir(parents=True, exist_ok=True)
    return root


def get_train_cfg(exp_name, max_iterations):
    """
    Restituisce la configurazione per il training.
    La configurazione include i parametri dell’algoritmo (PPO in questo esempio), 
    la policy, e i parametri del runner.
    """
    train_cfg_dict = {
        "num_steps_per_env": 15, # 100
        "save_interval": 200,
        "runner_class_name": "OnPolicyRunner",
        "empirical_normalization": True,
        "seed": 1,
        "logger": "tensorboard",
        "algorithm": {
            "normalize_advantage_per_mini_batch": True,
            "class_name": "PPO",
            "clip_param": 0.2, # 0.2
            "desired_kl": 0.02, # 0.02
            "entropy_coef": 0.001,
            "gamma": 0.995,
            "lam": 0.97,
            "learning_rate": 0.0001, # 0.001
            "max_grad_norm": 0.3, # 1.0
            "num_learning_epochs": 3, # 6
            "num_mini_batches": 8, # 8
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 0.5, # 1.0
        },
        "init_member_classes": {},
        "policy": {
            "class_name": "ActorCriticTanh", # "ActorCriticRecurrent", "ActorCritic"
            "activation": "elu",
            "actor_hidden_dims": [64, 64], # [256, 256, 256]
            "critic_hidden_dims": [64, 64], # [256, 256, 256]
            "init_noise_std": 0.65, # 1.0
            "rnn_type": "lstm",
            "rnn_hidden_size": 64, # 256
            "rnn_num_layers": 1, # 1
            "max_servo": 1,         # ≈ 20°
            "max_throttle": 1,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "policy_class_name": "ActorCriticTanh", # "ActorCriticRecurrent", "ActorCritic"
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
        },
    }

    return train_cfg_dict


def get_cfgs():
    """
    Restituisce le configurazioni per l’environment, le osservazioni, il reward e i comandi.
    Queste configurazioni sono usate per costruire l’environment WingedDroneEnv.
    """
    env_cfg = {
        "num_actions": 5,
        "dt": 0.01,
        "drone": "morphing_drone", # "bix3", "morphing_drone"
        # Termination criteria
        "termination_if_alpha_greater_than": 120,  # degrees
        "termination_if_beta_greater_than": 60,
        "termination_if_close_to_ground": 1.0,
        "termination_if_x_greater_than": 600.0,
        "termination_if_y_greater_than": 50.0,
        "termination_if_z_greater_than": 20.0,
        # Base pose
        "base_init_pos": [-30.0, 0.0, 10.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 100.0,
        "at_target_threshold": 0.1,
        "resampling_time_s": 3.0,
        "simulate_action_latency": False,
        "clip_actions": 1.0,
        # Visualization options
        "visualize_target": False,
        "visualize_camera": True,
        "max_visualize_FPS": 100,
        # Foresta
        "num_trees": 1000,
        "num_trees_eval": 200,
        "tree_radius": 0.75,
        "tree_height": 100.0,
        "unique_forests_eval": False,
        "growing_forest": False,
        # ---------- DOMAIN RANDOMIZATION -------------
        "robot_randomization": True,
        "rand_mass_frac":     0.05,   # ±5 %   (scarto relativo)
        "rand_inertia_frac":  0.05,   # ±5 %   (stesso fattore della massa)
        "rand_every_reset":   False,   # randomizza anche ai reset

        # ---------- AERODYNAMIC NOISE ----------------
        "aero_noise":         True,
        "aero_noise_sigma0":  0.03,   # varianza base   (2 % del modulo)
        "aero_noise_k":       0.15,    # incremento per rad di |α|+|β|
    }
    obs_cfg = {
        "num_obs": 29,
        "add_noise": True,   # or False for evaluation
        "noise_std": {
            "z": 0.05,
            "quat": 0.05,
            "vel": 0.05,
            "depth": 0.05,
            "last_thr": 0.0,
            "last_jnts": 0.0,
            "v_tgt": 0.0,
        },
    }
    reward_cfg = {
        "reward_scales": {
            "smooth": -2e-2, # -3e-3,
            "angular": -5e-3, # -2e-2,
            "crash": -40, #-20.0,
            "obstacle": -5e-2, # -1e-1,
            "energy": -1e-4,#-1e-3,
            "progress": 3e-1, # 4e-1
            "height": -4e-2, #-5e-3,
            "success": 0, #0
            "cosmetic": -3e-1, # 0.0
        },
    }
    command_cfg = {
        "num_commands": 3,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def train(
    exp_name,
    urdf_file,
    num_envs,
    max_iterations,
    parent_exp: str | None = None,
    parent_ckpt: int | None = None,
    privileged: bool | None = True,
    log_root: Path | None = None,
):
    gpu_id = os.getenv("CUDA_VISIBLE_DEVICES","0").split(",")[0]
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    # Directory di log per il training
    if log_root is None:
        log_root = _get_log_root()
    else:
        # normalizza anche se passato come stringa
        log_root = Path(log_root).expanduser().resolve()
        (log_root / "ea").mkdir(parents=True, exist_ok=True)
    log_dir = (log_root / "ea" / exp_name).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        gs.init(logging_level="error", backend=gs.gpu)
    except Exception:
        pass

    # Ottieni le configurazioni per l'environment e per il training
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg["use_wide"] = privileged

    train_cfg = get_train_cfg(exp_name, max_iterations)

    # Salva le configurazioni usate per il training in un file pickle per futura consultazione
    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    # Crea l'environment
    env = WingedDroneEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        urdf_file=urdf_file,
        show_viewer=False,
        device=device,
    )

    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    # ----------------------------------------------
    #    POLICY INHERITANCE  (se init_ckpt ≠ None)
    # ----------------------------------------------
    if parent_exp is not None and parent_ckpt is not None:
        # Cerca checkpoint genitore in più location (nuovo schema prima).
        cand_paths = [
            _get_log_root() / "ea" / parent_exp / f"model_{parent_ckpt}.pt",
            _get_log_root() / parent_exp / f"model_{parent_ckpt}.pt",  # compat
            Path("logs") / parent_exp / f"model_{parent_ckpt}.pt",     # legacy
        ]
        ckpt_path = next((p for p in cand_paths if p.is_file()), None)

        if ckpt_path is not None:
            print(f"[train] Inheriting weights from {ckpt_path}")
            runner.load(str(ckpt_path))
        else:
            print(f"[train] ⚠  parent checkpoint not found – starting from scratch.")

    noise_sigma = 0.05
    env.rigid_solver.noise_sigma_mag = noise_sigma
    env.rigid_solver.noise_sigma_dir = noise_sigma

    runner.learn(
        num_learning_iterations = max_iterations,
        init_at_random_ep_len   = True
    )
    gs.destroy()

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-forest",
                        help="Name of the experiment")
    parser.add_argument("-v", "--vis", action="store_true", default=False,
                        help="Enable viewer visualization")
    parser.add_argument("-B", "--num_envs", type=int, default=16384,
                        help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=1500,
                        help="Maximum number of training iterations")
    parser.add_argument("--parent_exp", type=str, default=None,
                        help="Parent experiment name for policy inheritance")
    parser.add_argument("--parent_ckpt", type=int, default=None,
                        help="Parent checkpoint number for policy inheritance")
    parser.add_argument("--wide_depth_critic", action="store_true", default = False,
                        help="If set, critic gets 240°×60 depth privileged obs")
    args = parser.parse_args()

    # Inizializza Genesis con livello di logging a warning
    # Inizializza Genesis con livello di logging a warning
    gs.init(
        logging_level    = "error",
        backend=gs.gpu,        # INFO o DEBUG per vedere tutti i messaggi
    )

    # Directory di log per il training
    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Ottieni le configurazioni per l'environment e per il training
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg["use_wide"] = args.wide_depth_critic
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # Salva le configurazioni usate per il training in un file pickle per futura consultazione
    with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    # Attiva la visualizzazione target se richiesto via argomenti
    if args.vis:
        env_cfg["visualize_target"] = True

    # Crea l'environment
    env = WingedDroneEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
        device="cuda:0",
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # ----------------------------------------------
    #    POLICY INHERITANCE  (se init_ckpt ≠ None)  
    # ----------------------------------------------
    if args.parent_exp is not None and args.parent_ckpt is not None:
        parent_dir = os.path.join("logs", args.parent_exp)
        ckpt_path  = os.path.join(parent_dir, f"model_{args.parent_ckpt}.pt")

        if os.path.isfile(ckpt_path):
            print(f"[train] Inheriting weights from {ckpt_path}")
            runner.load(ckpt_path)
        else:
            print(f"[train] ⚠  checkpoint {ckpt_path} not found – starting from scratch.")

    start_lim   = 30      # [deg]
    final_lim   = 80     # [deg]
    warmup_iter = 0 #150    # iterazioni in cui il limite cresce
    block_size  = 30      # finché non satura, si impara a blocchi di 30
    noise_sigma_final = 0.0

    curr_it = 0
    while curr_it < args.max_iterations:

        # ---------- calcola il limite per l’iterazione corrente -----------------
        if curr_it < warmup_iter:
            ratio = curr_it / warmup_iter
            lim   = start_lim + (final_lim - start_lim) * ratio
            noise_sigma = noise_sigma_final * ratio
            iters_this_call = min(block_size, args.max_iterations - curr_it)
        else:
            # *** limite già saturato: un’unica chiamata per tutte le iter restanti ***
            lim = final_lim
            noise_sigma = noise_sigma_final
            iters_this_call = args.max_iterations - curr_it   # ≡ “una” super-iterazione

        env.set_angle_limit(lim)
        env.rigid_solver.noise_sigma_mag = noise_sigma
        env.rigid_solver.noise_sigma_dir = noise_sigma
        # ---------- esegui learn() ------------------------------------------------
        runner.learn(
            num_learning_iterations = iters_this_call,
            init_at_random_ep_len   = (curr_it == 0)
        )

        curr_it += iters_this_call
        if curr_it > warmup_iter:            # abbiamo fatto la big-iter; tutto finito
            break

if __name__ == "__main__":
    main()
