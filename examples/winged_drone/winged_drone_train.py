import argparse
import os
import pickle
import shutil
from importlib import metadata

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
from winged_drone_env import WingedDroneEnv


def get_train_cfg(exp_name, max_iterations):
    """
    Restituisce la configurazione per il training.
    La configurazione include i parametri dell’algoritmo (PPO in questo esempio), 
    la policy, e i parametri del runner.
    """
    train_cfg_dict = {
        "num_steps_per_env": 100,
        "save_interval": 50,
        "runner_class_name": "OnPolicyRunner",
        "empirical_normalization": True,
        "seed": 1,
        "logger": "tensorboard",
        "algorithm": {
            "normalize_advantage_per_mini_batch": True,
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.02,
            "entropy_coef": 0.002,
            "gamma": 0.997,
            "lam": 0.97,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "class_name": "ActorCriticRecurrent", # "ActorCriticRecurrent", "ActorCritic"
            "activation": "elu",
            "actor_hidden_dims": [256, 256, 256], # [256, 256, 256]
            "critic_hidden_dims": [256, 256, 256], # [256, 256, 256]
            "init_noise_std": 1.0,
            "rnn_type": "lstm",
            "rnn_hidden_size": 256, # 256
            "rnn_num_layers": 1, # 1

        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "policy_class_name": "ActorCriticRecurrent", # "ActorCriticRecurrent", "ActorCritic"
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
        "drone": "bix3", # "bix3", "morphing_drone"
        # Termination criteria
        "termination_if_alpha_greater_than": 120,  # degrees
        "termination_if_beta_greater_than": 60,
        "termination_if_close_to_ground": 1.0,
        "termination_if_x_greater_than": 500.0,
        "termination_if_y_greater_than": 50.0,
        "termination_if_z_greater_than": 20.0,
        # Base pose
        "base_init_pos": [-30.0, 0.0, 10.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 40.0,
        "at_target_threshold": 0.1,
        "resampling_time_s": 3.0,
        "simulate_action_latency": False,
        "clip_actions": 1.0,
        # Visualization options
        "visualize_target": False,
        "visualize_camera": True,
        "max_visualize_FPS": 60,
        # Foresta
        "num_trees": 70,
        "num_trees_eval": 100,
        "tree_radius": 1.0,
        "tree_height": 100.0,
        "unique_forests_eval": False,
        # ---------- DOMAIN RANDOMIZATION -------------
        "robot_randomization": True,
        "rand_mass_frac":     0.1,   # ±5 %   (scarto relativo)
        "rand_inertia_frac":  0.1,   # ±5 %   (stesso fattore della massa)
        "rand_every_reset":   False,   # randomizza anche ai reset

        # ---------- AERODYNAMIC NOISE ----------------
        "aero_noise":         True,
        "aero_noise_sigma0":  0.05,   # varianza base   (2 % del modulo)
        "aero_noise_k":       0.5,    # incremento per rad di |α|+|β|
    }
    obs_cfg = {
        "num_obs": 29,
        "obs_scales": {
            "zpos": 0.01,
            "lin_vel": 0.1,
            "ang_vel": 0.2,
            "depth": 1/60,
        },
    }
    reward_cfg = {
        "reward_scales": {
            "smooth": -1e-0,
            "angular": -5e-3, # -5e-3,
            "crash": -10, #10.0,
            "obstacle": -5e-1, 
            "energy": -5e-3,#-5e-3,
            "progress": 1e-2, # 1e-2
            "height": -5e-3, #-5e-3,
            "success": 0, #50.0
        },
    }
    command_cfg = {
        "num_commands": 2,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-forest",
                        help="Name of the experiment")
    parser.add_argument("-v", "--vis", action="store_true", default=False,
                        help="Enable viewer visualization")
    parser.add_argument("-B", "--num_envs", type=int, default=2048,
                        help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=500,
                        help="Maximum number of training iterations")
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

    start_lim   = 20      # [deg]
    final_lim   = 80     # [deg]
    warmup_iter = 60     # iter totali in cui il limite sale
    block_size  = 20      # aggiorna ogni 10 iter
    # -----------------------------------
    curr_it = 0

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    while curr_it < args.max_iterations:
        # ----- limite valido per il blocco corrente ---------------------------
        # 1) iterazione “effettiva” considerata nel ramp (massimo warmup_iter)
        if curr_it < warmup_iter:
            ratio = curr_it / warmup_iter
            lim   = start_lim + (final_lim - start_lim) * ratio
            iters_this_call = min(block_size, args.max_iterations - curr_it)
        else:
            # *** limite già saturato: un’unica chiamata per tutte le iter restanti ***
            lim = final_lim
            iters_this_call = args.max_iterations - curr_it
        env.set_angle_limit(lim)

        # ----- esegui il blocco di training ------------------------------------
        runner.learn(
            num_learning_iterations = iters_this_call,
            init_at_random_ep_len   = (curr_it == 0)          # solo la primissima volta
        )
        curr_it += iters_this_call
        if curr_it > warmup_iter:            # abbiamo fatto la big-iter; tutto finito
            break
if __name__ == "__main__":
    main()
