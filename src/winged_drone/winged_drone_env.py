import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat, quat_to_R, xyz_to_quat
import time
from genesis.engine.solvers import RigidSolver
import numpy as np
from genesis.ext.pyrender.constants import RenderFlags


def gs_rand_float(lower, upper, shape, device):
    # gaussian noise
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

def gs_rand_normal(mean, std, shape, device):
    return torch.randn(size=shape, device=device) * std + mean

class WingedDroneEnv:

    BASE_OBS_SIZE = 8          # z-pos(1)+quat(4)+lin_vel_norm(1)
    THROTTLE_SIZE = 1
    MAX_DISTANCE = 40.0
    SHORT_RANGE = 0.0
    NUM_SECTORS_ACTOR  = 20      # 80 deg
    CONE_ACTOR_DEG     = 80.0
    NUM_SECTORS_CRITIC = 60      # 240 deg
    CONE_CRITIC_DEG    = 240.0

    NUM_SECTORS = None   # verrà settato in __init__
    CONE_ANGLE  = None
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, urdf_file=None,
                 show_viewer=False, eval=False, device="cuda"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(0)
        self.num_envs = num_envs
        self.rendered_env_num = self.num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.evaluation = eval
        self.growing_forest = True # self.evaluation and env_cfg.get("growing_forest", False)
        self.unique_forests_eval = env_cfg.get("unique_forests_eval", False)
        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.drone_name = env_cfg["drone"]
        self.show_viewer = show_viewer

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.reward_scales = reward_cfg["reward_scales"]

        CONTROL_HZ  = 25  # azioni alla PPO
        PHYSICS_HZ  = 200 # integrazione rigida
        dt_control  = 1.0 / CONTROL_HZ
        self.dt = dt_control
        substeps    = int(PHYSICS_HZ / CONTROL_HZ)   # 4

        self.dt = dt_control #env_cfg.get('dt', 0.01)
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        # repeat max episode length for each environment
        self.max_episode_length_per_env = torch.full((self.num_envs,), self.max_episode_length, device=self.device, dtype=gs.tc_int)
        # add a gaussian noise to the episode length 
        self.max_episode_length_per_env += gs_rand_normal(0, 1, (self.num_envs,), self.device).to(gs.tc_int)
        
        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt_control, substeps=substeps),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(-35.0, 0.0, 15.0),  # iniziale, poi seguiamo
                camera_lookat=(-28.0, 0.0, 10.0),
                res = (360, 360),                                              
            ),
            vis_options=gs.options.VisOptions(show_world_frame=False,
                                              world_frame_size=1.0,
                                              show_link_frame=False,
                                              plane_reflection=False,
                                              ambient_light=(0.1, 0.1, 0.1),
                                                shadow=False,
                                                background_color=(0.04, 0.08, 0.12),
                                                ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=False,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            renderer=gs.renderers.Rasterizer(),
        )



        self.scene.add_entity(gs.morphs.Plane())

        # drone entity
        self.base_init_pos = torch.tensor(env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        base_pos_np = self.base_init_pos.cpu().numpy()
        base_quat_np = self.base_init_quat.cpu().numpy()

       # camera
        self.camera_res = env_cfg.get("camera_res", (2560, 1920))
        self.camera_pos = env_cfg.get("camera_pos", (-5.0, 0.0, 2.0))
        self.camera_lookat = env_cfg.get("camera_lookat", (2.5, 0.0, 0.0))
        self.camera_fov = env_cfg.get("camera_fov", 30)

        self.rec_cam = None

        if self.drone_name == "morphing_drone":
            if urdf_file == None:
                urdf_file = "/home/avicari/Docker_RL/binding/Genesis/src/winged_drone/urdf_generated/[0.7, 0.2, -0.275, -0.275, 0.73, 0.18, 0.14, 0.16, 0.12, 1, 0, 0.1, 0.14].urdf" # STD DRONE
                #urdf_file = "/home/andrea/Documents/genesis/Genesis/genesis/assets/urdf/mydrone/[0.5, 0.15, -0.275, -0.275, 0.6, 0.14, 0.12, 0.12, 0.1, 1, 0, 0.1, 0.14].urdf"  # SMALLER DRONE
            self.drone = self.scene.add_entity(gs.morphs.URDF(
                file=urdf_file,
                pos=base_pos_np, quat=base_quat_np,
                collision=False, merge_fixed_links=True,
                links_to_keep=[
                    "aero_frame_fuselage",
                    "aero_frame_left_wing_prop",  "aero_frame_left_wing_free",
                    "aero_frame_right_wing_prop", "aero_frame_right_wing_free",
                    "aero_frame_elevator_left",   "aero_frame_elevator_right",
                    "aero_frame_rudder",
                    "elevator_left", "elevator_right","rudder",
                    "prop_frame_fuselage_0",
                    "fuselage","left_wing","right_wing",
                ],
            ))
            servo_joint_names = [
                "joint_0_sweep_left_wing",   # nuovo
                "joint_0_sweep_right_wing",  # nuovo
                "joint_1_twist_left_wing",
                "joint_1_twist_right_wing",
                "elevator_pitch_joint",
                "rudder_yaw_joint",
            ]

            # ④ ­– link su cui applicheremo le forze aerodinamiche
            self.link_name_array = [
                "fuselage",
                "left_wing",
                "right_wing",
                "elevator",
                "rudder",
                "prop_frame_fuselage_0",
            ]

        self.nominal_mass = float(sum(link.get_mass() for link in self.drone.links))
         # generate forests
        self._generate_forests()
        # initialize forest_ids

        # random initial assignment
        self.forest_ids = torch.randint(
            low=0,
            high=self.total_forests,
            size=(self.num_envs,),
            device=self.device,
            dtype=torch.int64
        )
        self.scene.build(n_envs=num_envs)

        self.robot_randomization = env_cfg.get("robot_randomization", False)
        if self.robot_randomization:
            self._randomize_physical_props() 

        self.rigid_solver = None
        for solver in self.scene.sim.solvers:
            if isinstance(solver, RigidSolver):
                self.rigid_solver = solver
                break
        if self.rigid_solver is None:
            raise Exception("RigidSolver not found!")
        # ---------- configura aerodinamica -----------------------------------
        if self.drone_name == "morphing_drone":
            if self.evaluation:
                self.rigid_solver._aero_log    = True
            else:
                self.rigid_solver._aero_log    = False                      # ② logga α,β,lift,drag
            self.rigid_solver.add_target(self.drone, urdf_file=urdf_file)   # ③ attiva aerodinamica
            self.span = self.rigid_solver.tip_to_tip
            self.rigid_solver._enable_noise = True
            

        print(f"WING SPAN: {self.span} m")
        # Identify servo joints (for wing twist) from URDF
        self.servo_dof_indices = [self.drone.get_joint(name).dof_idx_local for name in servo_joint_names]

        # Read joint limits from URDF
        self.throttle_limit = [0, 1]

        self.num_servos   = len(servo_joint_names)
        if self.drone_name == "bix3":
            self.num_actions = self.num_servos - 1 + self.THROTTLE_SIZE
            # rewrite joint_limits stacking the first, third and fourth servo
            self.joint_limits = [-0.3491, 0.3491]
        else:
            self.num_actions  = self.num_servos + self.THROTTLE_SIZE
            self.joint_limits = self.drone.get_dofs_limit(self.servo_dof_indices)

        self.use_wide = env_cfg.get("use_wide", True)

        if self.use_wide:
            self.NUM_SECTORS = self.NUM_SECTORS_CRITIC   # 60
            self.CONE_ANGLE  = self.CONE_CRITIC_DEG      # 240
        else:
            self.NUM_SECTORS = self.NUM_SECTORS_ACTOR    # 20
            self.CONE_ANGLE  = self.CONE_ACTOR_DEG       # 80

        # numero settori che finiranno nell'osservazione dell'attore
        self._n_act = self.NUM_SECTORS_ACTOR
        self._n_full = self.NUM_SECTORS
        # numero obs totali che l'attore vedrà
        self.num_obs = (
            self.BASE_OBS_SIZE     +
            1      +      # commands
            self.num_actions       +      # last actions
            self._n_act            )      # depth centrali

        ang_full = torch.linspace(
            - (self.CONE_CRITIC_DEG if self.use_wide else self.CONE_ACTOR_DEG)/2 * math.pi/180,
              (self.CONE_CRITIC_DEG if self.use_wide else self.CONE_ACTOR_DEG)/2 * math.pi/180,
              self._n_full, device=self.device)

        self.ray_dirs_body = torch.stack(
            [torch.cos(ang_full), torch.sin(ang_full), torch.zeros_like(ang_full)], 1)

        # indice d’inizio dei 20 centrali
        self._center_start = (self._n_full - self.NUM_SECTORS_ACTOR)//2

        if self.use_wide:
           self.num_privileged_obs = self.num_obs + (self._n_full - self._n_act)
        else:
            self.num_privileged_obs = self.num_obs

        # metti in coerenza i dizionari di config (utile per il logging)
        env_cfg["num_actions"] = self.num_actions
        obs_cfg["num_obs"]     = self.num_obs

        self.drone.set_dofs_kp(torch.full((self.num_servos,),10.0,device=self.device), self.servo_dof_indices)
        self.drone.set_dofs_kv(torch.full((self.num_servos,),  1.0,device=self.device), self.servo_dof_indices)

        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float32)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=torch.float32)
        self.success = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.pre_success = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float16)
        self.collision = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.crash_condition = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.pre_crash_condition = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float16)
        self.pre_collision = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float16)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float32)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.joint_position = torch.zeros((self.num_envs, self.num_servos), device=self.device, dtype=torch.float32)
        self.joint_velocity = torch.zeros((self.num_envs, self.num_servos), device=self.device, dtype=torch.float32)
        self.thurst = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        self.nan_envs = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int64)
        self.curr_limit = 120

        self.set_angle_limit(self.curr_limit)  # set initial angle limit

        self.depth = torch.empty((self.num_envs, self._n_full),
                                 device=self.device, dtype=torch.float16)

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

        self.reset()

        time.sleep(1)  # wait for the scene to be built


    # ================================================================
    # =============  API PUBBLICA PER LA REGISTRAZIONE  ===============
    # ================================================================
    def start_video(self, filename: str = "sim_record.mp4", fps: int | None = None):
        """Chiama prima dello step loop per cominciare a registrare."""
        if self._video_on:
            return
        self._video_file = filename
        self._video_fps  = fps or int(1 / self.dt)
        self.rec_cam.start_recording()
        self._video_on = True

    def stop_video(self):
        """Ferma la registrazione e salva il file MP4."""
        if not self._video_on:
            return
        self.rec_cam.stop_recording(self._video_file, fps=self._video_fps)
        self._video_on = False


    def _randomize_physical_props(self):
        """
        Domain-randomization di masse e inerzie.
        Usa l’API ufficiale: RigidLink.set_mass() scala automaticamente
        sia la massa che il tensore d’inerzia e aggiorna il solver.
        """
        if not self.robot_randomization or self.evaluation:
            return

        dm = self.env_cfg.get("rand_mass_frac", 0.05)      # ±5 %
        rng = torch.rand                                # alias locale

        for link in self.drone.links:
            mass0 = link.get_mass()                    # valore attuale
            if mass0>0:
                ratio  = 1.0 + (rng(1, device=self.device)*2 - 1) * dm
                link.set_mass(mass0 * ratio.item())          # API corretta


    def _generate_forests(self):
        # ─── parametri base ───────────────────────────────────────────────────────
        tree_radius = self.env_cfg.get("tree_radius", 1.0)
        tree_height = self.env_cfg.get("tree_height", 50.0)

        x_lower, x_upper = self.env_cfg.get("x_lower", 0),  self.env_cfg.get("x_upper", 150)
        y_lower, y_upper = self.env_cfg.get("y_lower", -50), self.env_cfg.get("y_upper", 50)

        # In evaluation lo spazio lungo x è più esteso
        if self.evaluation:
            x_upper = self.env_cfg.get("x_upper", 200)

        # ─── quante foreste? (regola unique_forests_eval) ─────────────────────────
        if self.evaluation:
            total_forests = self.num_envs * 5 if self.unique_forests_eval else 1
        else:
            total_forests = self.num_envs * 5
        self.total_forests = total_forests

        # ========================================================================
        # =============== FORESTA A DENSITÀ CRESCENTE (solo eval) ================
        # ========================================================================
        if self.growing_forest:
            dens_min = self.env_cfg.get("dens_min", 0.0)  # densità minima
            if self.evaluation:    
                dens_max = self.env_cfg.get("dens_max", 4.0)  # densità massima
            else:
                dens_max = self.env_cfg.get("dens_max", 3.0)  # densità massima
            width_x   = x_upper - x_lower
            expected_N = 0.5 * (dens_min + dens_max) * width_x
            num_trees  = int(math.ceil(expected_N))
            print(f"Growing forest: {num_trees} trees per forest")

            # Pdf lineare:  f(x) ∝ (x − x_lower)  ⇒  cdf F(x) = ((x−x0)/Δx)^2
            u  = torch.rand((total_forests, num_trees), device=self.device)
            xs = x_lower + width_x * torch.sqrt(u)        # inverse-transform

            ys = gs_rand_float(y_lower, y_upper,
                            (total_forests, num_trees), self.device)

            cylinders = torch.zeros((total_forests, num_trees, 3),
                                    dtype=torch.float16, device=self.device)
            cylinders[..., 0] = xs
            cylinders[..., 1] = ys
            cylinders[..., 2].fill_(tree_height * 0.5)

            self.cylinders_array = cylinders
            return  # finito: la versione “growing” non richiede altro

        # ========================================================================
        # ======================= FORESTE UNIFORMI (caso training) ===============
        # ========================================================================
        num_trees = self.env_cfg.get(
            "num_trees_eval" if self.evaluation else "num_trees",
            20 if self.evaluation else 100
        )
        cylinders = torch.zeros((total_forests, num_trees, 3),
                                dtype=torch.float16, device=self.device)

        xs = gs_rand_float(x_lower, x_upper, (total_forests, num_trees), self.device)
        ys = gs_rand_float(y_lower, y_upper, (total_forests, num_trees), self.device)

        # ― training: rimuovi randomicamente alcuni alberi per varietà
        if not self.evaluation:
            no_trees_int = gs_rand_float(0, num_trees, (total_forests,), self.device).to(torch.int)
            cols = torch.arange(num_trees, device=self.device).unsqueeze(0).expand(total_forests, -1)
            mask = cols >= (num_trees - no_trees_int.unsqueeze(1))
            xs[mask] = x_upper + 50
            ys[mask] = y_upper + 50

        cylinders[..., 0] = xs
        cylinders[..., 1] = ys
        cylinders[..., 2].fill_(tree_height * 0.5)

        # In evaluation, se non sono uniche, renderizzo solo la prima foresta
        if self.evaluation and not self.unique_forests_eval:
            for i in range(num_trees):
                self.scene.add_entity(
                    gs.morphs.Cylinder(
                        pos=cylinders[0, i].cpu().numpy(),
                        radius=tree_radius,
                        height=tree_height,
                        collision=False,
                        fixed=True
                    )
                )

        self.cylinders_array = cylinders

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = 0.0
        z_tgt  = 5
        self.commands[envs_idx, 1] = z_tgt           # target height
        v_tgt = gs_rand_float(6, 24,
                            (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = v_tgt           # target roll

        return
    
    def set_angle_limit(self, limit_deg: float):
        """Aggiorna dinamicamente i limiti usati per la crash-condition."""
        self.curr_limit = limit_deg
        self.roll_limit_rad = math.radians(limit_deg)
        self.pitch_limit_rad = math.radians(limit_deg)
        self.yaw_limit_rad = math.radians(limit_deg)

    def step(self, actions):
        #actions[:, 1:] = actions[:, 1:] / self.joint_limits[1]
        self.action = actions
        
        # 1) clamp throttle a [0,1]
        self.actions[:,0] = torch.clamp(
            self.action[:,0] * self.throttle_limit[1], 
            self.throttle_limit[0], 
            self.throttle_limit[1]
        )
        # 2) clamp joint pos a joint_limits
        self.actions[:,1:] = torch.clamp(
            self.action[:,1:] * self.joint_limits[1], 
            self.joint_limits[0], 
            self.joint_limits[1]
        )
                
        exec_servo = self.actions[:, 1:]
        exec_throttle = self.actions[:, 0]
            

        self.drone.control_dofs_position(exec_servo, self.servo_dof_indices)

        if self.drone_name == "morphing_drone":
            self.rigid_solver.set_throttle(exec_throttle)

        self.scene.step()

        # update buffers
        self.episode_length_buf += 1

        self.base_pos[:] = self.drone.get_dofs_position()[:, :3]
        self.base_quat[:] = self.drone.get_quat()
        #self.drone.set_dofs_position([0, 0, 0.2, 0.2, 0, 0], self.servo_dof_indices)
        self.joint_position = self.drone.get_dofs_position()[:, self.servo_dof_indices]
        self.joint_velocity = self.drone.get_dofs_velocity()[:, self.servo_dof_indices]
        self.torque = self.drone.get_dofs_control_force(self.servo_dof_indices)
        total_torque = self.drone.get_dofs_force(self.servo_dof_indices)
        self.base_euler = quat_to_xyz(
            self.base_quat,
            rpy=True, degrees=False
        )
        if self.evaluation:
            # --- thrust logging (sempre tensor 2D) ---------------------------------
            thr_t = self._extract_thrust_tensor()
            if self.thurst.shape != thr_t.shape:
                # ridimensiona buffer (raro, ma utile se #prop cambia)
                self.thurst = torch.zeros_like(thr_t)
            self.thurst.copy_(thr_t)
            self.alpha = self.rigid_solver._dbg["alpha"][:, 0]
            self.beta = self.rigid_solver._dbg["beta"][:, 0]
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = self.rigid_solver.get_dofs_velocity()[:, :3]
        self.norm_vel = torch.norm(self.base_lin_vel, dim=1)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        self.collision = self.check_collision()
        self.success = self.check_success()
        self.pre_success = self.success.clone()
        # check termination and reset
        if self.evaluation:
            self.crash_condition = (
                    (torch.abs(self.base_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
                |   (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
            )
        else:
            self.crash_condition = (
                (torch.abs(self.base_euler[:, 0]) > self.roll_limit_rad)
                | (torch.abs(self.base_euler[:, 1]) > self.pitch_limit_rad)
                | (torch.abs(self.base_euler[:, 2]) > self.yaw_limit_rad)
                | (torch.abs(self.base_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
                |   (torch.abs(self.base_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
                |   (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
            )

        self.reset_buf = (self.episode_length_buf > self.max_episode_length_per_env) | self.crash_condition | self.success | self.collision

        if self.evaluation:
            self.pre_crash_condition = self.crash_condition.clone()
            self.pre_collision = self.collision.clone()

        time_out_idx = (self.episode_length_buf > self.max_episode_length_per_env).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=torch.float32)
        self.extras["time_outs"][time_out_idx] = 1.0
        # Print true or false if the env is 2 steps old
        self.depth_vision()

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name] * self.dt * 100 / 2
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.last_actions[:] = self.actions[:]
        # ----------- slice depth per actor -----------------
        if self.use_wide:
            s, e = self._center_start, self._center_start + self.NUM_SECTORS_ACTOR
            depth_actor  = self.depth[:, s:e]                               # (B,20)
            depth_extra  = torch.cat([self.depth[:, :s], self.depth[:, e:]], dim=1)  # (B,40)
        else:
            depth_actor  = self.depth                                       # (B,20)
            depth_extra  = None                                             # per _make_critic_obs

        obs = self._organize_observations(depth_actor)
        obs = self._add_gaussian_obs_noise(obs)
        self.obs_buf = obs
        # ---------------- privileged: actor + 40 extra ----------------
        if self.use_wide:
            self.extras["observations"]["critic"] = self._make_critic_obs(depth_extra)

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _organize_observations(self, depth_actor: torch.Tensor) -> torch.Tensor:
        """
        Build the observation vector in one place.
        Does NOT add noise; noise is handled by _add_gaussian_obs_noise.
        Expected layout (matches your original step code):
        [ z_norm(1) |
            quat(4) |
            lin_vel_norm(3: vx/20, vy/10, vz/10) |
            depth_feat(NUM_SECTORS_ACTOR) |
            last_throttle(1) |
            last_joints_norm(num_actions-1) |
            v_tgt_norm(1) ]
        """
        # Base scalars and vectors
        z_norm   = (self.base_pos[:, 2].unsqueeze(1) - 10.0) / 10.0
        quat     = self.base_quat
        vx_norm  = self.base_lin_vel[:, 0].unsqueeze(1) / 20.0
        vy_norm  = self.base_lin_vel[:, 1].unsqueeze(1) / 10.0
        vz_norm  = self.base_lin_vel[:, 2].unsqueeze(1) / 10.0

        # Depth (near = 1, far = 0), exactly as in your code
        depth_feat = 1.0 - depth_actor / self.MAX_DISTANCE

        # Previous actions (same normalization as in your code)
        last_thr   = self.last_actions[:, 0].unsqueeze(1)                 # in [0, 1]
        last_jnts  = self.last_actions[:, 1:] / self.joint_limits[1]      # in [-1, 1] roughly

        # Commanded forward speed (normalized like your code)
        v_tgt_norm = self.commands[:, 2].unsqueeze(1) / 20.0

        obs = torch.cat([
            z_norm,                 # 1
            quat,                   # 4
            vx_norm, vy_norm, vz_norm,  # 3
            depth_feat,             # NUM_SECTORS_ACTOR
            last_thr,               # 1
            last_jnts,              # num_actions - 1
            v_tgt_norm,             # 1
        ], dim=1)

        # Sanity check: keep config consistent with the built vector
        if obs.shape[1] != self.num_obs:
            raise RuntimeError(f"num_obs mismatch: got {obs.shape[1]}, expected {self.num_obs}")

        return obs

    def _add_gaussian_obs_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Multiplicative noise per feature:
        obs *= (1 + N(0, std_feature))
        Le std provengono da obs_cfg["noise_std"] con chiavi:
        {"z","quat","vel","depth","last_thr","last_jnts","v_tgt"} (mancanti => 0.0)
        """
        if not self.obs_cfg.get("add_noise", False):
            return obs

        ns = self.obs_cfg.get("noise_std", {})
        get = lambda k: float(ns.get(k, 0.0))

        B = obs.shape[0]
        i = 0

        # z (1)
        if get("z") != 0.0:
            obs[:, i:i+1] *= (1.0 + torch.randn((B, 1), device=self.device) * get("z"))
        i += 1

        # quat (4)
        if get("quat") != 0.0:
            obs[:, i:i+4] *= (1.0 + torch.randn((B, 4), device=self.device) * get("quat"))
        i += 4

        # vel (3)
        if get("vel") != 0.0:
            obs[:, i:i+3] *= (1.0 + torch.randn((B, 3), device=self.device) * get("vel"))
        i += 3

        n_depth = self.NUM_SECTORS_ACTOR
        if get("depth") != 0.0:
            depth_feat = obs[:, i:i+n_depth]            # vicino=1, lontano=0
            scale = 4.0 * (1.0 - depth_feat)      # 1→5 quando si va da vicino a lontano
            std = get("depth") * scale                   # std: base→5× alla max distance
            obs[:, i:i+n_depth] *= (1.0 + torch.randn((B, n_depth), device=self.device) * std)
        i += n_depth

        return obs


    def _extract_thrust_tensor(self):
        """
        Converte self.rigid_solver._thr_flt in un torch.Tensor 2D [B, n_prop]
        sul device dell'env. Gestisce i casi:
        • già Tensor
        • ScalarField Genesis (ha .to_torch)
        • lista / np.ndarray / scalare
        """
        src = self.rigid_solver._thr_flt
        if isinstance(src, torch.Tensor):
            thr = src.to(self.device)
        elif hasattr(src, "to_torch"):
            thr = src.to_torch(device=self.device)
        else:
            thr = torch.as_tensor(src, dtype=torch.float32, device=self.device)

        # normalizza a 2D
        if thr.ndim == 0:                    # singolo valore ⇒ broadcast
            thr = thr.repeat(self.num_envs).unsqueeze(1)          # [B,1]
        elif thr.ndim == 1:                  # [B] ⇒ [B,1]
            thr = thr.unsqueeze(1)
        elif thr.ndim > 2:                   # schiaccia dalle colonne in poi
            thr = thr.view(self.num_envs, -1)

        return thr.contiguous()


    def check_collision(self, tol=0.05):
        """
        Collisione con alberi modellando il drone in pianta (XY) come un
        rettangolo OBB (lunghezza fusoliera + apertura alare).
        - base_pos è il NASO del drone.
        - full_len: lunghezza fusoliera dietro al naso.
        - half_span: semi-apertura alare.
        Il rettangolo viene ruotato solo di yaw.
        L’effettiva semi-apertura a terra è half_span*|cos(roll)|.
        """
        half_span = self.span / 2.0
        r_tree    = self.env_cfg.get("tree_radius", 1.0) + tol

        # ---------- coordinate droni/alberi ------------------------------------
        cyl_xy   = self.cylinders_array[self.forest_ids, :, :2]              # (E,T,2)
        drone_xy = self.base_pos[:, :2].unsqueeze(1)                         # (E,1,2)
        diff_xy  = cyl_xy - drone_xy                                         # (E,T,2)

        # ---------- ruota diff_xy nel body-frame solo con lo yaw --------------
        yaw  = self.base_euler[:, 2].unsqueeze(1)      # (E,1) rad
        roll = self.base_euler[:, 0].abs()             # (E,)
        cos_y = torch.cos(yaw)                      # (E,1)  << rimuovi extra unsqueeze
        sin_y = torch.sin(yaw)                      # (E,1)

        x_b =  diff_xy[..., 0] * cos_y + diff_xy[..., 1] * sin_y   # (E,T)
        y_b = -diff_xy[..., 0] * sin_y + diff_xy[..., 1] * cos_y   # (E,T)

        # ---------- apertura alare proiettata (dipende dal roll) -------------
        roll     = roll.abs()                   # (E,)
        half_span_eff = half_span * torch.cos(roll)                          # (E,)
        half_span_eff = half_span_eff.unsqueeze(1)                           # (E,1)

        # ---------- test rettangolo espanso -----------------------------------
        hit_x = (x_b >= -r_tree) & (x_b <= r_tree)
        hit_y = (y_b.abs() <= half_span_eff + r_tree)

        # ---------- risultato finale ------------------------------------------
        return (hit_x & hit_y).any(dim=1)


    def check_success(self):
        """
        Definisce il successo come il caso in cui la componente x della posizione base del drone supera un certo 
        limite (che rappresenta il limite lungo l'asse x della foresta). Tale limite viene preso dalla configurazione
        sotto la chiave "forest_x_limit" (default = 200 se non indicato).
        
        Restituisce un tensore booleano di shape (num_envs,).
        """
        forest_x_limit = self.env_cfg.get("forest_x_limit", 250)
        if self.evaluation:
            forest_x_limit = self.env_cfg.get("x_upper", 200)
        # L'environment ha successo se, per ciascun ambiente, la x del drone supera il limite.
        success_flags = (self.base_pos[:, 0] > forest_x_limit)
        
        return success_flags

    def depth_vision(self):
        """
        Calcola la distanza minima per settore includendo:
        • tronchi (ostacoli puntuali)
        • pareti laterali y = y_lower / y_upper
        """
        # ------------------------------------------------------------------
        E, S = self.num_envs, self.NUM_SECTORS
        default_d = self.MAX_DISTANCE
        max_d     = self.MAX_DISTANCE
        short_r   = self.SHORT_RANGE
        half_cone_nom = math.radians(self.CONE_ANGLE / 2)
        r_tree = self.env_cfg.get("tree_radius", 1.0)

        # se non ci sono alberi, riempi solo con la distanza ai muri
        if self.env_cfg.get("num_trees", 100) == 0:
            self.depth = torch.full((E, S), default_d, device=self.device)
            return

        # ------------------- coordinate alberi ----------------------------
        pos_xy = self.base_pos[:, :2].unsqueeze(1)                 # (E,1,2)
        cyl_xy = self.cylinders_array[self.forest_ids, :, :2]      # (E,T,2)
        diff   = cyl_xy - pos_xy                                   # (E,T,2)

        # ------------------- orientamento drone --------------------------
        roll = self.base_euler[:, 0]                               # (E,)
        yaw  = self.base_euler[:, 2].unsqueeze(1)                  # (E,1)

        cos_r = torch.cos(roll).clamp(min=1e-3)
        half_cone_eff = half_cone_nom * cos_r                      # (E,)
        sector_width  = (2 * half_cone_eff) / S                    # (E,)

        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)          # (E,1)

        # tronchi nel body-frame
        x_loc = diff[..., 0]*cos_yaw + diff[..., 1]*sin_yaw        # (E,T)
        y_loc = -diff[..., 0]*sin_yaw + diff[..., 1]*cos_yaw       # (E,T)

        r      = torch.sqrt(x_loc**2 + y_loc**2)
        r_surf = (r - r_tree).clamp(min=0.0)
        delta  = torch.asin(torch.clamp(r_tree / (r + 1e-6), max=1.0))
        theta  = torch.atan2(y_loc, x_loc)                         # (E,T)

        hc = half_cone_eff.unsqueeze(1)                            # (E,1)
        sw = sector_width.unsqueeze(1).clamp(min=1e-6)             # (E,1)

        theta_left  = theta - delta
        theta_right = theta + delta

        in_cone = (theta >= -hc) & (theta <= hc) & (r <= max_d)
        near    = (r <= short_r) & (x_loc >= 0.0)
        valid   = in_cone | near                                   # (E,T)

        idx_left  = torch.floor((theta_left  + hc) / sw).clamp(0, S-1).to(torch.int64)
        idx_right = torch.floor((theta_right + hc) / sw).clamp(0, S-1).to(torch.int64)

        # ------------------- distanza min per settore (tronchi) ----------
        dists = torch.full((E, S), default_d, device=self.device)  # (E,S)
        for sec in range(S):                                       # NB: sec, NON “s”
            mask_sec = valid & (idx_left <= sec) & (idx_right >= sec)
            r_sec    = torch.where(mask_sec, r_surf,
                                torch.full_like(r_surf, default_d))
            dists[:, sec] = torch.min(r_sec, dim=1).values

        # ==================================================================
        # ------------------- muri laterali --------------------------------
        y_min = self.env_cfg.get("y_lower", -50.0)
        y_max = self.env_cfg.get("y_upper",  50.0)

        dir_b = self.ray_dirs_body[:, :2]                          # (S,2)

        # ruota body→world con lo yaw
        dir_wx =  cos_yaw * dir_b[None, :, 0] - sin_yaw * dir_b[None, :, 1]  # (E,S)
        dir_wy =  sin_yaw * dir_b[None, :, 0] + cos_yaw * dir_b[None, :, 1]  # (E,S)

        wy = dir_wy.clamp(min=-0.999, max=0.999)                   # evita 0

        p_y = self.base_pos[:, 1].unsqueeze(1)                     # (E,1)

        t_left  = torch.where(wy < 0,
                            (y_min - p_y) / wy,
                            torch.full_like(wy, default_d))
        t_right = torch.where(wy > 0,
                            (y_max - p_y) / wy,
                            torch.full_like(wy, default_d))

        wall_d = torch.minimum(t_left, t_right).clamp(min=0.0, max=max_d)  # (E,S)

        # ------------------- profondità finale ---------------------------
        self.depth[:] = torch.minimum(dists, wall_d)                # (E,S)


    def get_observations(self):
        # restituisce (obs, extras) come richiesto da rsl-rl-lib ≥2.x
        infos = dict(self.extras or {})

        # preserva l’eventuale chiave "critic" già presente
        obs_dict = infos.get("observations", {})
        obs_dict["observations"] = self.obs_buf          # attore
        infos["observations"] = obs_dict

        return self.obs_buf, infos

    def get_privileged_observations(self):
        if self.use_wide:
            s = self._center_start
            e = s + self.NUM_SECTORS_ACTOR
            depth_extra = torch.cat([self.depth[:, :s],
                                     self.depth[:, e:]], dim=1)
            priv = torch.cat(
                [self.obs_buf,
                 1 - depth_extra / self.MAX_DISTANCE], dim=1)
            return priv, {}
        return self.obs_buf, {}
    
    def _make_critic_obs(self, depth_extra: torch.Tensor | None) -> torch.Tensor:
        """
        Restituisce sempre un tensor di shape (num_envs, 82):
        • se wide_depth_critic=True  ⇒ obs_buf + depth_extra  (40 valori)
        • se wide_depth_critic=False ⇒ obs_buf + zeri         (40 valori)
        """
        if self.use_wide:
            # depth_extra ha già shape (B,40)
            extra = 1.0 - depth_extra / self.MAX_DISTANCE
        else:
            return torch.cat([self.obs_buf], dim=1)
        return torch.cat([self.obs_buf, extra], dim=1)          # (B,82)
    
    def power_consumption(self):
        # ------------------------------------------------------------------ #
        #  INPUTS                                                            #
        # ------------------------------------------------------------------ #
        # --- estrai _thr_flt come (B, n_prop) torch.Tensor -----------------------
        thr   = self._extract_thrust_tensor()                          # (B, n_prop)

        # --- porta max_thrust su torch e broadcast ------------------------------
        mt = self.rigid_solver.max_thrust
        if hasattr(mt, "to_torch"):         # è un field Taichi
            mt = mt.to_torch(device=self.device)          # (B,) o (1,)
            if mt.ndim == 1:                              # → (B,1)
                mt = mt.unsqueeze(1)
        elif not torch.is_tensor(mt):                     # costante python
            mt = torch.tensor(float(mt), device=self.device).view(1, 1)

        thrust = thr * mt                                 # (B, n_prop)
        torque      = self.torque               # (num_envs, num_servos)
        joint_vel   = self.joint_velocity       # (num_envs, num_servos)

        # ------------------------------------------------------------------ #
        #  MULTIPLICATORI DI TORQUE (solo se NON è il bix3)                  #
        # ------------------------------------------------------------------ #
        sweep_multiplier = self.rigid_solver.sweep_multi
        twist_multiplier = self.rigid_solver.twist_multi
        tail_multiplier  = 2.0                  # 2 sweep, 2 twist, 2 tail  → totale 6

        # vettore [2.0, 2.0, 2.5, 2.5, 2.0, 2.0]
        multipliers = torch.tensor(
            [sweep_multiplier, sweep_multiplier,
            twist_multiplier, twist_multiplier,
            tail_multiplier,  tail_multiplier],
            device=self.device
        )

        # ------------------------------------------------------------------ #
        #  COEFFICIENTI PROPULSORI                                           #
        # ------------------------------------------------------------------ #
        if self.drone_name == "bix3":
            propellers_coeff_thrust_to_power = torch.tensor(
                [[0.0, 22.36, 2.367]], device=self.device
            )
            # parametri servocomandi per il bix3
            servomotor_power_constants = torch.tensor(
                [[3.4, 2.15, 0.35],
                [3.4, 2.15, 0.35],
                [3.4, 2.15, 0.35],
                [3.4, 2.15, 0.35]],
                device=self.device
            )
        else:
            # scala il torque con i moltiplicatori dedicati

            propellers_coeff_thrust_to_power = torch.tensor(
                [[0.0, 15.053, 2.431]], device=self.device
            )

            # 2 × KST X10 V8.0  +  4 × KST X08 Plus V2.0
            constants_x10 = torch.tensor([2.80, 1.25, 0.35], device=self.device)  # Ω, kV, kI
            constants_x08 = torch.tensor([8.84, 1.39, 0.55], device=self.device)

            servomotor_power_constants = torch.vstack((
                constants_x10.repeat(2, 1),     # primi 2 servocomandi
                constants_x08.repeat(4, 1)      # restanti 4
            ))

        # ------------------------------------------------------------------ #
        #  POTENZA DEI PROPULSORI                                            #
        # ------------------------------------------------------------------ #
        cp_each = (propellers_coeff_thrust_to_power[:, 0]
                + propellers_coeff_thrust_to_power[:, 1] * thrust
                + propellers_coeff_thrust_to_power[:, 2] * thrust**2)

        # totale → shape (B, 1) così combacia con consumption[:, :1]
        cp = cp_each
        # ------------------------------------------------------------------ #
        #  POTENZA DEI SERVOCOMANDI                                          #
        # ------------------------------------------------------------------ #
        R_const = servomotor_power_constants[:, 0][None, :]  # (1, num_servos)
        kV      = servomotor_power_constants[:, 1][None, :]
        kI      = servomotor_power_constants[:, 2][None, :]

        T = torque / multipliers[None, :]  # (num_envs, num_servos) moltiplica per i moltiplicatori
        V = joint_vel * multipliers[None, :]  # (num_envs, num_servos) moltiplica per i moltiplicatori
        P = (V * T) / (kV * kI) + (R_const * kV / kI) * T**2        # (num_envs, num_servos)

        Pj = torch.clamp(P, min=0)                                          # niente potenza negativa

        # ------------------------------------------------------------------ #
        #  IMBALLAGGIO RISULTATI                                             #
        # ------------------------------------------------------------------ #
        consumption = torch.zeros(
            (self.num_envs, self.THROTTLE_SIZE + self.num_servos),
            dtype=torch.float32, device=self.device
        )
        consumption[:, :self.THROTTLE_SIZE] = cp  # (num_envs, nprop)
        consumption[:, self.THROTTLE_SIZE:] = Pj

        # somme per ambiente
        self.cons_prop  = torch.sum(cp,  dim=1)     # solo propulsori
        self.cons_joint = torch.sum(Pj,  dim=1)     # solo servocomandi

        return torch.sum(consumption, dim=1)


    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self._resample_commands(envs_idx)
        num_resets = len(envs_idx)

        # scegli randomicamente fra tutte le foreste
        new_ids = torch.randint(
            low=0, high=self.cylinders_array.shape[0],
            size=(len(envs_idx),), device=self.device, dtype=torch.int64
        )
        self.forest_ids[envs_idx] = new_ids

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
            )
            self.episode_sums[key][envs_idx] = 0.0
        self.extras["episode"]["final_x"] = torch.mean(self.base_pos[envs_idx][:, 0]).item()
        self.extras["episode"]["final_y"] = torch.mean(self.base_pos[envs_idx][:, 1]).item()
        self.extras["episode"]["final_z"] = torch.mean(self.base_pos[envs_idx][:, 2]).item()

        # Add as extra how many env crashed, how many envs are successful and how many envs are in collision
        self.extras["episode"]["num_crashed"] = torch.sum(self.crash_condition[envs_idx]).item()
        self.extras["episode"]["num_success"] = torch.sum(self.success[envs_idx]).item()
        self.extras["episode"]["num_collision"] = torch.sum(self.collision[envs_idx]).item()

        # reset base
        if self.evaluation:
            self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device, dtype=torch.float32)
            self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device, dtype=torch.float32)
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        base_euler = quat_to_xyz(self.base_init_quat, rpy=True, degrees=False)
        self.base_euler[envs_idx] = base_euler.reshape(1, -1).expand(num_resets, -1)
        self.joint_position[envs_idx] = torch.zeros((num_resets, len(self.servo_dof_indices)), device=self.device, dtype=torch.float32)
        self.joint_velocity[envs_idx] = torch.zeros((num_resets, len(self.servo_dof_indices)),
                                                    device=self.device,
                                                    dtype=torch.float32)
        self.base_lin_vel[envs_idx] = torch.tensor([12.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
        self.base_ang_vel[envs_idx] = torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)

        self.base_lin_vel[envs_idx, 0] = self.commands[envs_idx, 2]

        if not self.evaluation:
            noise = gs_rand_normal(0.0, 5.0, (num_resets,), self.device)
            self.base_pos[envs_idx, 0] = torch.clamp(self.base_pos[envs_idx, 0] + noise, min=-50.0, max=-10.0)
            noise = gs_rand_normal(0.0, 10.0, (num_resets,), self.device)
            self.base_pos[envs_idx, 1] = torch.clamp(self.base_pos[envs_idx, 1] + noise, min=-40.0, max=40.0)
            noise = gs_rand_normal(0.0, 1.0, (num_resets,), self.device)
            self.base_pos[envs_idx, 2] = torch.clamp(self.base_pos[envs_idx, 2] + noise, min=6, max=14.0)
            noise = gs_rand_normal(0.0, 2.0, (num_resets,), self.device)
            self.base_lin_vel[envs_idx, 0] = torch.clamp(self.base_lin_vel[envs_idx, 0] + noise, min=5.0, max=19.0)
            noise = gs_rand_normal(0.0, 0.02, (num_resets, len(self.servo_dof_indices)), self.device)
            self.joint_position[envs_idx] += noise
            noise = gs_rand_normal(0.0, 0.1, (num_resets,), self.device)
            self.base_euler[envs_idx, 0] += noise
            noise = gs_rand_normal(0.0, 0.1, (num_resets,), self.device)
            self.base_euler[envs_idx, 1] += noise
            noise = gs_rand_normal(0.0, 0.1, (num_resets,), self.device)
            self.base_euler[envs_idx, 2] += noise
        self.base_quat[envs_idx] = xyz_to_quat(self.base_euler[envs_idx], degrees=False)
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        # Initial position is concatenation of base pos, base quat, joint position
        initial_position = torch.cat((self.base_pos[envs_idx], self.base_euler[envs_idx], self.joint_position[envs_idx]), dim=1)
        initial_velocity = torch.cat((self.base_lin_vel[envs_idx], self.base_ang_vel[envs_idx], self.joint_velocity[envs_idx]), dim=1)

        self.rigid_solver.set_dofs_position(initial_position, envs_idx=envs_idx)
        self.rigid_solver.set_dofs_velocity(initial_velocity, envs_idx=envs_idx)

        if self.rigid_solver._enable_noise:
            self.rigid_solver.randomize_aero_params(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.success[envs_idx] = False
        self.collision[envs_idx] = False
        self.crash_condition[envs_idx] = False

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # -------------------------------------------------------------
        #  1) inizializza la mappa di profondità a distanza massima
        #     (così depth_extra è consistente già al reset)
        # -------------------------------------------------------------
        self.depth.fill_(self.MAX_DISTANCE)

        zeros_extra = torch.zeros(
            (self.num_envs, self.NUM_SECTORS_CRITIC - self.NUM_SECTORS_ACTOR),
            device=self.device, dtype=torch.float32
        )
        critic_obs = torch.cat([self.obs_buf, zeros_extra], dim=1)  # (B,82)
        self.extras.setdefault("observations", {})
        self.extras["observations"]["critic"] = critic_obs

        # L’interfaccia di rsl‑rl ≥2.x vuole (obs, infos)
        return self.obs_buf, self.extras

    # ------------ reward functions----------------
    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_angular(self):
        angular_rew = torch.norm(self.base_ang_vel, dim=1)
        return angular_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        crash_rew[self.crash_condition | self.collision] = 1
        return crash_rew / (self.dt * 100)
    
    def _reward_energy(self):
        energy_rew = self.power_consumption() / (10 * self.dt) # (self.base_lin_vel[:, 0].clamp(min=3.0) * self.dt)
        return energy_rew

    def _reward_progress(self, sigma: float = 0.15):
        """
        Se disable_vcmd=True → usa la versione *lineare* (quella commentata nel tuo codice),
        cioè reward ∝ componente della velocità lungo la direzione comandata.
        Altrimenti usa la versione gaussiana centrata su v_cmd attuale.
        """
        # direzione desiderata (commands[:,0] in radianti)
        angles = self.commands[:, 0]
        d_unit = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)  # (E,2)
        v_xy   = self.base_lin_vel[:, :2]
        v_proj = torch.sum(v_xy * d_unit, dim=1)                             # (E,)
        # ========== versione corrente: gaussiana su v_cmd ==========
        v_tgt  = self.commands[:, 2].clamp(min=1e-3)
        x      = v_proj / v_tgt
        return torch.exp(-0.5 * ((x - 1.0) / sigma) ** 2)

        
    def _reward_obstacle(self):
        """
        Reward basato sulla prossimità agli ostacoli,
        ricavando x_loc e y_loc da self.depth e dai versori self.ray_dirs_body.
        """
        # self.depth: (E, S), self.ray_dirs_body: (S, 3)
        # estrai le componenti x e y dei versori (shape S)
        x_dirs = self.ray_dirs_body[:, 0].unsqueeze(0)  # (1, S)
        y_dirs = self.ray_dirs_body[:, 1].unsqueeze(0)  # (1, S)

        # ricostruisci le distanze x_loc e y_loc in body-frame (E, S)
        x_loc = self.depth * x_dirs
        y_loc = self.depth * y_dirs

        # distanza anisotropa e reward
        anisotropic_distance = torch.sqrt(x_loc**2 + 24.0 * y_loc**2) / 5.0
        alpha = 1.5
        clipped = torch.clamp_min(anisotropic_distance, 0.5)
        r_obs = torch.sum(torch.exp(-alpha * clipped), dim=1)
        return r_obs

    
    def _reward_height(self):
        # Penalizza l'altezza del drone
        dist = torch.square(self.base_pos[:, 2] - self.commands[:, 1])
        height = torch.zeros_like(dist, device=self.device, dtype=torch.float32)
        height[self.base_pos[:, 2]<5] = dist[self.base_pos[:, 2]<5]

        return height
    
    def _reward_success(self):
        # Penalizza il successo
        success_rew = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        success_rew[self.success] = 1
        return success_rew
    
    def _reward_cosmetic(self):
        # Reward cosmetico per di simmetria dei giunti 0-1 e 2-3
        cosmetic_rew = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        if self.drone_name == "bix3":
            cosmetic_rew += torch.square(self.joint_position[:, 0] - self.joint_position[:, 1])
            cosmetic_rew += torch.square(self.joint_position[:, 2] - self.joint_position[:, 3])
        else:
            cosmetic_rew += torch.square(self.joint_position[:, 0] + self.joint_position[:, 1])
            cosmetic_rew += torch.square(self.joint_position[:, 2] - self.joint_position[:, 3])
            #cosmetic_rew += torch.square(self.joint_position[:, 4])
            cosmetic_rew += torch.square(self.joint_position[:, 5])

        return cosmetic_rew
    
def main():
    # Configurazione dell'environment
    num_envs = 20
    env_cfg = {
        "num_actions": 0,             # ad es. thrust + controllo PD delle ali
        "simulate_action_latency": False,
        "episode_length_s": 10.0,
        "max_visualize_FPS": 60,
        "visualize_camera": True,
        "base_init_pos": [0.0, 0.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "clip_actions": 1.0,
        "termination_if_pitch_greater_than": 45.0,
        "termination_if_roll_greater_than": 45.0,
        "termination_if_x_greater_than": 100.0,
        "termination_if_y_greater_than": 100.0,
        "termination_if_z_greater_than": 100.0,
        "termination_if_close_to_ground": 0.1,
        "num_trees": 50,
        "tree_radius": 1.0,
        "tree_height": 50.0,
    }
    obs_cfg = {
        "num_obs": 14,  # aggiornare in base alle osservazioni concatenate
        "obs_scales": {
            "rel_pos": 1.0,
            "lin_vel": 1.0,
            "ang_vel": 1.0,
        },
    }
    reward_cfg = {
        "reward_scales": {
            "smooth": 0.1,
            "angular": 0.1,
            "crash": 1.0,
            "energy": 0.01,
            "obstacle": 0.0,
        }
    }
    command_cfg = {
        "num_commands": 3,  # ad esempio una posizione target 3D
    }

    gs.init(logging_level="error")
    # Istanzia l'environment con visualizzazione abilitata
    env = WingedDroneEnv(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True, device="cuda")
    
    # Esegui un reset per posizionare correttamente drone, foresta e pavimento
    env.reset()
    
    print("Ambiente iniziale creato e immobile.\n"
          "Verifica la scena nel viewer. Premi Ctrl+C per terminare.")

    try:
        # Mantiene il programma in esecuzione senza aggiornare la simulazione,
        # così da mostrare lo stato statico dell'environment.
        while True:
            env.step(env.actions)
    except KeyboardInterrupt:
        print("Chiusura della simulazione.")

if __name__ == "__main__":
    main()