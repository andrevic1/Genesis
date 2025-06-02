import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat, quat_to_R, xyz_to_quat
from morphing_drone.aer_solver_class import AerodynamicSolver
from bix3.aerodyn_solver import compute_aero_wrench as bix3_physics
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

    BASE_OBS_SIZE = 9          # z-pos(1)+quat(4)+lin_vel_norm(1)
    THROTTLE_SIZE = 1
    NUM_SECTORS = 15
    MAX_DISTANCE = 50.0
    SHORT_RANGE = 3.0
    CONE_ANGLE = 60.0
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, 
                 show_viewer=False, eval=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.rendered_env_num = self.num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.evaluation = eval
        self.unique_forests_eval = env_cfg.get("unique_forests_eval", False)
        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = env_cfg.get('dt', 0.01)
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        self.drone_name = env_cfg["drone"]
        self.show_viewer = show_viewer

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(-35.0, 0.0, 25.0),  # iniziale, poi seguiamo
                camera_lookat=(-28.0, 0.0, 20.0),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.rendered_env_num))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            #renderer=gs.renderers.Rasterizer(),
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # drone entity
        self.base_init_pos = torch.tensor(env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        base_pos_np = self.base_init_pos.cpu().numpy()
        base_quat_np = self.base_init_quat.cpu().numpy()

       # camera
        self.camera_res = env_cfg.get("camera_res", (640, 480))
        self.camera_pos = env_cfg.get("camera_pos", (-4.0, 0.0, 3.0))
        self.camera_lookat = env_cfg.get("camera_lookat", (1.5, 0.0, 0.0))
        self.camera_fov = env_cfg.get("camera_fov", 30)

        self.top_cam = None
        if self.evaluation and self.show_viewer:
            self.top_cam = self.scene.add_camera(
                res=self.camera_res,
                pos=tuple(self.base_init_pos.cpu().numpy() + self.camera_pos),
                lookat=tuple(self.base_init_pos.cpu().numpy()),
                fov=self.camera_fov,
                GUI=True,
            )

        if self.drone_name == "morphing_drone":
            urdf_file = "/home/andrea/Documents/genesis/Genesis/genesis/assets/urdf/mydrone/mydrone_new.urdf"
            self.drone = self.scene.add_entity(gs.morphs.URDF(
                file=urdf_file,
                pos=base_pos_np, quat=base_quat_np,
                collision=True, merge_fixed_links=True,
                links_to_keep=[
                    "aero_frame_fuselage","aero_frame_left_wing","aero_frame_right_wing",
                    "elevator","rudder","aero_frame_elevator", "aero_frame_rudder",
                    "prop_frame_fuselage_0",
                    "fuselage","left_wing","right_wing",
                ],
            ))
            servo_joint_names = [
                "joint_0_twist_left_wing",
                "joint_0_twist_right_wing",
                "elevator_pitch_joint",
                "rudder_yaw_joint",
            ]

            # ④ ­– link su cui applicheremo le forze aerodinamiche
            self.link_name_array = [
                "fuselage",          # per “total” wrench
                "left_wing",
                "right_wing",
                "elevator",
                "rudder",
                "prop_frame_fuselage_0",
            ]
            self.solver = AerodynamicSolver(device=self.device, urdf_file=urdf_file) 

        elif self.drone_name == "bix3":
            self.drone = self.scene.add_entity(gs.morphs.URDF(
                file='urdf/mydrone/bix3.urdf',
                pos=base_pos_np, quat=base_quat_np,
                collision=True, merge_fixed_links=True,
                links_to_keep=[
                    'fuselage','aero_frame_fuselage',
                    'right_aileron_wing','left_aileron_wing',
                    'fuselage_rudder_0','rudder_wing',
                    'fuselage_elevator_0','elevator_wing',
                    'prop_frame_fuselage',
                ]
            ))
            servo_joint_names= [
                'joint_right_aileron','joint_left_aileron',
                'joint_0__rudder_wing','joint_0__elevator_wing'
            ]
            self.link_name_array = ["fuselage"]
        else:
            print("No drone dio boia")

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
        self.num_obs      = (
            self.BASE_OBS_SIZE          # 13
            + self.num_servos           # joint position
            + self.NUM_SECTORS           # lidar frontale
            + self.num_actions          # last_actions
            #+ self.num_commands         # commands
        )

        angles = torch.linspace(
            -self.CONE_ANGLE/2 * math.pi/180,
            self.CONE_ANGLE/2 * math.pi/180,
            self.NUM_SECTORS,
            device=self.device
        )

        self.ray_dirs_body = torch.stack([
            torch.cos(angles),
            torch.sin(angles),
            torch.zeros_like(angles)
        ], dim=1)

        # metti in coerenza i dizionari di config (utile per il logging)
        env_cfg["num_actions"] = self.num_actions
        obs_cfg["num_obs"]     = self.num_obs

        # Set servo control parameters (PD gains, force range, etc.)
        self.drone.set_dofs_kp(torch.full((self.num_servos,),1000.0,device=self.device), self.servo_dof_indices)
        self.drone.set_dofs_kv(torch.full((self.num_servos,),  100.0,device=self.device), self.servo_dof_indices)

        max_tau = 100.0  # [Nm], scegli in base ai carichi aerodinamici
        self.drone.set_dofs_force_range(
            upper=torch.full((self.num_servos,),  max_tau, device=self.device),
            lower=torch.full((self.num_servos,), -max_tau, device=self.device),
            dofs_idx_local=self.servo_dof_indices
        )

        self.link_array = [self.drone.get_link(link_name) for link_name in self.link_name_array]
        self.link_idx_array = [link.idx_local for link in self.link_array]
        self.force_array = torch.zeros((self.num_envs, len(self.link_name_array), 3), dtype=torch.float32)
        self.torque_array = torch.zeros((self.num_envs, len(self.link_name_array), 3), dtype=torch.float32)
        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.success = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.pre_success = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.collision = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.crash_condition = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.pre_crash_condition = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.pre_collision = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)
        self.joint_position = torch.zeros((self.num_envs, self.num_servos), device=self.device, dtype=gs.tc_float)
        self.joint_velocity = torch.zeros((self.num_envs, self.num_servos), device=self.device, dtype=gs.tc_float)

        self.depth = torch.empty(
            (self.num_envs, self.NUM_SECTORS),
            device=self.device, dtype=torch.float16
        )

        self.rigid_solver = None
        for solver in self.scene.sim.solvers:
            if isinstance(solver, RigidSolver):
                self.rigid_solver = solver
                break
        if self.rigid_solver is None:
            raise Exception("RigidSolver not found!")

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

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
            mass0 = link.get_mass()                      # valore attuale
            ratio  = 1.0 + (rng(1, device=self.device)*2 - 1) * dm
            link.set_mass(mass0 * ratio.item())          # API corretta


    def _generate_forests(self):
        num_trees = self.env_cfg.get("num_trees", 100)
        tree_radius = self.env_cfg.get("tree_radius", 1.0)
        tree_height = self.env_cfg.get("tree_height", 50.0)
        x_lower, x_upper = self.env_cfg.get("x_lower", 0), self.env_cfg.get("x_upper", 70)
        y_lower, y_upper = self.env_cfg.get("y_lower", -50), self.env_cfg.get("y_upper", 50)

        if self.evaluation:
            x_upper = 200
            num_trees = self.env_cfg.get("num_trees_eval", 20)

        if self.evaluation:
            if self.unique_forests_eval:
                total_forests = self.num_envs * 10
            else:
                total_forests = 1
        else:
            total_forests = self.num_envs * 100

        self.total_forests = total_forests
        # vectorized creation
        if self.evaluation and not self.unique_forests_eval:
            cylinders = torch.zeros((total_forests, num_trees, 3), dtype=torch.float32, device=self.device)
        else:
            cylinders = torch.zeros((total_forests, num_trees, 3), dtype=torch.float16, device=self.device)
        xs = gs_rand_float(x_lower, x_upper, (total_forests, num_trees), self.device)
        ys = gs_rand_float(y_lower, y_upper, (total_forests, num_trees), self.device)
        cylinders[..., 0] = xs
        cylinders[..., 1] = ys
        cylinders[..., 2].fill_(tree_height * 0.5)

        if self.evaluation and not self.unique_forests_eval:
            # add to the scene the cylinders of the first forest
            for i in range(num_trees):
                self.scene.add_entity(gs.morphs.Cylinder(
                    pos=cylinders[0, i, :].cpu().numpy(),
                    radius=tree_radius,
                    height=tree_height,
                    collision=False,fixed=True
                ))

        self.cylinders_array = cylinders

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = 0.0

        if self.evaluation:
            min_z = self.command_cfg.get("min_z_target", 10.0)
            max_z = self.command_cfg.get("max_z_target", 10.0)
        else:
            min_z = self.command_cfg.get("min_z_target", 10.0)
            max_z = self.command_cfg.get("max_z_target", 10.0)

        z_tgt  = gs_rand_float(min_z,     max_z,  (len(envs_idx),), self.device)

        self.commands[envs_idx, 1] = z_tgt           # target height
        return
    

    def set_angle_limit(self, limit_deg: float):
        """Aggiorna dinamicamente i limiti usati per la crash-condition."""
        self.curr_limit = limit_deg                      # salvo per logging
        # α e β sono già in gradi → basta cambiare le soglie
        self.env_cfg["termination_if_alpha_greater_than"] = limit_deg
        self.env_cfg["termination_if_beta_greater_than"]  = limit_deg
        # per il roll lavoriamo in radianti, quindi convertiamo internamente
        self.roll_limit_rad = math.radians(limit_deg)


    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        # 1) clamp throttle a [0,1]
        self.actions[:,0] = torch.clamp(
            self.actions[:,0], 
            self.throttle_limit[0], 
            self.throttle_limit[1]
        )
        # 2) clamp joint pos a joint_limits
        self.actions[:,1:] = torch.clamp(
            self.actions[:,1:], 
            self.joint_limits[0], 
            self.joint_limits[1]
        )
        # 3) limit rate of change rispetto a last_actions
        delta = self.actions - self.last_actions
        delta = torch.clamp(delta, -0.05, 0.05)
        self.actions = self.last_actions + delta
                
        #self.actions[:,1:] = torch.tensor([-0.27, -0.145])
        #self.actions[:, 0] = 0.30
        #self.actions = torch.zeros_like(self.actions)
        if self.drone_name == "bix3":
            exec_servo = torch.stack([
            (self.actions[:,1]),     # δa destro
            -(self.actions[:,1]),    # δa sinistro
            self.actions[:,2],           # δr
            self.actions[:,3],           # δe
            ], dim=1)  # shape (B,4)
            exec_throttle = self.actions[:,0]
        else:
            exec_servo = self.actions[:, 1:]
            exec_throttle = self.actions[:, 0]
            

        self.drone.control_dofs_position(exec_servo, self.servo_dof_indices)

        if self.drone_name == "morphing_drone":
            aero_details = self.solver.compute_aero_details(self.drone, exec_throttle)
            self.alpha = aero_details["fuselage"]["alpha"]
            self.beta = aero_details["fuselage"]["beta"]
            
            force  = self.force_array
            torque = self.torque_array
            force.zero_()
            torque.zero_()
            for i, name in enumerate(self.link_name_array):
                if name == "elevator":
                    # somma i due semi-elevator
                    wf_L, wm_L = aero_details["elevator_left"]["wrench"]
                    wf_R, wm_R = aero_details["elevator_right"]["wrench"]
                    wf, wm = wf_L + wf_R, wm_L + wm_R

                else:
                    wf, wm = aero_details[name]["wrench"]

                force[:,i].copy_(wf)
                torque[:,i].copy_(wm)

        elif self.drone_name == "bix3":

            wrench = bix3_physics(self.drone, exec_throttle, max_thrust=15)
            F, M, alpha, beta = wrench['F'], wrench['M'], wrench['alpha'], wrench['beta']
            self.alpha = alpha
            self.beta = beta
            force = F.unsqueeze(1)
            torque = M.unsqueeze(1)

        # ---------- aerodynamical noise ------------------------------------------- #
        if self.env_cfg.get("aero_noise", True):
            # magnitudo relativa σ = σ0 + k·(|α|+|β|)
            abs_ang = (torch.abs(self.alpha) + torch.abs(self.beta))        # (E,) deg

            sigma = (self.env_cfg.get("aero_noise_sigma0", 0.02) +
                    self.env_cfg.get("aero_noise_k",      0.3) * abs_ang)  # (E,)

            sigma = sigma.view(-1, 1, 1)                                    # (E,1,1)
            eps_f = torch.randn_like(force)  * sigma * torch.norm(force , dim=2, keepdim=True).clamp(min=1.0)
            eps_m = torch.randn_like(torque) * sigma * torch.norm(torque, dim=2, keepdim=True).clamp(min=1.0)

            force  = force  + eps_f
            torque = torque + eps_m


        # Set alpha and beta in degrees
        self.alpha = torch.rad2deg(self.alpha)
        self.beta = torch.rad2deg(self.beta)
        # Apply a gaussian noise on F and M proportional to alpha and beta
        self.rigid_solver.apply_links_external_force(force=force, links_idx=self.link_idx_array)
        self.rigid_solver.apply_links_external_torque(torque=torque, links_idx=self.link_idx_array)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1

        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_dofs_position()[:, :3]
        self.base_quat[:] = self.drone.get_quat()
        self.joint_position = self.drone.get_dofs_position()[:, -len(self.servo_dof_indices):]
        self.torque = self.drone.get_dofs_force(self.servo_dof_indices)
        self.base_euler = quat_to_xyz(
            self.base_quat,
            rpy=True, degrees=False
        )
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
                    (torch.abs(self.alpha) > self.env_cfg["termination_if_alpha_greater_than"])
                |   (torch.abs(self.beta)  > self.env_cfg["termination_if_beta_greater_than"])
                |   (torch.abs(torch.abs(self.base_euler[:, 0])) > self.roll_limit_rad)  # roll
                |   (torch.abs(self.base_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
                |   (torch.abs(self.base_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
                |   (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
            )

        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition | self.success | self.collision

        if self.evaluation:
            self.pre_crash_condition = self.crash_condition.clone()
            self.pre_collision = self.collision.clone()

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        # b) top-cam solo in eval
        if self.top_cam is not None:
            top_pos  = (self.base_pos[0] + torch.tensor(self.camera_pos, device=self.device)).cpu().numpy()
            self.top_cam.set_pose(pos=tuple(top_pos), lookat=tuple(self.base_pos[0].cpu().numpy()))


        # Print true or false if the env is 2 steps old
        self.depth_vision()

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observation
        self.obs_buf = torch.cat(
            [
                (self.base_pos[:, 1].unsqueeze(1) - 50)/50,
                (self.base_pos[:, 2].unsqueeze(1) - 10)/10,
                self.base_quat,
                self.base_lin_vel[:, 0].unsqueeze(1)/20,
                self.base_lin_vel[:, 1].unsqueeze(1)/10,
                self.base_lin_vel[:, 2].unsqueeze(1)/10,
                self.joint_position/0.35,
                (self.depth) / self.MAX_DISTANCE,
                self.last_actions,
                #self.commands[:, 0].unsqueeze(1) / math.pi,
                #(self.commands[:, 1].unsqueeze(1) - 20)/20,
            ],
            axis=-1,
        )
        self.extras["observations"]["critic"] = self.obs_buf

        #def fmt(tensor):
            # tensor: 1D tensor o lista di float
        #    return [f"{x:.2f}" for x in tensor.tolist()]
        
        #if self.evaluation and self.num_envs == 1:
            #print("Time step:        ", fmt(self.episode_length_buf))
            #print("Position:         ", fmt(self.base_pos[0]))
            #print("Velocity:         ", fmt(self.base_lin_vel[0]))
            #print("Alpha:            ", fmt(self.alpha))
            #print("Beta:             ", fmt(self.beta))
            #print("Euler:            ", fmt(self.base_euler[0]))
            #print("Depth:            ", fmt(self.depth[0]))
            #print("Actions:          ", fmt(self.actions[0]))
            #print("Joint Position:   ", fmt(self.joint_position[0]))

        self.last_actions[:] = self.actions[:]

        if self.show_viewer:
            pos0 = self.base_pos[0].cpu().numpy()
            cam_pos = pos0 + np.array(self.camera_pos)
            cam_look = pos0 + np.array(self.camera_lookat)
            self.scene.viewer.set_camera_pose(
                pos=cam_pos,
                lookat=cam_look
            )

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
        
    def check_collision(self, tol=0.02):
        """
        Collisione con alberi modellando il drone in pianta (XY) come un
        rettangolo OBB (lunghezza fusoliera + apertura alare).
        - base_pos è il NASO del drone.
        - full_len: lunghezza fusoliera dietro al naso.
        - half_span: semi-apertura alare.
        Il rettangolo viene ruotato solo di yaw.
        L’effettiva semi-apertura a terra è half_span*|cos(roll)|.
        """
        half_span = self.env_cfg.get("wing_span",        1.5) * 0.5  # m
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
        forest_x_limit = self.env_cfg.get("forest_x_limit", 70)
        if self.evaluation:
            forest_x_limit = 200
        # L'environment ha successo se, per ciascun ambiente, la x del drone supera il limite.
        success_flags = self.base_pos[:, 0] > forest_x_limit
        
        return success_flags
    
    def depth_vision(self):
        """
        Calcola la distanza minima in ogni settore frontale (lidar 2-D).
        Il campo visivo orizzontale (half-cone) si riduce con il roll:
            half_cone_eff = half_cone_nominal · cos(|roll|)
        In pratica se il drone è arrotolato di 60 ° vede la metà
        della larghezza rispetto a quando è livellato.
        """
        # --- costanti e buffer -------------------------------------------------
        E, S = self.num_envs, self.NUM_SECTORS
        default_d = self.MAX_DISTANCE
        max_d     = self.MAX_DISTANCE
        short_r   = self.SHORT_RANGE
        half_cone_nom = math.radians(self.CONE_ANGLE / 2)          # rad

        # se non ci sono alberi: tutto a default
        if self.env_cfg.get("num_trees", 100) == 0:
            self.depth = torch.full((E, S), default_d, device=self.device)
            return

        # ------------------- posa e alberi ------------------------------------
        pos_xy = self.base_pos[:, :2].unsqueeze(1)                 # (E,1,2)
        cyl_xy = self.cylinders_array[self.forest_ids, :, :2]      # (E,T,2)
        diff   = cyl_xy - pos_xy                                   # (E,T,2)

        # ------------------- orientamento drone ------------------------------
        roll = self.base_euler[:, 0]
        yaw  = self.base_euler[:, 2].unsqueeze(1)

        # FOV effettivo che dipende dal roll
        cos_r          = torch.cos(roll).clamp(min=1e-3)           # evita div/0
        half_cone_eff  = half_cone_nom * cos_r                     # (E,)
        sector_width   = (2 * half_cone_eff) / S                   # (E,)

        # ruota in body-frame (dipende dallo yaw ma non dal roll)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        x_loc = diff[...,0]*cos_y + diff[...,1]*sin_y              # (E,T)
        y_loc = -diff[...,0]*sin_y + diff[...,1]*cos_y             # (E,T)

        r     = torch.sqrt(x_loc**2 + y_loc**2)
        theta = torch.atan2(y_loc, x_loc)                          # (E,T)

        # ------------------- maschere di validità -----------------------------
        hc = half_cone_eff.unsqueeze(1)                            # (E,1)
        in_cone = (theta >= -hc) & (theta <= hc) & (r <= max_d)
        near    = (r <= short_r) & (x_loc >= 0.0)
        valid   = in_cone | near                                   # (E,T)

        # ------------------- indice di settore -------------------------------
        sw = sector_width.unsqueeze(1).clamp(min=1e-6)             # (E,1)
        sector_idx = torch.floor((theta + hc) / sw).to(torch.int64)
        sector_idx = sector_idx.clamp(0, S-1)                      # (E,T)

        # ------------------- distanza minima per settore ----------------------
        dists = torch.full((E, S), default_d, device=self.device)  # (E,S)
        for s in range(S):                                         # poche iterazioni
            mask_s = valid & (sector_idx == s)
            r_s    = torch.where(mask_s, r, torch.full_like(r, default_d))
            dists[:, s] = torch.min(r_s, dim=1).values

        self.depth[:] = dists                # (E,S)


    def get_observations(self):
        # restituisce (obs, extras) come richiesto da rsl-rl-lib ≥2.x
        infos = dict(self.extras or {})
        # OnPolicyRunner si aspetta extras["observations"] come dict
        infos["observations"] = {"observations": self.obs_buf}
        return self.obs_buf, infos

    def get_privileged_observations(self):
        return None, {}
    
    def power_consumption(self):
        
        thrust = self.actions[:,0]  # shape(num_envs, nprop)
        torque = self.torque  # shape(num_envs, ndofs)

        propellers_coeff_thrust_to_power = torch.tensor([[ 0.,    15.053,  2.431]], device=self.device)
        servomotor_power_constants = torch.tensor([[2.4489796,  0.61086524, 1.4489796 ],
            [2.4489796,  0.61086524, 1.4489796 ], [2.4489796,  0.61086524, 1.4489796 ], [2.4489796,  0.61086524, 1.4489796 ]], device=self.device)

        consumption = torch.zeros((self.num_envs, self.THROTTLE_SIZE+self.num_servos), dtype=torch.float32, device=self.device)
        coeffs_prop = propellers_coeff_thrust_to_power[None, :, :] # shape(1,nprop,3)
        # polinomio
        cp = coeffs_prop[...,0] + coeffs_prop[...,1]*thrust + coeffs_prop[...,2]*(thrust**2)

        joint_vel = self.joint_velocity
        R_const  = servomotor_power_constants[:,0][None,:]
        kV       = servomotor_power_constants[:,1][None,:]
        kI       = servomotor_power_constants[:,2][None,:]

        T  = torque
        R_kV_over_kI = R_const*kV/kI
        P = (joint_vel*T)/(kV*kI) + R_kV_over_kI*(T**2)
        Pj = torch.clamp(P, min=0)

        consumption[:,:self.THROTTLE_SIZE] = cp.T
        consumption[:,self.THROTTLE_SIZE:] = Pj

        self.cons_prop = torch.sum(cp.T, axis=1)
        self.cons_joint= torch.sum(Pj, axis=1)

        return self.cons_prop

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
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
            self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device, dtype=gs.tc_float)
            self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device, dtype=gs.tc_float)
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        base_euler = quat_to_xyz(self.base_init_quat, rpy=True, degrees=False)
        self.base_euler[envs_idx] = base_euler.reshape(1, -1).expand(num_resets, -1)
        self.joint_position[envs_idx] = torch.zeros((num_resets, len(self.servo_dof_indices)), device=self.device, dtype=gs.tc_float)
        self.joint_velocity[envs_idx] = torch.zeros((num_resets, len(self.servo_dof_indices)),
                                                    device=self.device,
                                                    dtype=gs.tc_float)
        self.base_lin_vel[envs_idx] = torch.tensor([9.0, 0.0, 0.0], device=self.device, dtype=gs.tc_float)
        self.base_ang_vel[envs_idx] = torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=gs.tc_float)

        if not self.evaluation:
            noise = gs_rand_normal(0.0, 5.0, (num_resets,), self.device)
            self.base_pos[envs_idx, 0] = torch.clamp(self.base_pos[envs_idx, 0] + noise, min=-40.0, max=-5.0)
            noise = gs_rand_normal(0.0, 15.0, (num_resets,), self.device)
            self.base_pos[envs_idx, 1] = torch.clamp(self.base_pos[envs_idx, 1] + noise, min=-47.0, max=47.0)
            noise = gs_rand_normal(0.0, 4.0, (num_resets,), self.device)
            self.base_pos[envs_idx, 2] = torch.clamp(self.base_pos[envs_idx, 2] + noise, min=3, max=19.0)
            noise = gs_rand_normal(0.0, 3.0, (num_resets,), self.device)
            self.base_lin_vel[envs_idx, 0] = torch.clamp(self.base_lin_vel[envs_idx, 0] + noise, min=4.0, max=17.0)
            noise = gs_rand_normal(0.0, 0.05, (num_resets, len(self.servo_dof_indices)), self.device)
            self.joint_position[envs_idx] += noise
            noise = gs_rand_normal(0.0, 0.2, (num_resets,), self.device)
            self.base_euler[envs_idx, 0] = torch.clamp(self.base_euler[envs_idx, 0] + noise, min=-(self.roll_limit_rad-0.2), max=(self.roll_limit_rad-0.2))
            noise = gs_rand_normal(0.0, 0.1, (num_resets,), self.device)
            self.base_euler[envs_idx, 1] += noise
            noise = gs_rand_normal(0.0, 0.1, (num_resets,), self.device)
            self.base_euler[envs_idx, 2] += noise

        self.last_base_pos[envs_idx] = self.base_pos[envs_idx]
        self.base_quat[envs_idx] = xyz_to_quat(self.base_euler[envs_idx], degrees=False)
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        # Initial position is concatenation of base pos, base quat, joint position
        initial_position = torch.cat((self.base_pos[envs_idx], self.base_euler[envs_idx], self.joint_position[envs_idx]), dim=1)
        initial_velocity = torch.cat((self.base_lin_vel[envs_idx], self.base_ang_vel[envs_idx], self.joint_velocity[envs_idx]), dim=1)

        self.rigid_solver.set_dofs_position(initial_position, envs_idx=envs_idx)
        self.rigid_solver.set_dofs_velocity(initial_velocity, envs_idx=envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.success[envs_idx] = False
        self.collision[envs_idx] = False
        self.crash_condition[envs_idx] = False
    

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_angular(self):
        angular_rew = torch.norm(self.base_ang_vel, dim=1)
        return angular_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition | self.collision] = 1
        return crash_rew
    
    def _reward_energy(self):
        energy_rew = self.power_consumption()
        return energy_rew
    
    def _reward_progress(self):
        angles   = self.commands[:, 0]
        d_unit   = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)   # (E,2)
        v_xy     = self.base_lin_vel[:, :2]                                     # (E,2)

        return torch.sum(v_xy * d_unit, dim=1)                                  # (E,)
        return vx
    
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
        height = torch.zeros_like(dist, device=self.device, dtype=gs.tc_float)
        height[self.base_pos[:, 2]<5] = dist[self.base_pos[:, 2]<5]
        return height
    
    def _reward_success(self):
        # Penalizza il successo
        success_rew = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        success_rew[self.success] = 1
        return success_rew
    
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