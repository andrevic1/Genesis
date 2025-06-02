import sys, select, termios, tty, time
import numpy as np
import matplotlib.pyplot as plt
import genesis as gs
from aerodynamic_wrench_calculation import compute_aero_details  # usa la nuova funzione
from genesis.engine.solvers import RigidSolver
from genesis.utils.geom import quat_to_xyz, quat_to_R, axis_angle_to_R, xyz_to_quat
import torch

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

def create_forest(scene):
    # Imposta parametri della foresta, prelevandoli dalla configurazione con default
    num_trees = 10
    tree_radius = 1.0
    tree_height = 50
    device = "cuda:0"
    
    # Intervalli per le posizioni in x e y
    x_lower, x_upper = 0, 200
    y_lower, y_upper = -50, 50

    # Genera posizioni casuali per gli alberi, sfruttando la funzione gs_rand_float
    x_positions = gs_rand_float(x_lower, x_upper, (num_trees,), device).cpu().numpy()
    y_positions = gs_rand_float(y_lower, y_upper, (num_trees,), device).cpu().numpy()

    for i in range(num_trees):
        pos = (float(x_positions[i]), float(y_positions[i]), tree_height / 2)

        scene.add_entity(
            gs.morphs.Cylinder(
                pos=pos,
                quat=(1.0, 0.0, 0.0, 0.0),
                radius=tree_radius,
                height=tree_height,
                collision=True,
                fixed=True,
            )
        )


# Initialize Genesis (GPU backend)
gs.init(backend=gs.gpu)

# Create a simulation scene at 100Hz (dt=0.01)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=True
)

# Add a plane (ground)
plane = scene.add_entity(gs.morphs.Plane(
    pos=(0.0, 0.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),
    collision=True,
    visualization=True,
))

create_forest(scene)

starting_euler = np.array([0.0, 0.0, 0.0])
starting_quat = xyz_to_quat(starting_euler)

# Add the drone from URDF (make sure the path is correct)
drone = scene.add_entity(gs.morphs.URDF(
    file="urdf/mydrone/mydrone.urdf",
    pos=(-20.0, 0.0, 20.0),
    quat=(starting_quat),
    collision=True,
    merge_fixed_links=True,
    links_to_keep=["aero_frame_fuselage", "aero_frame_left_wing", "aero_frame_right_wing", 
                   "prop_frame_fuselage_0", "fuselage", "left_wing", "right_wing"],
))
# Build the scene (non-batched simulation: n_envs=1)
B = 1
scene.build(n_envs=B, env_spacing=(1.0, 1.0))

# Identify servo joints (for wing twist) from URDF
servo_joint_names = ["joint_0_twist_left_wing", "joint_0_twist_right_wing"]
servo_joints = [drone.get_joint(name) for name in servo_joint_names]
servo_dof_indices = sorted([drone.get_joint(name).dof_idx_local for name in servo_joint_names])

print(drone)
print(f"Servo dof indices: {servo_dof_indices}")

# Set servo control parameters (PD gains, force range, etc.)
drone.set_dofs_kp(kp=np.array([500.0, 500.0]), dofs_idx_local=servo_dof_indices)
drone.set_dofs_kv(kv=np.array([10.0, 10.0]), dofs_idx_local=servo_dof_indices)
drone.set_dofs_force_range(upper=np.array([3.4, 3.4]), lower=np.array([-3.4, -3.4]),
                            dofs_idx_local=servo_dof_indices)

# Set initial speed (as tensor, 8 dofs in free-body and actuated joints)
initial_speed = gs.tensor([[8.0, 0.0, 0, 0, 0, 0, 0, 0]])
# Retrieve RigidSolver
rigid_solver = None
for solver in scene.sim.solvers:
    if isinstance(solver, RigidSolver):
        rigid_solver = solver
        break
if rigid_solver is None:
    raise Exception("RigidSolver not found!")

state = rigid_solver.get_state(f=None)
rigid_solver.set_dofs_velocity(initial_speed)

# Set up keyboard input (via termios)
stdin_fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(stdin_fd)
tty.setcbreak(stdin_fd)

# Control variables: throttle and servo target angles (in radians)
throttle = 0.0
target_angles = [-0.1, -0.1]

angle_step = np.deg2rad(0.5)

# Variables for logging
time_log = []
pos_log = []      # Drone (fuselage) position (x, y, z)
rpy_log = []      # Euler angles (roll, pitch, yaw)
thrust_log = []   # Thrust value (derived from throttle)
joint_log = []    # Servo joint positions
alpha_log = []
beta_log = []

# Mapping aerodynamic element keys to link names for debugging:
link_mapping = {
    "fuselage": "aero_frame_fuselage",
    "elevator": "aero_frame_fuselage",
    "rudder":   "aero_frame_fuselage",
    "left_wing": "aero_frame_left_wing",
    "right_wing": "aero_frame_right_wing",
    "propeller": "prop_frame_fuselage_0",
    # "total": None   # non usato per debug arrow
}

print("Simulation started. Use Up/Down arrows for throttle, Left/Right arrows for servo angle. Press Ctrl+C to quit.")
sim_time = 0.0  # simulated time

try:
    while True:
        # Keyboard input processing using termios/select
        if select.select([sys.stdin], [], [], 0.0)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                seq = ch + sys.stdin.read(2)
                if seq == '\x1b[A':  # Up arrow: increase throttle
                    throttle = min(1.0, throttle + 0.025)
                    print(f"Throttle increased to {throttle:.2f}")
                elif seq == '\x1b[B':  # Down arrow: decrease throttle
                    throttle = max(0.0, throttle - 0.025)
                    print(f"Throttle decreased to {throttle:.2f}")
                elif seq == '\x1b[C':  # Right arrow: increase servo angles
                    target_angles[0] += angle_step
                    target_angles[1] += angle_step
                    target_angles = np.clip(target_angles, -1, 1)
                    print(f"Joint targets increased to {np.degrees(target_angles[0]):.1f}° (both)")
                elif seq == '\x1b[D':  # Left arrow: decrease servo angles
                    target_angles[0] -= angle_step
                    target_angles[1] -= angle_step
                    target_angles = np.clip(target_angles, -1, 1)
                    print(f"Joint targets decreased to {np.degrees(target_angles[0]):.1f}° (both)")
            elif ch == 'a':
                # 'a': move servos in opposite directions (increase left, decrease right)
                target_angles[0] += angle_step
                target_angles[1] -= angle_step
                target_angles = np.clip(target_angles, -1, 1)
                print(f"'A' pressed: left joint -> {np.degrees(target_angles[0]):.1f}°, right joint -> {np.degrees(target_angles[1]):.1f}°")
            elif ch == 'd':
                # 'd': move servos in opposite directions (decrease left, increase right)
                target_angles[0] -= angle_step
                target_angles[1] += angle_step
                target_angles = np.clip(target_angles, -1, 1)
                print(f"'D' pressed: left joint -> {np.degrees(target_angles[0]):.1f}°, right joint -> {np.degrees(target_angles[1]):.1f}°")
        
        # Invia i comandi di controllo in posizione ai giunti
        drone.control_dofs_position(np.array([target_angles]), servo_dof_indices)
        
        # Calcola i dettagli aerodinamici per ogni superficie
        aero_details = compute_aero_details(drone, throttle)
        
        # Applica le forze aerodinamiche (qui puoi continuare ad applicare i wrench totali, se desideri)
        # Per compatibilità, se preferisci, potresti applicare solo il wrench totale:

        link_name_array = ["fuselage", "left_wing", "right_wing", "prop_frame_fuselage_0"]
        link_array = [drone.get_link(link_name) for link_name in link_name_array]
        link_idx_array = [link.idx for link in link_array]
        force_array = np.zeros((B, len(link_name_array), 3), dtype=np.float32)
        torque_array = np.zeros((B, len(link_name_array), 3), dtype=np.float32)
        i = 0
        for link_name in link_name_array:
            if link_name == "fuselage":
                total_force, total_moment = aero_details["total"]["wrench"]
            else:
                total_force, total_moment = aero_details[link_name]["wrench"]
            force_array[0, i, :] = np.array(total_force.cpu().numpy())
            torque_array[0, i, :] = np.array(total_moment.cpu().numpy())
            i += 1

        rigid_solver.apply_links_external_force(force=force_array, links_idx=link_idx_array)
        rigid_solver.apply_links_external_torque(torque=torque_array, links_idx=link_idx_array)
        scene.clear_debug_objects()
        # Debug: disegna frecce per ogni superficie aerodinamica
        # Per ogni chiave in aero_details che ha "cp_offset", "lift" e "drag"
        for key, details in aero_details.items():
            if key in link_mapping and "cp_offset" in details:
                # Ottieni il nome del link corrispondente
                link_name = link_mapping[key]
                link = drone.get_link(link_name)
                # Ottieni la rotazione del link (come matrice NumPy)
                R_link = quat_to_R(link.get_quat())
                R_link_np = R_link.cpu().numpy() if hasattr(R_link, 'cpu') else np.array(R_link)
                # Converti cp_offset in array numpy
                cp_offset_local = details["cp_offset"].cpu().numpy()
                # Calcola posizione globale di cp_offset
                link_pos = np.array(link.get_pos().cpu().numpy())
                world_cp = link_pos + R_link_np @ cp_offset_local
                # Imposta scaling per visualizzazione (facoltativo, per regolare la lunghezza visiva)
                scale = 0.2  # adatta a seconda degli ordini di grandezza delle forze
                # Per lift: direzione locale (0, 0, 1) moltiplicata per lift
                alpha = details["alpha"].item()
                beta = details["beta"].item()
                axis_y = np.array([0, 1, 0], dtype=np.float32)
                axis_z = np.array([0, 0, 1], dtype=np.float32)
                R_y_np = axis_angle_to_R(axis_y, -alpha)
                R_z_np = axis_angle_to_R(axis_z, beta)
                R_aero = R_y_np @ R_z_np
                if details["lift"] is not None:
                    L_value = details["lift"].item()
                    local_lift = np.array([0, 0, L_value]) * scale
                    world_lift = R_link_np @ R_aero @ local_lift
                    # Disegna freccia di lift (verde)
                    scene.draw_debug_arrow(
                        pos=world_cp,
                        vec=world_lift,
                        radius=0.02,
                        color=(0, 1, 0, 1)
                    )
                # Per drag: direzione locale (1, 0, 0) moltiplicata per drag
                if details["drag"] is not None:
                    D_value = details["drag"].item()
                    local_drag = np.array([D_value, 0, 0]) * scale
                    world_drag = R_link_np @ R_aero @ local_drag
                    # Disegna freccia di drag (rossa)
                    scene.draw_debug_arrow(
                        pos=world_cp,
                        vec=world_drag,
                        radius=0.02,
                        color=(1, 0, 0, 1)
                    )
                # Per il rudder, se presente, si può disegnare anche il side_force (se presente)
                if key == "rudder" and "side_force" in details:
                    S_value = details["side_force"].item()
                    local_side = np.array([0, S_value, 0]) * scale
                    world_side = R_link_np @ R_aero @ local_side
                    # Disegna freccia per side_force (blu)
                    scene.draw_debug_arrow(
                        pos=world_cp,
                        vec=world_side,
                        radius=0.02,
                        color=(0, 0, 1, 1)
                    )
                # Per il propeller, se presente, si può disegnare anche il thrust (se presente)
            if key == "propeller":
                thrust = details["wrench"].item()[0]
                world_thrust = R_link_np @ np.array([0, 0, thrust]) * scale
                scene.draw_debug_arrow(
                    pos=link_pos,
                    vec=world_thrust,
                    radius=0.02,
                    color=(0, 1, 0, 1)
                )

        # Registra i dati per il plot
        state = rigid_solver.get_state(f=None)
        pos = np.array(state.qpos[0, :3].cpu().numpy()).squeeze()
        quat = np.array(state.qpos[0, 3:7].cpu().numpy()).squeeze()
        rpy = quat_to_xyz(quat)
        joint_pos = np.array(state.qpos[0, 7:].cpu().numpy()).squeeze()
        thrust = throttle * 15.0
        alpha = aero_details["fuselage"]["alpha"].item()
        beta = aero_details["fuselage"]["beta"].item()
        dofs_velocity = rigid_solver.get_dofs_velocity()
        
        offset = np.array([-5.0, 0.0, 3.0])
        offset_lookat = np.array([2.0, 0.0, 0.0])
        scene.viewer.set_camera_pose(
            pos=pos + offset,
            lookat=pos + offset_lookat,
        )

        time_log.append(sim_time)
        pos_log.append(pos)
        rpy_log.append(rpy)
        thrust_log.append(thrust)
        joint_log.append(joint_pos)
        alpha_log.append(alpha)
        beta_log.append(beta)

        
        # Step the simulation and update simulation time
        scene.step()
        sim_time += 0.01

except KeyboardInterrupt:
    print("\nSimulation terminated by user.")

finally:
    termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)

    # At the end of the simulation, plot the logged data
    time_log = np.array(time_log)
    pos_log = np.stack(pos_log)
    rpy_log = np.stack(rpy_log)
    thrust_log = np.array(thrust_log)
    joint_log = np.stack(joint_log)
    alpha_log = np.array(alpha_log)
    beta_log = np.array(beta_log)

    fig, axs = plt.subplots(5, 1, figsize=(10, 25), sharex=True)
    
    # Plot position (XYZ)
    axs[0].plot(time_log, pos_log[:, 0], label='x')
    axs[0].plot(time_log, pos_log[:, 1], label='y')
    axs[0].plot(time_log, pos_log[:, 2], label='z')
    axs[0].set_ylabel("Position (m)")
    axs[0].legend()
    axs[0].set_title("XYZ Position Over Time")
    
    # Plot Euler angles (rpy) in degrees
    axs[1].plot(time_log, np.degrees(rpy_log[:, 0]), label='roll')
    axs[1].plot(time_log, np.degrees(rpy_log[:, 1]), label='pitch')
    axs[1].plot(time_log, np.degrees(rpy_log[:, 2]), label='yaw')
    axs[1].set_ylabel("Angles (°)")
    axs[1].legend()
    axs[1].set_title("Euler Angles (rpy) Over Time")
    
    # Plot thrust over time
    axs[2].plot(time_log, thrust_log, label="Thrust")
    axs[2].set_ylabel("Thrust (N)")
    axs[2].legend()
    axs[2].set_title("Thrust Over Time")
    
    # Plot servo joint positions
    axs[3].plot(time_log, np.degrees(joint_log[:, 0]), label='Joint 1')
    axs[3].plot(time_log, np.degrees(joint_log[:, 1]), label='Joint 2')
    axs[3].set_ylabel("Joint Position (°)")
    axs[3].legend()
    axs[3].set_title("Servo Joint Positions Over Time")

    # Plot servo joint positions
    axs[4].plot(time_log, np.degrees(alpha_log), label='Joint 1')
    axs[4].plot(time_log, np.degrees(beta_log), label='Joint 2')
    axs[4].set_ylabel("Joint Position (°)")
    axs[4].legend()
    axs[4].set_title("Servo Joint Positions Over Time")

    # Optionally, you potrebbe aggiungere altri plot per altri dati
    axs[4].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
