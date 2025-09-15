#!/usr/bin/env python3
# coding: utf-8
"""
Demo tastiera + viewer Genesis
✧ Drone “morphing_drone” con elevator e rudder separati
✧ Solver aerodinamico compute_aero_details() aggiornato
"""

import sys, select, termios, tty, time
import numpy as np
import matplotlib.pyplot as plt
import genesis as gs
from aer_solver_class import AerodynamicSolver
from genesis.engine.solvers import RigidSolver
from genesis.utils.geom import quat_to_xyz, xyz_to_quat, quat_to_R, axis_angle_to_R

import torch


# ---------------------------------------------------------------
#  CG e AC helpers
# ---------------------------------------------------------------
def compute_cg(drone) -> np.ndarray:
    """
    Center of gravity in world–frame.
    È costante (finché non sganci massa), quindi basta calcolarlo una volta.
    """

    masses, pos = [], []
    for l in drone.links:                   # tutti i link presenti nel modello
        masses.append(l.get_mass())         # kg
        pos.append(l.inertial_pos.squeeze())  # (3,))  # (3,)
    m = np.asarray(masses)
    p = np.vstack(pos)
    
    return (m[:, None] * p).sum(0) / m.sum()            # (3,)

def compute_ac(aero_details: dict,
               drone,
               link_map: dict) -> np.ndarray:
    """
    Aerodynamic center in world–frame: media dei centri-pressione
    ponderata dalla portanza istantanea.
    Se la somma delle portanze è ~0 (volo rovesciato, stallo totale, ecc.)
    ritorna NaN.
    """
    cps, lifts = [], []
    for key, d in aero_details.items():
        if "cp_offset" not in d:           # sicurezza
            continue
        # posizione world del CP
        link = drone.get_link(link_map[key])
        R    = quat_to_R(link.get_quat()).cpu().numpy().squeeze()
        p_w  = link.inertial_pos.squeeze()
        cp_w = p_w + R @ d["cp_offset"].cpu().numpy()
        cps.append(cp_w)
        lifts.append(d["lift"].item())     # N

    L = np.asarray(lifts)
    if np.abs(L).sum() < 1e-6:             # evita divisioni instabili
        return np.full(3, np.nan)
    C = np.vstack(cps)
    return (L[:, None] * C).sum(0) / L.sum()



def extract_aero_details(solver, batch=0):
    """
    Converte solver._dbg in un dizionario indicizzato
    con i nomi che usi nel resto dello script.
    """
    if not hasattr(solver, "_dbg"):
        return {}

    dbg = solver._dbg
    out = {}
    for i, frame in enumerate(AERO_FRAMES):
        # Usa lo stesso mapping che avevi già
        logical_name = link_mapping.get(frame_to_part(frame), frame)
        out[logical_name] = {
            "alpha": torch.tensor(dbg["alpha"][batch, i]),
            "beta" : torch.tensor(dbg["beta" ][batch, i]),
            "lift" : torch.tensor(dbg["lift" ][batch, i]),
            "drag" : torch.tensor(dbg["drag" ][batch, i]),
            "side_force" : torch.tensor(dbg["side_force"][batch, i]),
            "cp_offset" : torch.tensor(dbg["cp_offset"][batch, i]),
        }
    return out

def frame_to_part(f):
    return {
        # fusoliera
        "aero_frame_fuselage"         : "fuselage",
        # ali
        "aero_frame_left_wing_prop"   : "left_wing_prop",
        "aero_frame_left_wing_free"   : "left_wing_free",
        "aero_frame_right_wing_prop"  : "right_wing_prop",
        "aero_frame_right_wing_free"  : "right_wing_free",
        # elevator
        "aero_frame_elevator_left"    : "elevator_left",
        "aero_frame_elevator_right"   : "elevator_right",
        # timone ed elica
        "aero_frame_rudder"           : "rudder",
        "prop_frame_fuselage_0"       : "propeller",
    }[f]


# ---------------- Parametri --------------------------------------------------
B            = 1                   # n° environment batch
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANGLE_STEP   = np.deg2rad(0.5)     # passo tasti freccia (rad)

CONTROL_HZ  = 100  # azioni alla PPO
PHYSICS_HZ  = 100 # integrazione rigida
dt_control  = 1.0 / CONTROL_HZ
substeps    = int(PHYSICS_HZ / CONTROL_HZ)   # 4

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

# ---------------- Foresta ----------------------------------------------------
def create_forest(scene):
    num_trees, r, h = 1, 1.0, 50.0
    xs = gs_rand_float(0, 200, (num_trees,), DEVICE).cpu().numpy()
    ys = gs_rand_float(-50, 50, (num_trees,), DEVICE).cpu().numpy()
    for x, y in zip(xs, ys):
        scene.add_entity(gs.morphs.Cylinder(
            pos=(float(x), float(y), h/2),
            quat=(1,0,0,0), radius=r, height=h,
            collision=False, fixed=True,
        ))

# ---------------- Genesis scene ---------------------------------------------
gs.init(backend=gs.gpu)
scene = gs.Scene(
    sim_options = gs.options.SimOptions(dt=dt_control, substeps=substeps),
    show_viewer = True,
    
)
scene.add_entity(gs.morphs.Plane(pos=(0,0,0), collision=False)).set_friction(0.01)
create_forest(scene)

# ---------------- Drone ------------------------------------------------------
eul = np.deg2rad([0, 0, 0])                # roll, pitch, yaw
start_quat = xyz_to_quat(eul)
urdf_file = "/home/andrea/Documents/genesis/Genesis/genesis/assets/urdf/mydrone/[0.7, 0.2, -0.275, -0.275, 0.73, 0.18, 0.14, 0.16, 0.12, 1, 0, 0.1, 0.14].urdf"
drone = scene.add_entity(gs.morphs.URDF(
    file = urdf_file,
    pos  = (-20, 0, 20), quat=tuple(start_quat),
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


# 1) Costruisci e builda la scena
scene.build(n_envs=B, env_spacing=(4.0,4.0))

rigid_solver = next(s for s in scene.sim.solvers if isinstance(s, RigidSolver))
AERO_FRAMES = [
    # fusoliera
    "aero_frame_fuselage",
    # ali
    "aero_frame_left_wing_prop",   "aero_frame_left_wing_free",
    "aero_frame_right_wing_prop",  "aero_frame_right_wing_free",
    # elevator (2 pannelli)
    "aero_frame_elevator_left",
    "aero_frame_elevator_right",
    # timone + elica
    "aero_frame_rudder",
    "prop_frame_fuselage_0",
]

# ---------- configura aerodinamica -----------------------------------

rigid_solver._aero_log    = True                      # ② logga α,β,lift,drag
rigid_solver.add_target(drone, urdf_file=urdf_file)   # ③ attiva aerodinamica
# ---------------- Servomeccanismi -------------------------------------------
servo_joint_names = [
    "joint_0_sweep_left_wing",   # nuovo
    "joint_0_sweep_right_wing",  # nuovo
    "joint_1_twist_left_wing",
    "joint_1_twist_right_wing",
    "elevator_pitch_joint",
    "rudder_yaw_joint",
]
(
    IDX_SWEEP_L,      #  joint_1_sweep_left_wing
    IDX_SWEEP_R,      #  joint_1_sweep_right_wing
    IDX_TWIST_L,
    IDX_TWIST_R,
    IDX_ELEV,
    IDX_RUDDER,
) = range(len(servo_joint_names))
servo_dof_idx = [drone.get_joint(n).dof_idx_local for n in servo_joint_names]

N_SERVO = len(servo_joint_names)
kp = torch.full((N_SERVO,), 10.0, device=DEVICE, dtype=torch.float32)
kv = torch.full((N_SERVO,),   2.0, device=DEVICE, dtype=torch.float32)
max_tau = 1  # [Nm], scegli in base ai carichi aerodinamici

# 1) crea un vettore di indici LONG sullo stesso device della simulazione
servo_idx = torch.tensor(servo_dof_idx, device=DEVICE, dtype=torch.long)

# 2) usa SEMPRE quel tensor per impostare guadagni / range
drone.set_dofs_kp(kp, servo_idx)
drone.set_dofs_kv(kv, servo_idx)
drone.set_dofs_force_range(
    upper=torch.full((N_SERVO,),  max_tau, device=DEVICE),
    lower=torch.full((N_SERVO,), -max_tau, device=DEVICE),
    dofs_idx_local=servo_idx
)


# ---------------- Stato iniziale --------------------------------------------
# 3) imposta la posizione iniziale dei servomeccanismi
servo_pos = torch.zeros((B, N_SERVO), device=DEVICE, dtype=torch.float32)
servo_pos[:, :2] = -0.4                            # -0.1 rad pre-twist
servo_pos[:, 2] = 0.4                          # -0.1 rad pre-twist
servo_pos[:, 3] = 0.4                            # -0.1 rad pre-twist
drone.set_dofs_position(servo_pos, servo_idx)

vel0 = gs.tensor([[8.0,0,0, 0,0,0, 0,0,0,0,0,0]]).repeat(B,1)  # lin+ang+joint
vel0[:,0] += gs_rand_float(-0.0, 0.0, (B,), DEVICE)
rigid_solver.set_dofs_velocity(vel0)

# ---------------- Input tastiera (non-blocking) -----------------------------
stdin_fd, old_tty = sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno())
tty.setcbreak(stdin_fd)

throttle      = torch.zeros(B, device=DEVICE)
servo_targets = servo_pos

# ---------------- Link sui quali applicare le forze -------------------------
link_order = AERO_FRAMES
link_objs  = [drone.get_link(n) for n in link_order]

link_idx   = [l.idx for l in link_objs]
print(f"Link index: {link_idx}")

link_mapping = {
    "aero_frame_fuselage"        : "aero_frame_fuselage",
    # ali
    "aero_frame_left_wing_prop"  : "aero_frame_left_wing_prop",
    "aero_frame_left_wing_free"  : "aero_frame_left_wing_free",
    "aero_frame_right_wing_prop" : "aero_frame_right_wing_prop",
    "aero_frame_right_wing_free" : "aero_frame_right_wing_free",
    # elevator
    "aero_frame_elevator_left"   : "aero_frame_elevator_left",
    "aero_frame_elevator_right"  : "aero_frame_elevator_right",
    # timone + elica
    "aero_frame_rudder"          : "aero_frame_rudder",
    "prop_frame_fuselage_0"      : "prop_frame_fuselage_0",
}



# Lpog arrays
time_log = []
pos_log = []
vel_log = []
rpy_log = []
thrust_log = []
joint_log = []
alpha_log = []
beta_log = []
alpha_elev_r_log = []
alpha_elev_l_log = []
beta_elev_log = []
moment_log = [] 
force_log = []
CG_log = []
AC_log = []

# ---------------- Aerodynamic solver ----------------------------------------
#solver = AerodynamicSolver(device=DEVICE, urdf_file=urdf_file, log_details=True) 
scene.step()                 # ← deve stare NEL ciclo, prima di leggere aero

# ---------------- Loop simulazione ------------------------------------------
print(" ⬆ / ⬇ throttle | ← / → twist | w/s elevator | q/e rudder  –  Ctrl-C esci")
t = 0.0
try:
    while True:
        if B == 1:
            print(f"\n  Servo targets: {servo_targets[0].cpu().numpy()}")
        # ---------- controlli ------------------------------------------------
        # ----- comandi al drone --------------------------------------------------
        drone.control_dofs_position(servo_targets, servo_idx)
        # clamp throttle at max=1
        throttle.clamp_(0, 1)
        rigid_solver.set_throttle(throttle)
        #rigid_solver._aero_step()

        # ----- AVANZA LA SIMULAZIONE (obbligatorio!) -----------------------------

        # ---------- aerodinamica --------------------------------------------
        #aero, F, P = solver.apply_aero_forces(drone, throttle, rigid_solver)

        torque = drone.get_dofs_force(servo_idx)

        aero = extract_aero_details(rigid_solver)

        # ---------- camera + logging per il primo env ------------------------
        state = rigid_solver.get_state(f=None)
        pos0  = state.qpos[0,:3].cpu().numpy()
        vel0  = drone.get_dofs_velocity()[0,:3].cpu().numpy()
        quat0 = state.qpos[0,3:7].cpu().numpy()
        rpy0  = quat_to_xyz(quat0)
        thrust_val = throttle[0].item()
        alpha0 = aero["aero_frame_fuselage"]["alpha"].item()
        beta0  = aero["aero_frame_fuselage"]["beta"].item()
        #alpha_elev_r = aero["elevator_right"]["alpha"][0].item()
        #alpha_elev_l = aero["elevator_left"]["alpha"][0].item()
        #beta_elev  = aero["elevator_right"]["beta"][0].item()
        joint_pos = drone.get_dofs_position()[0, servo_idx].cpu().numpy()

        #print(f"\r x={pos0[0]:6.1f}  z={pos0[2]:5.1f}  α={alpha0:+5.1f}°  β={beta0:+5.1f}°  Thr={thrust_val:4.1f}  ", end="")

        scene.viewer.set_camera_pose(pos=pos0 + np.array([-5,1,3.5]),
                                     lookat=pos0 + np.array([ 2,0,0]))
        #scene.viewer.set_camera_pose(pos=pos0 + np.array([-0.2,2,0]),
        #                             lookat=pos0 + np.array([ -0.2,0,0]))

        # ---------- step -----------------------------------------------------

        time_log.append(t)
        pos_log.append(pos0)
        vel_log.append(vel0)
        rpy_log.append(rpy0)
        thrust_log.append(thrust_val)
        joint_log.append(joint_pos)
        alpha_log.append(alpha0)
        beta_log.append(beta0)
        #alpha_elev_r_log.append(alpha_elev_r)
        #alpha_elev_l_log.append(alpha_elev_l)
        #beta_elev_log.append(beta_elev)
        print(f"\n  α={np.degrees(alpha0):+5.2f}°  β={np.degrees(beta0):+5.2f}°  Thr={thrust_val:4.1f}  ", end="")
        print(f"\n  pitch={np.degrees(rpy0[1]):+5.1f}°  roll={np.degrees(rpy0[0]):+5.1f}°  yaw={np.degrees(rpy0[2]):+5.1f}°  ", end="")
        print(f"\n  pos={pos0[0]:6.1f}  y={pos0[1]:5.1f}  z={pos0[2]:5.1f}  ", end="")
        print(f"\n  servo={joint_pos}  ", end="")
        CG_body = compute_cg(drone) 
        AC_body = compute_ac(aero, drone, link_mapping)
        print(f"\n  CG={CG_body}  AC={AC_body}  ", end="")
        CG_world = (drone.get_link("fuselage").get_pos().cpu().numpy() + quat_to_R(drone.get_link("fuselage").get_quat().cpu().numpy().squeeze(0)) @ CG_body).squeeze(0)
        AC_world = (drone.get_link("aero_frame_fuselage").get_pos().cpu().numpy() + quat_to_R(drone.get_link("aero_frame_fuselage").get_quat().cpu().numpy().squeeze(0)) @ AC_body).squeeze(0)

        # facoltativo: palline di debug nel viewer
        if B == 1:

            scene.draw_debug_sphere(pos=CG_world, radius=0.07, color=(1,1,0,1))   # giallo
            if not np.isnan(AC_world[0]):
                scene.draw_debug_sphere(pos=AC_world, radius=0.07, color=(0,0,0,1))   # nero

            scale = 0.2               # lunghezza visuale delle frecce
            axis_y = np.array([0, 1, 0], dtype=np.float32)
            axis_z = np.array([0, 0, 1], dtype=np.float32)

            for key, details in aero.items():
                if key not in AERO_FRAMES:
                    #print(f"Link {key} non trovato in link_mapping o cp_offset non presente.")
                    continue

                link     = drone.get_link(link_mapping[key])

                # >>> to­rch → numpy e rimuovi la dim. batch (1,⋯)  <<<
                R_link   = quat_to_R(link.get_quat()).cpu().numpy().squeeze(0)   # (3,3)
                link_pos = link.get_pos().cpu().numpy().squeeze(0)              # (3,)
                cp_local = details["cp_offset"].cpu().numpy()      # (3,)

                world_cp = link_pos + R_link @ cp_local

                scene.draw_debug_sphere(
                    pos=world_cp,
                    radius=0.03,           # regola la dimensione a piacere
                    color=(1.0, 0.0, 1.0, 1.0)  # magenta
                )

                # rotazione locale aerodinamica per α / β
                alpha = details["alpha"].item()
                beta  = details["beta"].item()
                R_aero = axis_angle_to_R(axis_y, -alpha) @ axis_angle_to_R(axis_z, beta)

                # ---------------- lift (verde)
                if details.get("lift") is not None:
                    L = details["lift"].item()
                    v_lift  = R_link @ R_aero @ (np.array([0,0,L])*scale)
                    scene.draw_debug_arrow(pos=world_cp, vec=v_lift,
                                        radius=0.02, color=(0,1,0,1))

                # ---------------- drag (rosso)
                if details.get("drag") is not None:
                    D = details["drag"].item()
                    v_drag  = R_link @ R_aero @ (np.array([D,0,0])*scale)
                    scene.draw_debug_arrow(pos=world_cp, vec=v_drag,
                                        radius=0.02, color=(1,0,0,1))

                # ---------------- side-force per rudder (blu)
                if key == "rudder" and details.get("side_force") is not None:
                    S = details["side_force"].item()
                    v_side  = R_link @ R_aero @ (np.array([0,S,0])*scale)
                    scene.draw_debug_arrow(pos=world_cp, vec=v_side,
                                        radius=0.02, color=(0,0,1,1))

                # ---------------- thrust per propeller (verde puntiforme)
                if key == "propeller":
                    T = details["force"][0, 0].item()        # Fx del batch 0
                    v_thrust = R_link @ np.array([0, 0, T]) * scale
                    scene.draw_debug_arrow(pos=link_pos, vec=v_thrust,
                                        radius=0.02, color=(0,1,0,1))
                    
        r, _, _ = select.select([sys.stdin], [], [])
        ch = sys.stdin.read(1)
        if ch == " ":
            # passo in avanti
            scene.step()
            scene.clear_debug_objects()

            t += dt_control       
        elif ch == "\x1b":                       # frecce
            seq = ch + sys.stdin.read(2)
            if   seq == "\x1b[A": throttle.add_(0.025).clamp_(0,1)
            elif seq == "\x1b[B": throttle.sub_(0.025).clamp_(0,1)
            elif seq == "\x1b[C":
                servo_targets[:,0:2].add_( ANGLE_STEP)   # roll right
            elif seq == "\x1b[D":
                servo_targets[:,0:2].sub_( ANGLE_STEP)   # roll left
        elif ch.lower() == "l": 
            servo_targets[:,0].add_( ANGLE_STEP)  # elevator up
            servo_targets[:,1].sub_( ANGLE_STEP)  # elevator up
        elif ch.lower() == "k": 
            servo_targets[:,0].sub_( ANGLE_STEP)  # elevator up
            servo_targets[:,1].add_( ANGLE_STEP)  # elevator up
        elif ch.lower() == "w": servo_targets[:,IDX_ELEV].add_( ANGLE_STEP)  # elevator up
        elif ch.lower() == "s": servo_targets[:,IDX_ELEV].sub_( ANGLE_STEP)  # elevator down
        elif ch.lower() == "q": servo_targets[:,IDX_RUDDER].add_( ANGLE_STEP)  # rudder left
        elif ch.lower() == "e": servo_targets[:,IDX_RUDDER].sub_( ANGLE_STEP)  # rudder right
        elif ch.lower() == "z":                               # sweep avanti
            servo_targets[:, IDX_SWEEP_L].add_(ANGLE_STEP)
            servo_targets[:, IDX_SWEEP_R].sub_(ANGLE_STEP)
        elif ch.lower() == "x":                               # sweep indietro
            servo_targets[:, IDX_SWEEP_L].sub_(ANGLE_STEP)
            servo_targets[:, IDX_SWEEP_R].add_(ANGLE_STEP)
        servo_targets.clamp_(-1.0, 1.0)

except KeyboardInterrupt:
    print("\nSimulazione terminata.")
finally:
    termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_tty)

    '''
    # At the end of the simulation, plot the logged data
    time_log = np.array(time_log)
    pos_log = np.stack(pos_log)
    vel_log = np.stack(vel_log)
    rpy_log = np.stack(rpy_log)
    thrust_log = np.array(thrust_log)
    joint_log = np.stack(joint_log)
    alpha_log = np.array(alpha_log)
    beta_log = np.array(beta_log)
    alpha_elev_l_log = np.array(alpha_elev_l_log)
    alpha_elev_r_log = np.array(alpha_elev_r_log)
    beta_elev_log = np.array(beta_elev_log)


    fig, axs = plt.subplots(7, 1, figsize=(14, 25), sharex=True)
    
    # Plot position (XYZ)
    axs[0].plot(time_log, pos_log[:, 0], label='x')
    axs[0].plot(time_log, pos_log[:, 1], label='y')
    axs[0].plot(time_log, pos_log[:, 2], label='z')
    axs[0].set_ylabel("Position (m)")
    axs[0].legend()
    axs[0].set_title("XYZ Position Over Time")

    # Plot velocity (XYZ)
    axs[1].plot(time_log, vel_log[:, 0], label='vx')
    axs[1].plot(time_log, vel_log[:, 1], label='vy')
    axs[1].plot(time_log, vel_log[:, 2], label='vz')
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].legend()
    axs[1].set_title("XYZ Velocity Over Time")
    
    # Plot Euler angles (rpy) in degrees
    axs[2].plot(time_log, np.degrees(rpy_log[:, 0]), label='roll')
    axs[2].plot(time_log, np.degrees(rpy_log[:, 1]), label='pitch')
    axs[2].plot(time_log, np.degrees(rpy_log[:, 2]), label='yaw')
    axs[2].set_ylabel("Angles (°)")
    axs[2].legend()
    axs[2].set_title("Euler Angles (rpy) Over Time")
    
    # Plot thrust over time
    axs[3].plot(time_log, thrust_log, label="Thrust")
    axs[3].set_ylabel("Thrust (N)")
    axs[3].legend()
    axs[3].set_title("Thrust Over Time")
    
    # Plot servo joint positions
    axs[4].plot(time_log, np.degrees(joint_log[:, 0]), label='Joint 1')
    axs[4].plot(time_log, np.degrees(joint_log[:, 1]), label='Joint 2')
    axs[4].plot(time_log, np.degrees(joint_log[:, 2]), label='Joint 3')
    axs[4].plot(time_log, np.degrees(joint_log[:, 3]), label='Joint 4')
    axs[4].set_ylabel("Joint Position (°)")
    axs[4].legend()
    axs[4].set_title("Servo Joint Positions Over Time")

    axs[5].plot(time_log, np.degrees(alpha_log), label='alpha')
    axs[5].plot(time_log, np.degrees(beta_log), label='beta')
    axs[5].set_ylabel("Aero Angles (°)")
    axs[5].legend()
    axs[5].set_title("Fuselage Angles Over Time")

    axs[6].plot(time_log, np.degrees(alpha_elev_l_log), label='alpha_elev_l')
    axs[6].plot(time_log, np.degrees(alpha_elev_r_log), label='alpha_elev_r')
    axs[6].plot(time_log, np.degrees(beta_elev_log), label='beta_elev')
    axs[6].set_ylabel("Elevator Angles (°)")
    axs[6].legend()
    axs[6].set_title("Elevator Angles Over Time")

    # Optionally, you potrebbe aggiungere altri plot per altri dati
    axs[6].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show(block=False)

    # ------------------------------------------------------------------
    # Figure 2 – Momenti aerodinamici su ogni link
    # ------------------------------------------------------------------
    moment_log = np.stack(moment_log)   # (N, 6, 3)

    fig2, axs2 = plt.subplots(len(link_order), 1,
                              figsize=(14, 3*len(link_order)),
                              sharex=True)

    for idx, name in enumerate(link_order):
        axs2[idx].plot(time_log, moment_log[:, idx, 0], label='Mx')
        axs2[idx].plot(time_log, moment_log[:, idx, 1], label='My')
        axs2[idx].plot(time_log, moment_log[:, idx, 2], label='Mz')
        axs2[idx].set_ylabel("Moment [N·m]")
        axs2[idx].set_title(f"Momenti su {name}")
        axs2[idx].legend(loc="upper right")

    axs2[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show(block=False)

    force_log = np.stack(force_log)

    fig3, axs3 = plt.subplots(len(link_order), 1,
                                figsize=(14, 3*len(link_order)),
                                sharex=True)
    for idx, name in enumerate(link_order):
        axs3[idx].plot(time_log, force_log[:, idx, 0], label='Fx')
        axs3[idx].plot(time_log, force_log[:, idx, 1], label='Fy')
        axs3[idx].plot(time_log, force_log[:, idx, 2], label='Fz')
        axs3[idx].set_ylabel("Force [N]")
        axs3[idx].set_title(f"Force su {name}")
        axs3[idx].legend(loc="upper right")
    axs3[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show(block=False)

    # ------------------------------------------------------------------
    # Figure 3 – CG e AC
    # ------------------------------------------------------------------
    CG_log = np.stack(CG_log)
    AC_log = np.stack(AC_log)
    fig4, axs4 = plt.subplots(2, 1,
                              figsize=(14, 6),
                              sharex=True)
    axs4[0].plot(time_log, CG_log[:, 0], label='x')
    axs4[0].plot(time_log, CG_log[:, 1], label='y')
    axs4[0].plot(time_log, CG_log[:, 2], label='z')
    axs4[0].set_ylabel("CG (m)")
    axs4[0].legend()
    axs4[0].set_title("CG Over Time")
    axs4[1].plot(time_log, AC_log[:, 0], label='x')
    axs4[1].plot(time_log, AC_log[:, 1], label='y')
    axs4[1].plot(time_log, AC_log[:, 2], label='z')
    axs4[1].set_ylabel("AC (m)")
    axs4[1].legend()
    axs4[1].set_title("AC Over Time")
    axs4[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


    '''