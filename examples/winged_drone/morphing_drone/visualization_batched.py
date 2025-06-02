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


# ---------------- Parametri --------------------------------------------------
B            = 1                   # n° environment batch
DT           = 0.01
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANGLE_STEP   = np.deg2rad(0.5)     # passo tasti freccia (rad)

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
            collision=True, fixed=True,
        ))

# ---------------- Genesis scene ---------------------------------------------
gs.init(backend=gs.gpu)
scene = gs.Scene(
    sim_options = gs.options.SimOptions(dt=DT),
    show_viewer = True,
    
)
scene.add_entity(gs.morphs.Plane(pos=(0,0,0), collision=True)).set_friction(0.01)
create_forest(scene)

# ---------------- Drone ------------------------------------------------------
eul = np.deg2rad([0, 0, 0])                # roll, pitch, yaw
start_quat = xyz_to_quat(eul)
urdf_file = "/home/andrea/Documents/genesis/Genesis/genesis/assets/urdf/mydrone/mydrone_new.urdf"
drone = scene.add_entity(gs.morphs.URDF(
    file = urdf_file,
    pos  = (-20, 0, 20), quat=tuple(start_quat),
    collision=True, merge_fixed_links=True,
    links_to_keep=[
        "aero_frame_fuselage","aero_frame_left_wing","aero_frame_right_wing",
        "elevator","rudder","aero_frame_elevator", "aero_frame_rudder",
        "prop_frame_fuselage_0",
        "fuselage","left_wing","right_wing",
    ],
))

print("Drone loaded:", drone)

scene.build(n_envs=B, env_spacing=(4.0,4.0))

# ---------------- Servomeccanismi -------------------------------------------
servo_joint_names = [
    "joint_0_twist_left_wing",
    "joint_0_twist_right_wing",
    "elevator_pitch_joint",
    "rudder_yaw_joint",
]
servo_dof_idx = [drone.get_joint(n).dof_idx_local for n in servo_joint_names]

N_SERVO = len(servo_joint_names)
kp = torch.full((N_SERVO,), 500.0, device=DEVICE, dtype=torch.float32)
kv = torch.full((N_SERVO,),   20.0, device=DEVICE, dtype=torch.float32)
max_tau = 3.0  # [Nm], scegli in base ai carichi aerodinamici

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
servo_pos[:, :2] = -0.                            # -0.1 rad pre-twist
servo_pos[:, 2] = 0.0                            # -0.1 rad pre-twist
servo_pos[:, 3] = 0.0                            # -0.1 rad pre-twist
drone.set_dofs_position(servo_pos, servo_idx)

vel0 = gs.tensor([[8.0,0,0, 0,0,0, 0,0,0,0]]).repeat(B,1)  # lin+ang+joint
vel0[:,0] += gs_rand_float(-0.0, 0.0, (B,), DEVICE)
rigid_solver = next(s for s in scene.sim.solvers if isinstance(s, RigidSolver))
rigid_solver.set_dofs_velocity(vel0)

# ---------------- Input tastiera (non-blocking) -----------------------------
stdin_fd, old_tty = sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno())
tty.setcbreak(stdin_fd)

throttle      = torch.zeros(B, device=DEVICE) + 0.0
servo_targets = servo_pos

# ---------------- Link sui quali applicare le forze -------------------------
link_order = [
    "aero_frame_fuselage",
    "aero_frame_left_wing",
    "aero_frame_right_wing",
    "aero_frame_elevator",
    "aero_frame_rudder",
    "prop_frame_fuselage_0",
]
link_objs  = [drone.get_link(n) for n in link_order]
link_idx   = [l.idx for l in link_objs]

link_mapping = {           # per debug-arrow
    "fuselage": "aero_frame_fuselage",
    "left_wing": "aero_frame_left_wing",
    "right_wing": "aero_frame_right_wing",
    "elevator_left": "aero_frame_elevator",
    "elevator_right": "aero_frame_elevator",
    "rudder":  "aero_frame_rudder",
    "propeller": "prop_frame_fuselage_0",
}

link2aero = {
    "aero_frame_fuselage"   : "fuselage",
    "aero_frame_left_wing"  : "left_wing",
    "aero_frame_right_wing" : "right_wing",
    "aero_frame_rudder"     : "rudder",
    "prop_frame_fuselage_0" : "prop_frame_fuselage_0",
    # l’elevator fa caso a sé (due semiali)
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
solver = AerodynamicSolver(device=DEVICE, urdf_file=urdf_file) 

# ---------------- Loop simulazione ------------------------------------------
print(" ⬆ / ⬇ throttle | ← / → twist | w/s elevator | q/e rudder  –  Ctrl-C esci")
t = 0.0
try:
    while True:
        if B == 1:
            print(f"Servo targets: {servo_targets[0].cpu().numpy()}")

        # ---------- controlli ------------------------------------------------
        drone.control_dofs_position(servo_targets, servo_idx)

        # ---------- aerodinamica --------------------------------------------
        aero = solver.compute_aero_details(drone, throttle)         # nuovo solver

        F = torch.zeros((B, len(link_order), 3), device=DEVICE)
        pos = torch.zeros((B, len(link_order), 3), device=DEVICE)  # posizioni CP
        M = torch.zeros_like(F)

        for i, name in enumerate(link_order):
            if name == "aero_frame_elevator":
                wf_L, wm_L = aero["elevator_left"]["wrench"]
                wf_R, wm_R = aero["elevator_right"]["wrench"]
                wf, wm = wf_L + wf_R, wm_L + wm_R
                cp_local = (aero["elevator_left"]["cp_offset"] + aero["elevator_right"]["cp_offset"])/ 2.0

            else:
                aero_key = link2aero[name]          # usa la tabella
                wf, _   = aero[aero_key]["wrench_body"]
                cp_local = aero[aero_key]["cp_offset"].squeeze(0)

            #if name == "elevator" or name == "rudder":
                # Forza aerodinamica per elevator e rudder
            #    wf = torch.zeros_like(wf)
            #    wm = torch.zeros_like(wm)

            #F[:,i], M[:,i] = wf, wm
            F[:,i] = wf
            pos[:, i] = cp_local
            #print(f"Link {name}: F={F[:,i].cpu().numpy()}  M={M[:,i].cpu().numpy()}")

        
        #moment_log.append(M[0].detach().cpu().numpy())
        #force_log.append(F[0].detach().cpu().numpy())
        #torque = drone.get_dofs_force(servo_idx)
        rigid_solver.apply_links_external_force_link_frame(
            pos, F, link_idx, 0,
        )

        #rigid_solver.apply_links_external_force (F, link_idx)
        #rigid_solver.apply_links_external_torque(M, link_idx)

        # ---------- camera + logging per il primo env ------------------------
        state = rigid_solver.get_state(f=None)
        pos0  = state.qpos[0,:3].cpu().numpy()
        vel0  = drone.get_dofs_velocity()[0,:3].cpu().numpy()
        quat0 = state.qpos[0,3:7].cpu().numpy()
        rpy0  = quat_to_xyz(quat0)
        thrust_val = throttle[0].item()*15.0
        alpha0 = aero["fuselage"]["alpha"][0].item()
        beta0  = aero["fuselage"]["beta"][0].item()
        alpha_elev_r = aero["elevator_right"]["alpha"][0].item()
        alpha_elev_l = aero["elevator_left"]["alpha"][0].item()
        beta_elev  = aero["elevator_right"]["beta"][0].item()
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
        alpha_elev_r_log.append(alpha_elev_r)
        alpha_elev_l_log.append(alpha_elev_l)
        beta_elev_log.append(beta_elev)


        CG = aero["CG"][0].cpu().numpy()
        AC = aero["AC"][0].cpu().numpy()
        
        # prendi posa e orientazione del frame fusoliera
        pos_fus = drone.get_link("fuselage").get_pos()[0].cpu().numpy()      # (3,) world
        quat_fus= drone.get_link("fuselage").get_quat()[0]    # (4,) world
        R_fus   = quat_to_R(quat_fus.unsqueeze(0))[0].cpu().numpy()          # (3,3)

        # CG in fuselage-frame = R_fus^T * (CG_world - pos_fus)
        CG_local = R_fus.T @ (CG - pos_fus)              # (3,)
        AC_local = R_fus.T @ (AC - pos_fus)              # (3,)

        print(f"\n  CG: {CG_local[0]:6.4f} {CG_local[1]:6.4f} {CG_local[2]:6.4f}  AC: {AC_local[0]:6.2f} {AC_local[1]:6.2f} {AC_local[2]:6.2f}", end="")
        print(f"\n  α={np.degrees(alpha0):+5.2f}°  β={np.degrees(beta0):+5.2f}°  Thr={thrust_val:4.1f}  ", end="")
        print(f"\n  pitch={np.degrees(rpy0[1]):+5.1f}°  roll={np.degrees(rpy0[0]):+5.1f}°  yaw={np.degrees(rpy0[2]):+5.1f}°  ", end="")
        print(f"\n  pos={pos0[0]:6.1f}  y={pos0[1]:5.1f}  z={pos0[2]:5.1f}  ", end="")
        print("\n  ‖M_res‖ =", aero["M_residual"].norm(dim=1).item())


        pos_wingR = drone.get_link("right_wing").get_pos()[0].cpu().numpy()
        pos_wingL = drone.get_link("left_wing").get_pos()[0].cpu().numpy()
        pos_elev = drone.get_link("elevator").get_pos()[0].cpu().numpy()
        pos_rudder = drone.get_link("rudder").get_pos()[0].cpu().numpy()

        wingR_local = R_fus.T @ (pos_wingR - pos_fus)              # (3,)
        wingL_local = R_fus.T @ (pos_wingL - pos_fus)              # (3,)
        elev_local = R_fus.T @ (pos_elev - pos_fus)              # (3,)
        rudder_local = R_fus.T @ (pos_rudder - pos_fus)              # (3,)


        CG_log.append(CG_local)
        AC_log.append(AC_local)

        # facoltativo: palline di debug nel viewer
        if B == 1:

            scene.draw_debug_sphere(pos=CG, radius=0.04, color=(1,1,0,1))  # giallo
            scene.draw_debug_sphere(pos=AC, radius=0.04, color=(0,0,0,1))  # nero

            scale = 0.2               # lunghezza visuale delle frecce
            axis_y = np.array([0, 1, 0], dtype=np.float32)
            axis_z = np.array([0, 0, 1], dtype=np.float32)

            for key, details in aero.items():
                if key not in link_mapping or "cp_offset" not in details:
                    #print(f"Link {key} non trovato in link_mapping o cp_offset non presente.")
                    continue

                link     = drone.get_link(link_mapping[key])

                # >>> to­rch → numpy e rimuovi la dim. batch (1,⋯)  <<<
                R_link   = quat_to_R(link.get_quat()).cpu().numpy().squeeze(0)   # (3,3)
                link_pos = link.get_pos().cpu().numpy().squeeze(0)              # (3,)
                cp_local = details["cp_offset"].cpu().numpy().squeeze(0)        # (3,)

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
                    T = details["wrench"][0][0, 0].item()        # Fx del batch 0
                    v_thrust = R_link @ np.array([0, 0, T]) * scale
                    scene.draw_debug_arrow(pos=link_pos, vec=v_thrust,
                                        radius=0.02, color=(0,1,0,1))
                    
        r, _, _ = select.select([sys.stdin], [], [])
        ch = sys.stdin.read(1)
        if ch == " ":
            # passo in avanti
            scene.step()
            scene.clear_debug_objects()

            t += DT       
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
        elif ch.lower() == "w": servo_targets[:,2].add_( ANGLE_STEP)  # elevator up
        elif ch.lower() == "s": servo_targets[:,2].sub_( ANGLE_STEP)  # elevator down
        elif ch.lower() == "q": servo_targets[:,3].add_( ANGLE_STEP)  # rudder left
        elif ch.lower() == "e": servo_targets[:,3].sub_( ANGLE_STEP)  # rudder right
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