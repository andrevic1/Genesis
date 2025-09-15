#!/usr/bin/env python3
"""
Real-time Genesis simulation of Bixler-3 using aerodynamic wrench and keyboard control.

Controls:
  ↑ / ↓  : throttle (0.0 → 1.0 step 0.025)
  J / L  : aileron ±0.5°
  I / K  : elevator ±0.5°
  U / O  : rudder ±0.5°
  X      : reset all servos to 0

Outputs final plots of position, attitude, thrust, servo deflections, velocities, angle of attack (alpha), sideslip (beta), and joint positions.
Adds debug printing of joint names and their indices.
"""
import sys
import select
import termios
import tty
import numpy as np
import matplotlib.pyplot as plt
import genesis as gs
from genesis.engine.solvers import RigidSolver
from genesis.utils.geom import quat_to_xyz, xyz_to_quat
import time

from aerodynamic_wrench_calculation import compute_aero_wrench

# Simulation parameters
SCENE_DT = 0.01  # 100 Hz
MAX_THRUST_N = 15.0

# Keyboard configuration
KEY_THROTTLE_UP = '\x1b[A'
KEY_THROTTLE_DOWN = '\x1b[B'
KEY_AIL_PLUS = 'j'
KEY_AIL_MINUS = 'l'
KEY_ELE_PLUS = 'i'
KEY_ELE_MINUS = 'k'
KEY_RUD_PLUS = 'u'
KEY_RUD_MINUS = 'o'
KEY_RESET = 'x'

STEP_THROTTLE = 0.025
STEP_ANG = np.deg2rad(1.0)   # 0.5°

def read_key_nonblocking():
    if select.select([sys.stdin], [], [], 0)[0]:
        ch1 = sys.stdin.read(1)
        if ch1 == '\x1b':
            return ch1 + sys.stdin.read(2)
        return ch1
    return None

# Initialize Genesis
gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=SCENE_DT), show_viewer=True)

# Ground plane
plane = scene.add_entity(gs.morphs.Plane(
    pos=(0,0,0), quat=(1,0,0,0), collision=True, visualization=True))

# Load drone URDF
starting_quat = xyz_to_quat(np.array([0,0.1,0]))
drone = scene.add_entity(gs.morphs.URDF(
    file="urdf/mydrone/bix3.urdf",
    pos=(-20,0,30), quat=starting_quat,
    collision=True, merge_fixed_links=True,
    links_to_keep=["fuselage", "aero_frame_fuselage",
                   "right_aileron_wing", "left_aileron_wing", "fuselage_rudder_0",
                   "rudder_wing", "fuselage_elevator_0", "elevator_wing",
                   "prop_frame_fuselage"],))
scene.build(n_envs=1, env_spacing=(1,1))

# Identify and debug servo joints
servo_joint_names = ["joint_right_aileron", "joint_left_aileron", "joint_0__rudder_wing", "joint_0__elevator_wing"]
# Map joint names to their DOF indices
joint_dof_map = {name: drone.get_joint(name).dof_idx_local for name in servo_joint_names}
# Flatten and sort for control
servo_dofs = [idx for name, idx in joint_dof_map.items()]
print("Servo joint to DOF mapping:")
for name, idx in joint_dof_map.items():
    print(f"  {name}: DOF index {idx}")

# Initialize servo positions to zero
drone.set_dofs_position(np.zeros((1, len(servo_dofs))), dofs_idx_local=servo_dofs)

# PD gains
drone.set_dofs_kp(np.array([2000.0]*len(servo_dofs)), dofs_idx_local=servo_dofs)
drone.set_dofs_kv(np.array([50.0]*len(servo_dofs)), dofs_idx_local=servo_dofs)

# Initial forward speed
rigid_solver = next(s for s in scene.sim.solvers if isinstance(s, RigidSolver))
# set linear velocity vx=7 m/s
init_vel = gs.tensor([[10,0,0,0,0,0,0,0,0,0]])
rigid_solver.set_dofs_velocity(init_vel)

# Terminal raw mode
dfd = sys.stdin.fileno()
old_attr = termios.tcgetattr(dfd)
tty.setcbreak(dfd)

throttle = 0.4
servo_targets = np.array([-0., 0., -0., -0.0])  # right aileron, left aileron, rudder, elevator

# Logs
logs = {key: [] for key in ['time','pos','rpy','thrust','servos','vel','alpha','beta','jpos']}

print("Controls: ↑/↓ throttle, J/L aileron, I/K elevator, U/O rudder, X reset, Ctrl-C exit")
try:
    sim_time = 0.0
    while True:
        key = read_key_nonblocking()
        if key:
            if key == KEY_THROTTLE_UP:
                throttle = min(1.0, throttle + STEP_THROTTLE)
            elif key == KEY_THROTTLE_DOWN:
                throttle = max(0.0, throttle - STEP_THROTTLE)
            elif key.lower() == KEY_ELE_PLUS:
                servo_targets[3] += STEP_ANG
            elif key.lower() == KEY_ELE_MINUS:
                servo_targets[3] -= STEP_ANG
            elif key.lower() == KEY_RUD_PLUS:
                servo_targets[2] += STEP_ANG
            elif key.lower() == KEY_RUD_MINUS:
                servo_targets[2] -= STEP_ANG
            elif key.lower() == KEY_AIL_PLUS:
                servo_targets[0] += STEP_ANG
                servo_targets[1] -= STEP_ANG
            elif key.lower() == KEY_AIL_MINUS:
                servo_targets[0] -= STEP_ANG
                servo_targets[1] += STEP_ANG
            elif key.lower() == KEY_RESET:
                servo_targets[:] = 0.0
        servo_targets = np.clip(servo_targets,
                                     np.deg2rad(-20), np.deg2rad(20))
            # Debug print current targets
        # Apply servo positions
        print(f"Throttle: {throttle:.2f}, Servos: {np.degrees(servo_targets)}")
        drone.control_dofs_position(servo_targets.reshape(1,-1), servo_dofs)

        # Compute aerodynamic wrench and angles
        F, M, alpha, beta = compute_aero_wrench(drone, throttle, max_thrust=MAX_THRUST_N)
        fuselage = drone.get_link("fuselage")
        solver = rigid_solver
        solver.apply_links_external_force(force=[[F]], links_idx=[fuselage.idx])
        solver.apply_links_external_torque(torque=[[M]], links_idx=[fuselage.idx])
        scene.step()
        sim_time += SCENE_DT

        # State and joint positions
        state = solver.get_state(f=None)
        qpos = state.qpos[0].cpu().numpy()
        vel = solver.get_dofs_velocity()[0,:3].cpu().numpy()
        # Explicitly request each joint from map for clarity
        jpos = np.array([drone.get_dofs_position(d)[0,0].item() for d in servo_dofs])
        pos = qpos[:3]
        quat = qpos[3:7]
        rpy = quat_to_xyz(quat)

        # Log data
        logs['time'].append(sim_time)
        logs['pos'].append(pos)
        logs['rpy'].append(rpy)
        logs['thrust'].append(throttle*MAX_THRUST_N)
        logs['servos'].append(servo_targets.copy())
        logs['vel'].append(vel)
        logs['alpha'].append(alpha)
        logs['beta'].append(beta)
        logs['jpos'].append(jpos)
        cam_pos = pos + np.array([-5,0,3])
        scene.viewer.set_camera_pose(pos=cam_pos, lookat=pos)
        scene.step()

except KeyboardInterrupt:
    pass
finally:
    termios.tcsetattr(dfd, termios.TCSADRAIN, old_attr)

# Convert logs to arrays
import numpy as _np

t = _np.array(logs['time'])
pos = _np.vstack(logs['pos'])
rpy = _np.vstack(logs['rpy'])
thr = _np.array(logs['thrust'])
serv = _np.vstack(logs['servos'])
vel = _np.vstack(logs['vel'])
alpha = _np.degrees(_np.array(logs['alpha']))
beta = _np.degrees(_np.array(logs['beta']))
jpos = _np.vstack(logs['jpos'])

# Plot
fig, axs = plt.subplots(9,1, figsize=(8,18), sharex=True)
axs[0].plot(t,pos); axs[0].set_ylabel('XYZ (m)')
axs[1].plot(t,_np.degrees(rpy)); axs[1].set_ylabel('RPY (°)')
axs[2].plot(t,thr); axs[2].set_ylabel('Thrust (N)')
axs[3].plot(t,_np.degrees(serv[:,0]),label='ail'); axs[3].plot(t,_np.degrees(serv[:,1]),label='rud')
axs[3].legend(); axs[3].set_ylabel('Servo targets (°)')
axs[4].plot(t,_np.degrees(serv[:,2])); axs[4].set_ylabel('Elev target (°)')
axs[5].plot(t,vel[:,0],label='vx'); axs[5].plot(t,vel[:,1],label='vy'); axs[5].plot(t,vel[:,2],label='vz')
axs[5].legend(); axs[5].set_ylabel('Velocity (m/s)')
axs[6].plot(t,alpha); axs[6].set_ylabel('Alpha (°)')
axs[7].plot(t,beta); axs[7].set_ylabel('Beta (°)')
axs[8].plot(t,_np.degrees(jpos[:,0]),label=f"{servo_joint_names[0]}")
axs[8].plot(t,_np.degrees(jpos[:,1]),label=f"{servo_joint_names[1]}")
axs[8].plot(t,_np.degrees(jpos[:,2]),label=f"{servo_joint_names[2]}")
axs[8].plot(t,_np.degrees(jpos[:,3]),label=f"{servo_joint_names[3]}")
axs[8].legend(); axs[8].set_ylabel('Joint pos (°)')
axs[8].set_xlabel('Time (s)')
plt.tight_layout()
plt.show()

