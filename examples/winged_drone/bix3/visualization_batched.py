#!/usr/bin/env python3
"""
Batched real-time Genesis simulation of Bixler-3 (B envs) without Python loops.
All state & control arrays are batched torch tensors on GPU.
"""
import sys, select, termios, tty
import torch
import genesis as gs
from genesis.engine.solvers import RigidSolver
from genesis.utils.geom import xyz_to_quat

# Batched aerodynamic wrench calculation
from aerodyn_solver import compute_aero_wrench

# --------------------------- Parameters --------------------------------------
B        = 1                  # number of environments
DT       = 0.01               # simulation timestep
MAX_THR  = 15.0               # max prop thrust [N]
DEVICE   = torch.device('cuda')
DTYPE    = torch.float32

# Control step sizes
STEP_THR = 0.025
STEP_ANG = torch.deg2rad(torch.tensor(0.2, device=DEVICE))
ANG_LIM  = torch.deg2rad(torch.tensor(20.0, device=DEVICE))

# Keyboard mapping
KEYS = {
    'thr_up': '\x1b[A', 'thr_dn': '\x1b[B',
    'ail_p':  'j',      'ail_m': 'l',
    'ele_p':  'i',      'ele_m': 'k',
    'rud_p':  'u',      'rud_m': 'o',
    'reset':  'x',
}

# ------------------------ Initialize Genesis -------------------------------
gs.init(backend=gs.gpu, logging_level="error")
scene = gs.Scene(gs.options.SimOptions(dt=DT), show_viewer=True)
scene.add_entity(
    gs.morphs.Plane(pos=(0,0,0), quat=(1,0,0,0), collision=True, visualization=True)
)

# URDF loader wants a CPU-side sequence, not a CUDA tensor
vec = torch.tensor([0.0, 0.1, 0.0])                         # incidence 0.1 rad
start_quat = xyz_to_quat(vec).cpu().numpy().tolist()        # to Python list

drone = scene.add_entity(gs.morphs.URDF(
    file='urdf/mydrone/bix3.urdf',
    pos=(-20,0,30), quat=start_quat,
    collision=True, merge_fixed_links=False,
    links_to_keep=[
        'fuselage','aero_frame_fuselage',
        'right_aileron_wing','left_aileron_wing',
        'fuselage_rudder_0','rudder_wing',
        'fuselage_elevator_0','elevator_wing',
        'prop_frame_fuselage',
    ]
))
scene.build(n_envs=B, env_spacing=(1,1))

# ------------------------- Servo Setup --------------------------------------
SERVOS     = [
    'joint_right_aileron','joint_left_aileron',
    'joint_0__rudder_wing','joint_0__elevator_wing'
]
servo_idx  = torch.tensor(
    [drone.get_joint(j).dof_idx_local for j in SERVOS],
    device=DEVICE
)
print('Servo DOF idx:', dict(zip(SERVOS, servo_idx.tolist())))


drone.set_dofs_kp(torch.full((len(SERVOS),),800.0,device=DEVICE), servo_idx)
drone.set_dofs_kv(torch.full((len(SERVOS),),  0.0,device=DEVICE), servo_idx)

# ------------------------ Initial State -------------------------------------
rigid        = next(s for s in scene.sim.solvers if isinstance(s, RigidSolver))
init_v       = torch.tensor([1,0,0,0,0,0,0,0,0,0],device=DEVICE).repeat(B,1)
print('Initial velocity:', init_v[0,0:3].tolist())
rigid.set_dofs_velocity(init_v)
drone.set_dofs_position(torch.zeros((B,len(SERVOS)),device=DEVICE), servo_idx)

throttle     = torch.full((B,),0.4,device=DEVICE)
servo_target = torch.tensor([0.0,0.0,0.0,-0.3],device=DEVICE).repeat(B,1)

# ------------------------ Keyboard Helper -----------------------------------
def read_key():
    if select.select([sys.stdin],[],[],0)[0]:
        c = sys.stdin.read(1)
        return c + sys.stdin.read(2) if c == '\x1b' else c
    return None

fd  = sys.stdin.fileno()
old = termios.tcgetattr(fd)
tty.setcbreak(fd)
print("Controls: ↑/↓ thr, J/L ail, I/K ele, U/O rud, X reset, Ctrl-C quit")

# -------------------------- Main Loop ---------------------------------------
try:
    while True:
        k = read_key()
        if k:
            if   k == KEYS['thr_up']:   throttle = (throttle + STEP_THR).clamp(0.0,1.0)
            elif k == KEYS['thr_dn']:   throttle = (throttle - STEP_THR).clamp(0.0,1.0)
            elif k == KEYS['ail_p']:    servo_target[:,0] += STEP_ANG; servo_target[:,1] -= STEP_ANG
            elif k == KEYS['ail_m']:    servo_target[:,0] -= STEP_ANG; servo_target[:,1] += STEP_ANG
            elif k == KEYS['ele_p']:    servo_target[:,3] += STEP_ANG
            elif k == KEYS['ele_m']:    servo_target[:,3] -= STEP_ANG
            elif k == KEYS['rud_p']:    servo_target[:,2] += STEP_ANG
            elif k == KEYS['rud_m']:    servo_target[:,2] -= STEP_ANG
            elif k == KEYS['reset']:    servo_target.zero_()
            servo_target = servo_target.clamp(-ANG_LIM, ANG_LIM)
            print(f"Thr {throttle[0]:.2f} | Servo°: {torch.rad2deg(servo_target[0]).tolist()}")
            print("joint pos =", torch.rad2deg(drone.get_dofs_position()[:, -4:]))

        # apply batched servo targets
        drone.control_dofs_position(servo_target, servo_idx)

        # compute batched aerodynamic wrench
        wrench = compute_aero_wrench(drone, throttle, max_thrust=MAX_THR)

        F, M = wrench['F'], wrench['M']        # (B,3) ciascuno

        # applico la forza/momento al solo link fusoliera
        fus = drone.get_link('fuselage')
        rigid.apply_links_external_force (F.unsqueeze(1), links_idx=[fus.idx])
        rigid.apply_links_external_torque(M.unsqueeze(1), links_idx=[fus.idx])

        # -----------------------------------------------------------------
        # 2. aggiorno le palline di debug (solo B == 1)
        # -----------------------------------------------------------------
        if B == 1:
            scene.clear_debug_objects()
            scene.draw_debug_sphere(pos=wrench['CG_world'][0].cpu(),
                                    radius=0.04, color=(1,1,0,1))  # giallo
            scene.draw_debug_sphere(pos=wrench['AC_world'][0].cpu(),
                                    radius=0.04, color=(0,0,0,1))  # nero

        # step physics + viewer
        pos = rigid.get_state(f=None).qpos[0,:3].cpu()
        scene.viewer.set_camera_pose(pos=pos + torch.tensor([-5,0,3]).cpu(), lookat=pos)
        scene.step()
except KeyboardInterrupt:
    pass
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
