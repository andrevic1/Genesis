# aerodynamic_wrench_calculation.py
#!/usr/bin/env python3
"""
aerodynamic_wrench_calculation.py

Compute aerodynamic + propulsive wrench for Bixler-3 about its fuselage COM.

Returns:
    F (3,) torque      : Force vector in world frame
    M (3,) torque      : Moment vector in world frame, at fuselage COM
"""


import numpy as np
import torch
import genesis as gs
from genesis.utils.geom import quat_to_R, quat_to_xyz, axis_angle_to_R

# Aero parameters (match C++ BixlerDynamics)
RHO = 1.225
S = 0.276
c = 0.185
b = 1.54
kappa = 0.016
CL0 = 0.3900; CL_alpha = 4.5321; CL_q = 0.3180; CL_de = 0.527
CD0 = 0.0765; CD_alpha = 0.3346; CD_q = 0.354; CD_de = 0.004
CY0 = 0.0; CY_beta = -0.033; CY_p = -0.100; CY_r = 0.039; CY_da = 0.0; CY_dr = 0.225
Cl0 = 0.0; Cl_beta = -0.081; Cl_p = -0.529; Cl_r = 0.159; Cl_da = 0.453; Cl_dr = 0.005
Cm0 = 0.0200; Cm_alpha = -1.4037; Cm_q = -0.1324; Cm_de = -0.4236
Cn0 = 0.0; Cn_beta = 0.189; Cn_p = -0.083; Cn_r = -0.948; Cn_da = 0.041; Cn_dr = -0.077

def compute_aero_wrench(drone, throttle, wind=None, max_thrust=15.0):
    # 1) CG globale via Genesis:
    #    - drone.links          → lista di RigidLink
    #    - get_links_pos()      → (n_links,3) tensor di COM world-frame
    #    - RigidLink.inertial_mass
    links  = drone.links
    masses = np.array([L.inertial_mass for L in links])
    coms    = []
    for L in links:
            p = L.get_pos().cpu().numpy()
            q = L.get_quat().cpu().numpy()
            R = quat_to_R(torch.tensor(q)).cpu().numpy()
            com_local = L.inertial_pos             # estrai offset COM dal link
            com_world = p + R.dot(com_local)
            coms.append(com_world)
    coms = np.vstack(coms)                        # shape (n_links,3)
    total_mass = masses.sum()
    CG_global = (coms * masses[:,None]).sum(axis=0) / total_mass

    # 2) Fuselage link pos/quaternion
    fus     = drone.get_link(name="aero_frame_fuselage")
    pos_fuse= fus.get_pos().cpu().numpy().squeeze()
    quat    = fus.get_quat().cpu().numpy()
    R_w2b   = quat_to_R(torch.tensor(quat)).cpu().numpy().squeeze()
    euler   = quat_to_xyz(torch.tensor(quat)).cpu().numpy()

    # 3) Velocità body-frame, α, β
    v_world  = fus.get_vel().cpu().numpy().squeeze()
    v_wind  = (wind if wind is not None else np.zeros(3)) - v_world
    vel_body = R_w2b.T.dot(v_wind)
    u,v,w   = vel_body

    V = np.linalg.norm(vel_body)
    alpha    = np.arctan2(w,u) if V>1e-6 else 0.0
    beta     = np.arcsin(v/V)  if V>1e-6 else 0.0

    # 4) Rates p,q,r in body
    omega_body = R_w2b.T.dot(fus.get_ang().cpu().numpy().squeeze())
    p,q,r      = omega_body

    # 5) Deflessioni servo & motor
    servo_joint_names = ["joint_right_aileron", "joint_left_aileron", "joint_0__rudder_wing", "joint_0__elevator_wing"]
    servo_dofs = sorted(drone.get_joint(j).dof_idx_local for j in servo_joint_names)
    actuators = drone.get_dofs_position(servo_dofs).cpu().numpy()

    delta_a = (actuators[0, 0] - actuators[0, 1])
    delta_e = actuators[0, 3]
    delta_r = actuators[0, 2]

    # 6) Coefficienti aerodinamici (lineari)
    CL = CL0 + CL_alpha*alpha + CL_q*c/(2*V)*q + CL_de*delta_e
    CD = CD0 + CD_alpha*alpha + CD_q*c/(2*V)*q + CD_de*delta_e
    CY = (CY0 + CY_beta*beta + CY_p*b/(2*V)*p + CY_r*b/(2*V)*r
          + CY_da*delta_a + CY_dr*delta_r)
    Cl = Cl0 + Cl_beta*beta + Cl_p*b/(2*V)*p + Cl_r*b/(2*V)*r + Cl_da*delta_a + Cl_dr*delta_r
    Cm = Cm0 + Cm_alpha*alpha + Cm_q*c/(2*V)*q + Cm_de*delta_e
    Cn = Cn0 + Cn_beta*beta + Cn_p*b/(2*V)*p + Cn_r*b/(2*V)*r + Cn_da*delta_a + Cn_dr*delta_r
    # 7) Lift, Drag, Side forces + aero moments in wind frame
    L = 0.5*RHO*V**2*S*CL
    D = 0.5*RHO*V**2*S*CD
    Y = 0.5*RHO*V**2*S*CY
    l = 0.5*RHO*V**2*S*c*Cl
    m = 0.5*RHO*V**2*S*c*Cm
    n = 0.5*RHO*V**2*S*c*Cn

    axis_y = np.array([0, 1, 0], dtype=np.float32)
    axis_z = np.array([0, 0, 1], dtype=np.float32)
    R_y_np = axis_angle_to_R(axis_y, -alpha.item())
    R_z_np = axis_angle_to_R(axis_z, beta.item())
    R_wind = R_y_np.dot(R_z_np)
    F_a = R_wind.dot(np.array([D, Y, L]))
    M_a = np.array([l, m, n])

    # 8) Spinta propulsiva dal link prop_frame_fuselage (rispetta l'origin.rpy URDF)
    prop_link = drone.get_link(name="prop_frame_fuselage")
    quat_p     = prop_link.get_quat().cpu().numpy()
    R_p        = quat_to_R(torch.tensor(quat_p)).cpu().numpy()
    thrust_dir = R_p.dot(np.array([1.0, 0.0, 0.0]))
    T_body     = thrust_dir * (throttle * max_thrust)

    # 9) Forza totale in world
    F_tot = R_w2b.dot(F_a + T_body.squeeze())

    # 10) Momento totale **traslato** da CG_global a pos_fuse
    M_a_w = R_w2b.dot(M_a)
    r     = CG_global - pos_fuse
    M_tot = M_a_w + np.cross(r, R_w2b.dot(F_a + T_body.squeeze()))

    print(f"F_tot: {F_tot}")
    print(f"M_tot: {M_tot}")

    return F_tot, M_tot, alpha, beta