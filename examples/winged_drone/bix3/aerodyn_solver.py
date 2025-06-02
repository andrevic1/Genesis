#!/usr/bin/env python3
"""
Batched aerodynamic + propulsive wrench for Bixler-3.
Pure torch GPU implementation, with correct batching of environments.
"""
import torch
from genesis.utils.geom import quat_to_R

DEVICE = torch.device('cuda')
DTYPE  = torch.float32

# Physical constants
RHO, S, c, b = 1.225, 0.276, 0.185, 1.54
MAX_THRUST   = 15.0
# Aerodynamic derivatives
CL0, CL_alpha, CL_q, CL_de = 0.39,    4.5321, 0.3180, 0.527
CD0, CD_alpha, CD_q, CD_de = 0.0765,  0.3346, 0.354,  0.004
CY0, CY_beta, CY_p, CY_r, CY_da, CY_dr = (0.0, -0.033, -0.100, 0.039, 0.0, 0.225)
Cl0, Cl_beta, Cl_p, Cl_r, Cl_da, Cl_dr = (0.0, -0.081, -0.529, 0.159, 0.453, 0.005)
Cm0, Cm_alpha, Cm_q, Cm_de = 0.0200, -1.4037, -0.1324, -0.4236
Cn0, Cn_beta, Cn_p, Cn_r, Cn_da, Cn_dr = (0.0, 0.189, -0.083, -0.948, 0.041, -0.077)

# Rodrigues rotation batch

def rot(axis: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation batch: axis (3,), ang (B,) → (B,3,3)"""
    # axis: local rotation axis in body frame
    axis = axis.to(DEVICE, DTYPE)
    x, y, z = axis[0], axis[1], axis[2]
    # broadcast angle
    c = torch.cos(ang)[:, None, None]
    s = torch.sin(ang)[:, None, None]
    t = 1 - c
    B = ang.shape[0]
    # identity
    I = torch.eye(3, device=DEVICE, dtype=DTYPE).unsqueeze(0).expand(B, 3, 3)
    # outer product axis ⊗ axis
    outer = (axis.view(1, 3, 1) * axis.view(1, 1, 3)).expand(B, 3, 3)
    # skew-symmetric matrix K for axis
    K = torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]],
                     device=DEVICE, dtype=DTYPE)
    K = K.unsqueeze(0).expand(B, 3, 3)
    # Rodrigues formula
    return c * I + s * K + t * outer

# ---------------------------------------------------------------------------
def compute_aero_wrench(drone, throttle: torch.Tensor, *, wind=None, max_thrust=MAX_THRUST):
    """
    Return batched aerodynamic wrench:
      F_w (B,3), M_w (B,3), alpha (B,), beta (B,)
    """
    # Ensure throttle is 1D batch
    if throttle.ndim != 1:
        raise ValueError("`throttle` must be a 1D tensor of shape (B,)")
    B = throttle.shape[0]
    throttle = throttle.to(DEVICE, DTYPE)

    # 1) Global CG per environment -----------------------------------------
    links = drone.links
    NL = len(links)
    # Stack positions and quaternions: shapes (NL, B, 3) and (NL, B, 4)
    pos_all  = torch.stack([L.get_pos() for L in links], dim=0).to(DEVICE, DTYPE).view(NL, B, 3)
    quat_all = torch.stack([L.get_quat() for L in links], dim=0).to(DEVICE, DTYPE).view(NL, B, 4)
    # Convert to rotation matrices: (NL*B,3,3) → reshape to (NL, B, 3,3)
    Rl = quat_to_R(quat_all.view(NL*B, 4)).view(NL, B, 3, 3)
    # Local COM offsets
        # Local COM offsets (per link) as torch tensor
    com_local = torch.stack([
        torch.tensor(L.inertial_pos, device=DEVICE, dtype=DTYPE)
        for L in links
    ], dim=0)  # (NL,3) for L in links], dim=0)  # (NL,3)
    # Broadcast and compute world COM: pos_all + Rl ∘ com_local
    com_world = pos_all + torch.einsum('nbij,nj->nbi', Rl, com_local)
    # Masses per link
    masses = torch.tensor([L.inertial_mass for L in links], device=DEVICE, dtype=DTYPE)  # (NL,)
    # Weighted average per environment
    numer = (com_world * masses[:,None,None]).sum(dim=0)    # (B,3)
    denom = masses.sum()                                    # scalar
    CG_global = numer / denom                            # (B,3)

    # 2) Fuselage pose + kinematics ----------------------------------------
    fus   = drone.get_link(name="aero_frame_fuselage")
    pf    = fus.get_pos().to(DEVICE).view(B,3)
    qf    = fus.get_quat().to(DEVICE).view(B,4)
    R_w2b = quat_to_R(qf)                                     # (B,3,3)

    # Velocities body-frame, angles alpha, beta
    v_w   = fus.get_vel().to(DEVICE).view(B,3)
    wind  = torch.zeros_like(v_w) if wind is None else wind.to(DEVICE)
    v_rel = wind - v_w
    v_body= torch.einsum('bij,bj->bi', R_w2b.transpose(1,2), v_rel)
    u, v, w = v_body.T
    Vmag     = v_body.norm(dim=1).clamp_min(1e-6)
    alpha    = torch.atan2(w, u)
    beta     = torch.asin(v / Vmag)

    # Angular rates p,q,r
    omega_w = fus.get_ang().to(DEVICE).view(B,3)
    pqr     = torch.einsum('bij,bj->bi', R_w2b.transpose(1,2), omega_w)
    p, q, r = pqr.T

    # 3) Servo deflections --------------------------------------------------
    DOFS = [
        "joint_right_aileron", "joint_left_aileron",
        "joint_0__rudder_wing", "joint_0__elevator_wing",
    ]
    dof_idx = torch.tensor([drone.get_joint(d).dof_idx_local for d in DOFS], device=DEVICE)
    actuators = drone.get_dofs_position(dof_idx).to(DEVICE)      # (B,4)
    delta_a = 0.5 *(actuators[:,0] - actuators[:,1])
    delta_e = actuators[:,3]
    delta_r = actuators[:,2]

    # 4) Aerodynamic coefficients (batched)-----------------------------------
    qS = 0.5 * RHO * Vmag**2 * S
    CL = CL0 + CL_alpha * alpha + CL_q  * c/(2*Vmag)*q + CL_de * delta_e
    CD = CD0 + CD_alpha * alpha + CD_q  * c/(2*Vmag)*q + CD_de * delta_e
    CY = (CY0 + CY_beta * beta + CY_p * b/(2*Vmag)*p + CY_r * b/(2*Vmag)*r
          + CY_da * delta_a + CY_dr * delta_r)
    Cl = (Cl0 + Cl_beta * beta + Cl_p * b/(2*Vmag)*p + Cl_r * b/(2*Vmag)*r
          + Cl_da * delta_a + Cl_dr * delta_r)
    Cm = Cm0 + Cm_alpha * alpha + Cm_q * c/(2*Vmag)*q + Cm_de * delta_e
    Cn = (Cn0 + Cn_beta * beta + Cn_p * b/(2*Vmag)*p + Cn_r * b/(2*Vmag)*r
          + Cn_da * delta_a + Cn_dr * delta_r)

    L = qS * CL;    D = qS * CD;    Y = qS * CY
    l = qS * c * Cl; m = qS * c * Cm; n = qS * c * Cn

    # 5) Convert aerodynamic forces/moments to body frame ------------------
    R_y = rot(torch.tensor([0,1,0], device=DEVICE), -alpha)
    R_z = rot(torch.tensor([0,0,1], device=DEVICE),  beta)
    R_wind = torch.bmm(R_y, R_z)                               # (B,3,3)

    F_a_body = torch.bmm(R_wind, torch.stack([D, Y, L], dim=1).unsqueeze(2)).squeeze(2)
    M_a_body = torch.stack([l, m, n], dim=1)

    # 6) Propulsive thrust in body frame -----------------------------------
    prop   = drone.get_link(name="prop_frame_fuselage")
    R_p    = quat_to_R(prop.get_quat().to(DEVICE).view(B,4))   # (B,3,3)
    thrust_dir = torch.einsum('bij,j->bi', R_p, torch.tensor([1.,0.,0.], device=DEVICE))
    T_body     = thrust_dir * throttle.unsqueeze(1) * max_thrust

    # 7) Total wrench in body & transform to world ------------------------
    F_body = F_a_body + T_body
    M_body = M_a_body + torch.cross(CG_global - pf, F_body, dim=1)

    F_world = torch.einsum('bij,bj->bi', R_w2b, F_body)
    M_world = torch.einsum('bij,bj->bi', R_w2b, M_body)

    F_mag2 = (F_body**2).sum(dim=1, keepdim=True).clamp_min(1e-6)  # evito div/0
    r_cp_body = torch.cross(F_body, M_body, dim=1) / F_mag2         # (B,3)

    # Trasformo il vettore in world e sommo al CG globale
    AC_world = CG_global + torch.einsum('bij,bj->bi', R_w2b, r_cp_body)

    # Se la forza è quasi nulla, colloco AC sul CG (caso stallo / lancio a mano)
    mask = (F_mag2.squeeze(1) < 1e-5)
    AC_world[mask] = CG_global[mask]

    # === 9) Valore di ritorno (tutto in un dizionario) ================
    return {
        'F'        : F_world,     # (B,3)
        'M'        : M_world,     # (B,3)
        'alpha'    : alpha,       # (B,)
        'beta'     : beta,        # (B,)
        'CG_world' : CG_global,   # (B,3)
        'AC_world' : AC_world,    # (B,3)
    }
