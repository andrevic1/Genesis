# -*- coding: utf-8 -*-
"""
Batched aerodynamic wrench solver – versione compatibile con URDF  ⟨fuselage + tail + rudder⟩

Richiede:
    PyTorch ≥ 1.13 (CUDA facoltativo)
    un entity-wrapper con .get_link(name).get_vel() / .get_quat()
"""

import math
import torch
from genesis.utils.geom import quat_to_xyz
from genesis.utils.geom import quat_to_R          # ⬅ per i link nel loop finale


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# — geometria e costanti di riferimento —
AIR_DENSITY       = 1.225
A_FUSELAGE        = 0.075
CHORD_FUSELAGE    = 0.75

A_WING            = 0.16
CHORD_WING        = 0.2

A_ELEVATOR        = 0.045          # intera superficie orizzontale (dx+sx)
CHORD_ELEVATOR    = 0.15
AR_ELEVATOR       = A_ELEVATOR / CHORD_ELEVATOR**2

A_RUDDER          = 0.020
CHORD_RUDDER      = 0.10
AR_RUDDER         = A_RUDDER / CHORD_RUDDER**2

MAX_THRUST        = 15.0           # N
RATIO_PROP        = 0.00092        # Nm / N

# — aerodinamica —
CL_ALPHA_2D     = 6 * math.pi
ALPHA_0_2D      = -2.0  * math.pi/180
ALPHA_0_2D_FUS  =  0.0
ALPHA_STALL     = 12.0 * math.pi/180
M_SMOOTH        = 0.1
CD_0            = 0.1

# — posizionamento centro di pressione in fusoliera (frame fusoliera) —
CG2CHORD        = 0.31
CP_AERO_START   = 0.25
CP_AERO_END     = 0.5

# — down-wash —
K_EPS_TAIL      = 0.1   # coefficiente lineare ε = K * Cl_w/πAR

# ---------------------------------------------------------------------------
#  Utility: quaternion ➔ rotation-matrix  (B,4) ➔ (B,3,3)
# ---------------------------------------------------------------------------

def quat_to_R_batch(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=1)
    R = torch.empty((*q.shape[:-1], 3, 3), dtype=q.dtype, device=q.device)
    R[:,0,0] = 1 - 2*(y*y + z*z)
    R[:,0,1] =     2*(x*y - z*w)
    R[:,0,2] =     2*(x*z + y*w)
    R[:,1,0] =     2*(x*y + z*w)
    R[:,1,1] = 1 - 2*(x*x + z*z)
    R[:,1,2] =     2*(y*z - x*w)
    R[:,2,0] =     2*(x*z - y*w)
    R[:,2,1] =     2*(y*z + x*w)
    R[:,2,2] = 1 - 2*(x*x + y*y)
    return R

# ---------------------------------------------------------------------------
#  Sigmoid blend per la transizione stall/lineare
# ---------------------------------------------------------------------------

def sigmoid_torch(x, x_cut, M):
    deg  = x * 180/math.pi
    cut  = x_cut * 180/math.pi
    num  = 1 + torch.exp(-M*(deg-cut)) + torch.exp( M*(deg+cut))
    den  = (1 + torch.exp(-M*(deg-cut))) * (1 + torch.exp(M*(deg+cut)))
    return num / den

# ---------------------------------------------------------------------------
#  Coefficienti aerodinamici 3D
# ---------------------------------------------------------------------------

def AeroCoefficients(frame: str, AR: float, alpha: torch.Tensor):
    AR = torch.as_tensor(AR, dtype=torch.float32, device=alpha.device)
    c_l_alpha_3D = (CL_ALPHA_2D * AR) / (2 + torch.sqrt(torch.tensor(AR**2+4.0, device=device)))
    if frame == "elevator" or frame == "rudder" or frame == "fuselage":
        c_l_lin = c_l_alpha_3D * (alpha - ALPHA_0_2D_FUS)
    else:
        c_l_lin = c_l_alpha_3D * (alpha - ALPHA_0_2D)
    c_d_lin = CD_0 + (c_l_lin**2)/(torch.pi*AR)
    cl_st  = 2 * torch.sin(alpha) * torch.cos(alpha)
    cd_st  = 2 * torch.sin(alpha)**2
    blend  = sigmoid_torch(alpha, ALPHA_STALL, M_SMOOTH)
    c_l    = (1-blend)*c_l_lin + blend*cl_st
    c_d    = (1-blend)*c_d_lin + blend*cd_st
    return c_l, c_d

# ---------------------------------------------------------------------------
#  CP offset in frame locale del link
# ---------------------------------------------------------------------------

def compute_cp_offset(frame: str,
                      chord: float,
                      alpha: torch.Tensor) -> torch.Tensor:
    """
    Centro di pressione (B,3) nel frame locale del link.
    """
    B       = alpha.shape[0]
    alpha_a = torch.abs(alpha)
    z0      = torch.zeros(B, device=alpha.device)

    # ---------------- ali ---------------------------------------------------
    if frame.endswith("wing"):
        cp_x = (-CG2CHORD
                + (alpha_a/(math.pi/2))*(CP_AERO_END-CP_AERO_START)
                + CP_AERO_START) * chord                       # (B,)
        return torch.stack((cp_x, z0, z0), dim=1)

    # ---------------- fusoliera --------------------------------------------
    if frame == "aero_frame_fuselage":
        cp_x = -0.344 + (
                + (alpha_a/(math.pi/2))*(CP_AERO_END-CP_AERO_START)
                + CP_AERO_START) * chord
        return torch.stack((cp_x, z0, z0), dim=1)

    # ---------------- elevator (due metà) ----------------------------------
    if frame in ("elev_left", "elev_right"):
        cp_x = (-CG2CHORD
                + (alpha_a/(math.pi/2))*(CP_AERO_END-CP_AERO_START)
                + CP_AERO_START) * chord + 0.2
        sign = +1.0 if frame == "elev_left" else -1.0
        cp_y = torch.full_like(cp_x, sign * 0.25 * A_ELEVATOR / CHORD_ELEVATOR)
        cp_z = z0 + 0.01
        return torch.stack((cp_x, cp_y, cp_z), dim=1)

    # ---------------- rudder -----------------------------------------------
    if frame == "rudder":
        cp_x = torch.full((B,), (-CG2CHORD + 0.25) * chord, device=alpha.device)
        cp_y = z0
        cp_z = torch.full((B,), 0.5 * A_RUDDER / CHORD_RUDDER, device=alpha.device)
        return torch.stack((cp_x, cp_y, z0), dim=1)

    # -----------------------------------------------------------------------
    return torch.zeros((B, 3), device=alpha.device)


# ---------------------------------------------------------------------------
#  Axis-angle ➔ rotation-matrix (batch)
# ---------------------------------------------------------------------------

def axis_angle_to_R_batch(axis, angles):
    B = angles.shape[0]
    ax = torch.tensor(axis, dtype=torch.float32, device=angles.device)
    ax = (ax / ax.norm()).unsqueeze(0).expand(B,3)
    cos = torch.cos(angles).view(B,1,1)
    sin = torch.sin(angles).view(B,1,1)
    I   = torch.eye(3, device=angles.device).unsqueeze(0).expand(B,3,3)
    v   = ax.view(B,3,1)
    outer = v @ v.transpose(1,2)
    a,b,c = ax[:,0], ax[:,1], ax[:,2]
    z = torch.zeros_like(a)
    skew = torch.stack([
        torch.stack([ z, -c,  b], dim=1),
        torch.stack([ c,  z, -a], dim=1),
        torch.stack([-b,  a,  z], dim=1)
    ], dim=1)
    return cos*I + sin*skew + (1-cos)*outer

# ---------------------------------------------------------------------------
#  Core per-link
# ---------------------------------------------------------------------------

def compute_aero_details_for_link(frame: str,
                                  link,
                                  throttle=0.0, entity=None):
    v = torch.as_tensor(link.get_vel(),  device=device, dtype=torch.float32)
    q = torch.as_tensor(link.get_quat(), device=device, dtype=torch.float32)
    euler = quat_to_xyz(q)
    B = v.size(0)

    # rotazioni
    R  = quat_to_R_batch(q)               # body→world
    vL = torch.bmm(R.transpose(1,2), (-v).unsqueeze(2)).squeeze(2)  # aria in body
    V  = vL.norm(dim=1)
    alpha = torch.atan2(vL[:,2], vL[:,0])
    beta  = torch.where(V>1e-6, torch.asin(vL[:,1]/V), torch.zeros_like(V))

    Ry = axis_angle_to_R_batch([0,1,0], -alpha)
    Rz = axis_angle_to_R_batch([0,0,1],  beta)
    Rw = torch.bmm(Ry, Rz)   # wind→body

    thr = throttle if torch.is_tensor(throttle) \
          else torch.full((B,), float(throttle), device=device)

    # ------------------------------------------------------------------ wing
    if frame in ("aero_frame_left_wing", "aero_frame_right_wing"):
        AR = A_WING/CHORD_WING**2
        cl, cd = AeroCoefficients("wing", AR, alpha)
        L = 0.5 * AIR_DENSITY * A_WING * cl * V**2
        D = 0.5 * AIR_DENSITY * A_WING * cd * V**2
        cp = compute_cp_offset(frame, CHORD_WING, alpha)
        F_wind = torch.stack(( D, torch.zeros_like(D), L), dim=1)
        F_b = torch.bmm(Rw, F_wind.unsqueeze(2)).squeeze(2)
        M_b = torch.cross(cp, F_b, dim=1)
        F_w = torch.bmm(R, F_b.unsqueeze(2)).squeeze(2)
        M_w = torch.bmm(R, M_b.unsqueeze(2)).squeeze(2)
        name = "left_wing" if "left" in frame else "right_wing"
        return {name: {"lift": L, "drag": D, "cp_offset": cp,
                       "alpha": alpha, "beta": beta,
                       "wrench": (F_w, M_w)}}

    # -------------------------------------------------------------- fuselage
    if frame == "aero_frame_fuselage":
        ARf = A_FUSELAGE/CHORD_FUSELAGE**2
        cl, cd = AeroCoefficients(frame, ARf, alpha)
        L = 0.5 * AIR_DENSITY * A_FUSELAGE * cl * V**2
        D = 0.5 * AIR_DENSITY * A_FUSELAGE * cd * V**2
        cp = compute_cp_offset(frame, CHORD_FUSELAGE, alpha)
        F_wind = torch.stack((D, torch.zeros_like(D), L), dim=1)
        F_b = torch.bmm(Rw, F_wind.unsqueeze(2)).squeeze(2)
        M_b = torch.cross(cp, F_b, dim=1)
        F_w = torch.bmm(R, F_b.unsqueeze(2)).squeeze(2)
        M_w = torch.bmm(R, M_b.unsqueeze(2)).squeeze(2)
        return {"fuselage": {"lift": L, "drag": D, "cp_offset": cp,
                             "alpha": alpha, "beta": beta,
                             "wrench": (F_w, M_w)}}

    # ---------------------------------------------------------------- tail - elevator
    # ---------------------------------------------------------------- elevator (due semi-ali) ----------
    if frame == "aero_frame_elevator":
        # down-wash causato dalle wings
        ARw = A_WING / CHORD_WING**2

        link_name = "aero_frame_left_wing"
        link = entity.get_link(link_name)
        v = torch.as_tensor(link.get_vel(),  device=device, dtype=torch.float32)
        q = torch.as_tensor(link.get_quat(), device=device, dtype=torch.float32)
        # rotazioni
        R_w  = quat_to_R_batch(q)               # body→world
        vL = torch.bmm(R_w.transpose(1,2), (-v).unsqueeze(2)).squeeze(2)  # aria in body
        alpha_wing_l = torch.atan2(vL[:,2], vL[:,0])
        
        link_name = "aero_frame_right_wing"
        link = entity.get_link(link_name)
        v = torch.as_tensor(link.get_vel(),  device=device, dtype=torch.float32)
        q = torch.as_tensor(link.get_quat(), device=device, dtype=torch.float32)
        # rotazioni
        R_w  = quat_to_R_batch(q)               # body→world
        vL = torch.bmm(R_w.transpose(1,2), (-v).unsqueeze(2)).squeeze(2)  # aria in body
        alpha_wing_r = torch.atan2(vL[:,2], vL[:,0])


        cl_wr, _ = AeroCoefficients("aero_frame_right_wing", ARw, alpha_wing_r)
        eps = K_EPS_TAIL * cl_wr / (math.pi * ARw)
        alpha_eff_r = alpha - eps         
        
        cl_wl, _ = AeroCoefficients("aero_frame_left_wing", ARw, alpha_wing_l)
        eps = K_EPS_TAIL * cl_wl / (math.pi * ARw)
        alpha_eff_l = alpha - eps                        # (B,)

        # coeff. aero
        cl_e_r, cd_e_r = AeroCoefficients("elevator", AR_ELEVATOR, alpha_eff_r)
        # superfici dimezzate
        S_half = A_ELEVATOR / 2.0
        L_half = 0.5 * AIR_DENSITY * S_half * cl_e_r * V**2
        D_half = 0.5 * AIR_DENSITY * S_half * cd_e_r * V**2
        Fwind_L = torch.stack((D_half, torch.zeros_like(D_half), L_half), dim=1)

        # rotazione wind→body con α_eff e β
        Ry = axis_angle_to_R_batch([0,1,0], -alpha_eff_r)
        Rw_elev_r = torch.bmm(Ry, Rz)

        # coeff. aero
        cl_e_l, cd_e_l = AeroCoefficients("elevator", AR_ELEVATOR, alpha_eff_l)
        # superfici dimezzate
        S_half = A_ELEVATOR / 2.0
        L_half = 0.5 * AIR_DENSITY * S_half * cl_e_l * V**2
        D_half = 0.5 * AIR_DENSITY * S_half * cd_e_l * V**2
        Fwind_R = torch.stack((D_half, torch.zeros_like(D_half), L_half), dim=1)

        # rotazione wind→body con α_eff e β
        Ry = axis_angle_to_R_batch([0,1,0], -alpha_eff_l)
        Rw_elev_l = torch.bmm(Ry, Rz)

        # ---------- semiala SX (y > 0) ----------
        cp_L = compute_cp_offset("elev_left", CHORD_ELEVATOR, alpha_eff_l)   # (B,3)
        Fb_L = torch.bmm(Rw_elev_l, Fwind_L.unsqueeze(2)).squeeze(2)
        Mb_L = torch.cross(cp_L, Fb_L, dim=1)
        Fw_L = torch.bmm(R, Fb_L.unsqueeze(2)).squeeze(2)
        Mw_L = torch.bmm(R, Mb_L.unsqueeze(2)).squeeze(2)

        # ---------- semiala DX (y < 0) ----------
        cp_R = compute_cp_offset("elev_right", CHORD_ELEVATOR, alpha_eff_r)
        # (stesse forze perché stessa superficie)
        Fb_R = torch.bmm(Rw_elev_r, Fwind_R.unsqueeze(2)).squeeze(2)
        Mb_R = torch.cross(cp_R, Fb_R, dim=1)
        Fw_R = torch.bmm(R, Fb_R.unsqueeze(2)).squeeze(2)
        Mw_R = torch.bmm(R, Mb_R.unsqueeze(2)).squeeze(2)

        return {
            "elevator_left":  {"lift": L_half, "drag": D_half, "cp_offset": cp_L,
                            "alpha": alpha_eff_l, "beta": beta,
                            "wrench": (Fw_L, Mw_L)},
            "elevator_right": {"lift": L_half, "drag": D_half, "cp_offset": cp_R,
                            "alpha": alpha_eff_r, "beta": beta,
                            "wrench": (Fw_R, Mw_R)},
        }


    # ---------------------------------------------------------------- rudder
    if frame == "aero_frame_rudder":
        cl, cd = AeroCoefficients("rudder", AR_RUDDER, beta)
        D  = 0.5 * AIR_DENSITY * A_RUDDER * cd * V**2
        Y  = 0.5 * AIR_DENSITY * A_RUDDER * cl * V**2
        cp = compute_cp_offset("rudder", CHORD_RUDDER, beta)

        F_wind = torch.stack((D, Y, torch.zeros_like(D)), dim=1)
        F_b = torch.bmm(Rw, F_wind.unsqueeze(2)).squeeze(2)
        M_b = torch.cross(cp, F_b, dim=1)
        F_w = torch.bmm(R, F_b.unsqueeze(2)).squeeze(2)
        M_w = torch.bmm(R, M_b.unsqueeze(2)).squeeze(2)
        return {"rudder": {"drag": D, "side_force": Y, "cp_offset": cp,
                           "beta": beta, "alpha": alpha,
                           "wrench": (F_w, M_w)}}

    # ------------------------------------------------------------- propeller
    if frame == "prop_frame_fuselage_0":
        dirv = torch.tensor([0,0,1], device=device, dtype=torch.float32)
        F_b = dirv.unsqueeze(0) * (thr.unsqueeze(1) * MAX_THRUST)
        M_b = dirv.unsqueeze(0) * (thr * MAX_THRUST * RATIO_PROP).unsqueeze(1)
        F_w = torch.bmm(R, F_b.unsqueeze(2)).squeeze(2)
        M_w = torch.bmm(R, M_b.unsqueeze(2)).squeeze(2)
        return {"prop_frame_fuselage_0": {"wrench": (F_w, M_w)}}

    return {}

# ---------------------------------------------------------------------------
#  Entry-point
# ---------------------------------------------------------------------------

def compute_aero_details(entity, throttle):
    """
    Calcola e restituisce i wrench aero per:
      fuselage, wings, elevator (sx/dx), rudder, propeller
    """
    frames = {
        "aero_frame_fuselage"   : "aero_frame_fuselage",
        "aero_frame_left_wing"  : "aero_frame_left_wing",
        "aero_frame_right_wing" : "aero_frame_right_wing",
        "aero_frame_elevator"   : "aero_frame_elevator",
        "aero_frame_rudder"     : "aero_frame_rudder",
        "prop_frame_fuselage_0" : "prop_frame_fuselage_0"
    }
    out = {}
    for frame, link_name in frames.items():
        out.update(
            compute_aero_details_for_link(frame,
                                          entity.get_link(link_name),
                                          throttle, 
                                          entity)
        )

    # ---------- CG globale ---------------------------------------------
    links   = entity.links
    NL      = len(links)

    pos_all  = torch.stack([torch.as_tensor(L.get_pos(),  device=device, dtype=torch.float32)
                            for L in links])                 # (NL,B,3)
    quat_all = torch.stack([torch.as_tensor(L.get_quat(), device=device, dtype=torch.float32)
                            for L in links])                 # (NL,B,4)

    R_links  = quat_to_R_batch(quat_all.view(-1,4)).view(NL, -1, 3, 3)
    com_loc  = torch.stack([torch.tensor(L.inertial_pos, device=device, dtype=torch.float32)
                            for L in links])                 # (NL,3)
    masses   = torch.tensor([L.inertial_mass for L in links],
                             device=device, dtype=torch.float32)                  # (NL,)

    com_w = pos_all + torch.einsum('nbij,nj->nbi', R_links, com_loc)
    CG    = (com_w * masses[:,None,None]).sum(dim=0) / masses.sum()  # (B,3)


    # ---------- somma forze & momenti aerodinamici ----------------------
    F_tot = torch.zeros_like(CG)
    M_tot = torch.zeros_like(CG)
    key2frame = {
        "fuselage"      : "aero_frame_fuselage",
        "left_wing"     : "aero_frame_left_wing",
        "right_wing"    : "aero_frame_right_wing",
        "elevator_left" : "aero_frame_elevator",
        "elevator_right": "aero_frame_elevator",
        "rudder"        : "aero_frame_rudder",
    }

    for key, data in out.items():
        if key.startswith("prop") or "cp_offset" not in data:
            continue                                         # thrust fuori

        F_w, _  = data["wrench"]                            # (B,3)
        cp_loc  = data["cp_offset"]                         # (B,3)

        link = entity.get_link(key2frame[key])
        p_link   = torch.as_tensor(link.get_pos(),  device=device)  # (B,3)
        R_link   = quat_to_R_batch(torch.as_tensor(link.get_quat(),
                                                   device=device))   # (B,3,3)
        cp_world = p_link + torch.bmm(R_link, cp_loc.unsqueeze(2)).squeeze(2)

        r = cp_world - CG
        F_tot += F_w
        M_tot += torch.cross(r, F_w, dim=1)

    # ---------- centro aerodinamico globale ----------------------------
    eps     = 1e-6
    r_AC    = torch.cross(F_tot, M_tot, dim=1) / \
              (F_tot.norm(dim=1).clamp_min(eps).unsqueeze(1)**2)
    AC      = CG + r_AC

    # --------- append al dict di uscita --------------------------------
    out["CG"] = CG
    out["AC"] = AC
    return out
