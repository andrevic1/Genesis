import numpy as np
import torch
import math
from genesis.utils.geom import quat_to_R, axis_angle_to_R

device = torch.device("cuda")

# Constants and parameters (as in your code)
AIR_DENSITY = 1.225
A_FUSELAGE = 0.075
CHORD_FUSELAGE = 0.75
A_WING = 0.16
CHORD_WING = 0.2
MAX_THRUST = 15.0
CL_ALPHA_2D = 4 * math.pi
ALPHA_0_2D = -4.0 * (math.pi/180.0)
ALPHA_STALL = 12.0 * (math.pi/180.0)
M_SMOOTH = 0.1
CD_0 = 0.05
ALPHA_0_2D_FUS = 0.0
CG2CHORD = 0.33
CP_AERO_START = 0.25
CP_AERO_END = 0.5
CP_OFFSET_WING = 0.0
A_ELEVATOR = 0.025
CHORD_ELEVATOR = 0.1
AR_ELEVATOR = A_ELEVATOR / (CHORD_ELEVATOR**2)
A_RUDDER = 0.02
CHORD_RUDDER = 0.1
AR_RUDDER = A_RUDDER / (CHORD_RUDDER**2)
K_EPS_TAIL = 0.1
RATIO_PROP = 0.000092

def sigmoid_torch(x, x_cutoff, M):
    deg_x = x * (180.0/math.pi)
    deg_x_cutoff = x_cutoff * (180.0/math.pi)
    num = 1 + torch.exp(-M * (deg_x - deg_x_cutoff)) + torch.exp(M * (deg_x + deg_x_cutoff))
    den = (1 + torch.exp(-M * (deg_x - deg_x_cutoff))) * (1 + torch.exp(M * (deg_x + deg_x_cutoff)))
    return num/den

def AeroCoefficients(aero_frame, AR, alpha):
    c_l_alpha_3D = (CL_ALPHA_2D * AR) / (2 + torch.sqrt(torch.tensor(AR**2+4.0, device=device)))
    if aero_frame == "aero_frame_fuselage":
        c_l_lin = c_l_alpha_3D * (alpha - ALPHA_0_2D_FUS)
    else:
        c_l_lin = c_l_alpha_3D * (alpha - ALPHA_0_2D)
    c_d_lin = CD_0 + (c_l_lin**2)/(math.pi*AR)
    c_l_st = 2*torch.sin(alpha)*torch.cos(alpha)
    c_d_st = 2*(torch.sin(alpha))**2
    blend = sigmoid_torch(alpha, ALPHA_STALL, M_SMOOTH)
    c_l = (1-blend)*c_l_lin + blend*c_l_st
    c_d = (1-blend)*c_d_lin + blend*c_d_st
    return c_l, c_d

def compute_cp_offset(aero_frame, chord, alpha):
    if aero_frame in ["aero_frame_left_wing","aero_frame_right_wing"]:
        alpha_abs = torch.abs(alpha)
        cp_x = (-CG2CHORD + (alpha_abs/(math.pi/2))*(CP_AERO_END-CP_AERO_START)+CP_AERO_START)*chord+CP_OFFSET_WING
        return torch.tensor([cp_x, 0.0, 0.0], device=device)
    elif aero_frame == "aero_frame_fuselage":
        alpha_abs = torch.abs(alpha)
        cp_x = (-0.344 + (alpha_abs/(math.pi/2))*(CP_AERO_END-CP_AERO_START)+CP_AERO_START)*chord
        return torch.tensor([cp_x, 0.0, 0.0], device=device)
    elif aero_frame in ["elev_left"]:
        cp_x = -0.344 + CHORD_FUSELAGE + 0.25*CHORD_ELEVATOR
        return torch.tensor([cp_x, 0.25 * A_ELEVATOR/CHORD_ELEVATOR, 0.0], device=device)
    elif aero_frame in ["elev_right"]:
        cp_x = -0.344 + CHORD_FUSELAGE + 0.25*CHORD_ELEVATOR
        return torch.tensor([cp_x, -0.25 * A_ELEVATOR/CHORD_ELEVATOR, 0.0], device=device)
    elif aero_frame in ["rud"]:
        cp_x = -0.344 + CHORD_FUSELAGE + 0.25*CHORD_RUDDER
        cp_z = 0.5*A_RUDDER/CHORD_RUDDER
        return torch.tensor([cp_x, 0.0, cp_z], device=device)
    else:
        return torch.zeros(3, device=device)

def compute_aero_details_for_link(aero_frame, link, throttle=0.0):
    # Get velocity and orientation from the link
    v = link.get_vel()
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, device=device, dtype=torch.float32)
    else:
        v = v.to(device).float()
    
    q = link.get_quat()
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q, device=device, dtype=torch.float32)
    else:
        q = q.to(device).float()
    
    # Get rotation matrix from quaternion
    R = quat_to_R(q).squeeze()
    
    # Relative velocity (here assumed wind is zero, modify se necessario)
    rel_v = -v  # oppure aggiungere un vettore vento se richiesto
    speed = torch.norm(rel_v)
    if speed < 1e-3:
        # Se la velocità è troppo piccola, restituisce zeri per ogni quantità
        zeros = torch.zeros(3, device=device)
        return {"wrench": (zeros, zeros), "lift": torch.tensor(0.0, device=device),
                "drag": torch.tensor(0.0, device=device), "cp_offset": zeros, "alpha": torch.zeros(1, device=device), "beta": torch.zeros(1, device=device),}
    
    # Trasforma il vettore velocità in coordinate locali del link
    local_v = torch.matmul(R.t(), rel_v.T)
    Vinf_norm = torch.norm(local_v)
    alpha = torch.atan2(local_v[2], local_v[0])
    beta = torch.asin(local_v[1] / Vinf_norm)
    print(f"rel_v: {rel_v}")
    print (f"local_v: {local_v}")
    print(f"alpha: {alpha.item()}, beta: {beta.item()}")

    # Compute R_aero:
    # Per determinare l'orientamento aerodinamico, ruota attorno a y con -alpha ed attorno a z con beta.
    axis_y = np.array([0, 1, 0], dtype=np.float32)
    axis_z = np.array([0, 0, 1], dtype=np.float32)
    R_y_np = axis_angle_to_R(axis_y, -alpha.item())
    R_z_np = axis_angle_to_R(axis_z, beta.item())
    R_y = torch.tensor(R_y_np, device=device, dtype=torch.float32)
    R_z = torch.tensor(R_z_np, device=device, dtype=torch.float32)
    R_aero = torch.matmul(R_y, R_z)
    
    if aero_frame in ["aero_frame_left_wing", "aero_frame_right_wing"]:
        # --- Wing surface ---
        area = A_WING
        chord = CHORD_WING
        AR = area/(chord**2)
        c_l, c_d = AeroCoefficients(aero_frame, AR, alpha)

        L = 0.5 * area * AIR_DENSITY * c_l * (Vinf_norm**2)
        D = 0.5 * area * AIR_DENSITY * c_d * (Vinf_norm**2)
        cp_offset = compute_cp_offset(aero_frame, chord, alpha)

        F_wind_local = torch.tensor([D, 0.0, L], device=device)

        F_local = torch.matmul(R_aero, F_wind_local)
        M_local = torch.linalg.cross(cp_offset, F_local)
        force_world = torch.matmul(R, F_local)
        moment_world = torch.matmul(R, M_local)

        return {"wing": {"lift": L, "drag": D, "cp_offset": cp_offset, "alpha": alpha, "beta": beta,
                          "wrench": (force_world, moment_world)}}
    
    elif aero_frame == "aero_frame_fuselage":
        # --- Fuselage base ---
        area_fus = A_FUSELAGE
        chord_fus = CHORD_FUSELAGE
        AR_fus = area_fus/(chord_fus**2)
        c_l_fus, c_d_fus = AeroCoefficients("aero_frame_fuselage", AR_fus, alpha)
        L_fus = 0.5 * area_fus * AIR_DENSITY * c_l_fus * (Vinf_norm**2)
        D_fus = 0.5 * area_fus * AIR_DENSITY * c_d_fus * (Vinf_norm**2)
        cp_offset_fus = compute_cp_offset("aero_frame_fuselage", chord_fus, alpha)
        F_wind_local_fus = torch.tensor([D_fus, 0.0, L_fus], device=device)
        F_local_fus = torch.matmul(R_aero, F_wind_local_fus)
        M_local_fus = torch.linalg.cross(cp_offset_fus, F_local_fus)

        c_l_ref, _ = AeroCoefficients("aero_frame_right_wing", A_WING/(CHORD_WING**2), alpha)
        epsilon = K_EPS_TAIL * (c_l_ref / (math.pi * A_WING/(CHORD_WING**2)))
        alpha_tail_right = alpha - epsilon

        c_l_ref, _ = AeroCoefficients("aero_frame_left_wing", A_WING/(CHORD_WING**2), alpha)
        epsilon = K_EPS_TAIL * (c_l_ref / (math.pi * A_WING/(CHORD_WING**2)))
        alpha_tail_left = alpha - epsilon

        # Aerodynamic frame rotation
        R_y_np = axis_angle_to_R(axis_y, -alpha_tail_right.item())
        R_z_np = axis_angle_to_R(axis_z, beta.item())
        R_y = torch.tensor(R_y_np, device=device, dtype=torch.float32)
        R_z = torch.tensor(R_z_np, device=device, dtype=torch.float32)
        R_aero = torch.matmul(R_y, R_z)
        # --- Elevator ---

        c_l_elev, c_d_elev = AeroCoefficients("aero_frame_fuselage", AR_ELEVATOR, alpha_tail_right)
        L_elev = 0.5 * A_ELEVATOR/2 * AIR_DENSITY * c_l_elev * (Vinf_norm**2)
        D_elev = 0.5 * A_ELEVATOR/2 * AIR_DENSITY * c_d_elev * (Vinf_norm**2)
        cp_offset_elev = compute_cp_offset("elev_right", CHORD_ELEVATOR, alpha_tail_right)
        F_wind_local_elev = torch.matmul(R_aero, torch.tensor([D_elev, 0.0, L_elev], device=device))
        M_local_elev = torch.linalg.cross(cp_offset_elev, F_wind_local_elev)
        
        # Aerodynamic frame rotation
        R_y_np = axis_angle_to_R(axis_y, -alpha_tail_left.item())
        R_z_np = axis_angle_to_R(axis_z, beta.item())
        R_y = torch.tensor(R_y_np, device=device, dtype=torch.float32)
        R_z = torch.tensor(R_z_np, device=device, dtype=torch.float32)
        R_aero = torch.matmul(R_y, R_z)

        c_l_elev, c_d_elev = AeroCoefficients("aero_frame_fuselage", AR_ELEVATOR, alpha_tail_left)
        L_elev = 0.5 * A_ELEVATOR/2 * AIR_DENSITY * c_l_elev * (Vinf_norm**2)
        D_elev = 0.5 * A_ELEVATOR/2 * AIR_DENSITY * c_d_elev * (Vinf_norm**2)
        cp_offset_elev = compute_cp_offset("elev_left", CHORD_ELEVATOR, alpha_tail_left)
        F_wind_local_elev += torch.matmul(R_aero, torch.tensor([D_elev, 0.0, L_elev], device=device))
        M_local_elev += torch.linalg.cross(cp_offset_elev, torch.matmul(R_aero, torch.tensor([D_elev, 0.0, L_elev], device=device)))

        # --- Rudder ---
        c_l_rud, c_d_rud = AeroCoefficients("aero_frame_fuselage", AR_RUDDER, beta)
        D_rud = 0.5 * A_RUDDER * AIR_DENSITY * c_d_rud * (Vinf_norm**2)
        Y_rud = 0.5 * A_RUDDER * AIR_DENSITY * c_l_rud * (Vinf_norm**2)
        cp_offset_rud = compute_cp_offset("rud", CHORD_RUDDER, beta)
        F_wind_local_rud = torch.matmul(R_aero, torch.tensor([D_rud, Y_rud, 0.0], device=device))
        M_local_rud = torch.linalg.cross(cp_offset_rud, F_wind_local_rud)
        # --- Totals ---
        F_local_total = F_local_fus + F_wind_local_elev + F_wind_local_rud
        M_local_total = M_local_fus + M_local_elev + M_local_rud
        force_world = torch.matmul(R, F_local_total)
        moment_world = torch.matmul(R, M_local_total)
        return {"fuselage": {"lift": L_fus, "drag": D_fus, "cp_offset": cp_offset_fus, "alpha": alpha, "beta": beta,
                             "wrench": (torch.matmul(R, F_local_fus), torch.matmul(R, M_local_fus))},
                "elevator": {"lift": L_elev, "drag": D_elev, "cp_offset": cp_offset_elev, "alpha": alpha, "beta": beta,
                             "wrench": (torch.matmul(R, F_wind_local_elev), torch.matmul(R, M_local_elev))},
                "rudder":   {"lift": None, "drag": D_rud, "side_force": Y_rud, "cp_offset": cp_offset_rud, "alpha": alpha, "beta": beta,
                             "wrench": (torch.matmul(R, F_wind_local_rud), torch.matmul(R, M_local_rud))},
                "total": {"wrench": (force_world, moment_world)}}
    
    elif aero_frame == "prop_frame_fuselage_0":
        # --- Propeller ---
        thrust_dir_local = torch.tensor([0.0, 0.0, 1.0], device=device)
        thrust_force_local = throttle * MAX_THRUST * thrust_dir_local
        F_local = thrust_force_local
        M_local = torch.tensor([0, 0, throttle * MAX_THRUST * RATIO_PROP], device=device)
        force_world = torch.matmul(R, F_local)
        moment_world = torch.matmul(R, M_local)
        return {"prop_frame_fuselage_0": {"wrench": (force_world, moment_world)}}
    
    else:
        # Default: return zeros
        zeros = torch.zeros(3, device=device)
        return {"default": {"wrench": (zeros, zeros),
                            "lift": torch.tensor(0.0, device=device),
                            "drag": torch.tensor(0.0, device=device),
                            "cp_offset": zeros}}

def compute_aero_details(entity, throttle):
    """
    Given an entity (e.g., a drone) and a throttle value, compute a dictionary in cui
    per ogni superficie aerodinamica (fuselage, elevator, rudder, left wing, right wing, propeller)
    vengono restituiti i wrenches (force e moment), i Lift, i Drag e il cp_offset (posizione di applicazione).
    """
    # Retrieve the relevant links by name
    fuselage_link = entity.get_link("aero_frame_fuselage")
    left_wing_link = entity.get_link("aero_frame_left_wing")
    right_wing_link = entity.get_link("aero_frame_right_wing")
    prop_link = entity.get_link("prop_frame_fuselage_0")
    
    result = {}
    
    # Fuselage + elevator and rudder details
    details_fus = compute_aero_details_for_link("aero_frame_fuselage", fuselage_link, throttle)
    result.update(details_fus)
    
    # Left wing details
    details_left = compute_aero_details_for_link("aero_frame_left_wing", left_wing_link, throttle)
    result.update({"left_wing": details_left.get("wing", {})})
    
    # Right wing details
    details_right = compute_aero_details_for_link("aero_frame_right_wing", right_wing_link, throttle)
    result.update({"right_wing": details_right.get("wing", {})})
    
    # Propeller details
    details_prop = compute_aero_details_for_link("prop_frame_fuselage_0", prop_link, throttle)
    result.update(details_prop)
    
    return result
