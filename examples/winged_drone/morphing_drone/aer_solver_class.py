# -*- coding: utf-8 -*-
"""
Batched aerodynamic wrench solver – class-based version compatible with URDF
⟨fuselage + wings + elevator + rudder + propeller⟩.

All the formerly-global constants live in the constructor, so you can spawn
different solvers (e.g. one on CUDA, one on CPU) without side-effects.

Dependencies
------------
* PyTorch ≥ 1.13 (CUDA optional)
* genesis.utils.geom.quat_to_xyz / quat_to_R
"""

from __future__ import annotations

import math
from typing import Dict, Any
from pathlib import Path

import torch
from genesis.utils.geom import quat_to_xyz, quat_to_R  # quat_to_R used only once
import xml.etree.ElementTree as ET

# --------------------------------------------------------------------------- #
#  Main class                                                                 #
# --------------------------------------------------------------------------- #
class AerodynamicSolver:
    """Batched aerodynamic wrench solver regrouped into a single class."""

    # --------------------------------------------------------------------- #
    #  Construction                                                         #
    # --------------------------------------------------------------------- #
    def __init__(self, *, device: torch.device | None = None, urdf_file: str | None = None):
        # choose device lazily
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.urdf_file = urdf_file

        # ── geometry  ────────────────────────────────────────────────────
        self.AIR_DENSITY = 1.225
        self.MAX_THRUST = 15.0          # N
        self.RATIO_PROP = 0.00092       # Nm / N

        # ── aerodynamics  ────────────────────────────────────────────────
        self.CL_ALPHA_2D = 6 * math.pi
        self.ALPHA_0_2D = -0.0 * math.pi / 180
        self.ALPHA_0_2D_FUS = 0.0
        self.ALPHA_STALL = 16.0 * math.pi / 180
        self.M_SMOOTH = 0.1
        self.CD_0 = 0.1

        # ── CP placement / down-wash  ─────────────────────────────────────
        self.CP_AERO_START = 0.25
        self.CP_AERO_END = 0.5
        self.K_EPS_TAIL = 0.1           # ε = K · Cl_w / πAR

        self.A_FUSELAGE = self.CHORD_FUSELAGE = 0.0
        self.A_WING     = self.CHORD_WING     = 0.0
        self.A_ELEVATOR = self.CHORD_ELEVATOR = 0.0
        self.A_RUDDER   = self.CHORD_RUDDER   = 0.0
        self.CG_fuselage = torch.zeros(3)
        self.CG_tochord     = 0.31

        if urdf_file:
            self.parameters_from_collision_boxes(Path(urdf_file))

    # ------------------------------------------------------------------ #
    #  Small helpers (static)                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def quat_to_R_batch(q: torch.Tensor) -> torch.Tensor:
        """Quaternion → rotation matrix.  (B,4) → (B,3,3)"""
        w, x, y, z = q.unbind(1)
        R = torch.empty((*q.shape[:-1], 3, 3), dtype=q.dtype, device=q.device)
        R[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)
        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
        R[:, 1, 2] = 2 * (y * z - x * w)
        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)
        R[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)
        return R
    
    def parameters_from_collision_boxes(self, urdf_path: Path):
        """
        Calcola superficie planare e corda primaria di fusoliera, ali, elevator
        e timone analizzando le <collision><geometry><box size="…"/> del file URDF.
        """
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # helper rapido -------------------------------------------------
        def box_dims(link_name: str):
            link = root.find(f".//link[@name='{link_name}']")
            if link is None:
                raise ValueError(f"Link <{link_name}> non trovato nell’URDF!")
            box = link.find(".//collision/geometry/box")
            if box is None:
                raise ValueError(f"Link <{link_name}> non contiene una <box> collision!")
            sx, sy, sz = map(float, box.get("size").split())
            print(f"  {link_name}: size=({sx:.3f}, {sy:.3f}, {sz:.3f})")
            return (sx, sy, sz)

        # -------- fusoliera -------------------------------------------
        s_fus = box_dims("fuselage")
        self.CHORD_FUSELAGE = s_fus[2]               # lunghezza
        self.A_FUSELAGE     = s_fus[0] * s_fus[2]    # area frontale
        # CG locale nel frame fusoliera (inertial/origin)
        fus_inertial = root.find(".//link[@name='fuselage']/inertial/origin")
        self.CG_fuselage = torch.tensor(
            list(map(float, fus_inertial.get("xyz").split())),
            dtype=torch.float32,
            device=self.device,
        )

        # -------- ali (somma dx+sx) -----------------------------------
        s_w   = box_dims("right_wing")               # destra == sinistra
        chord, span = s_w[1], s_w[2]
        self.CHORD_WING = chord
        self.A_WING     = chord * span           # due semiali
        wing_inertial = root.find(".//link[@name='right_wing']/inertial/origin")

        # -------- elevator --------------------------------------------
        s_e   = box_dims("elevator")
        chord, span = s_e[1], s_e[2]
        self.CHORD_ELEVATOR = chord
        self.A_ELEVATOR     = chord * span           # intera superf.
        elevator_inertial = root.find(".//link[@name='elevator']/inertial/origin")


        # -------- rudder ----------------------------------------------
        s_r   = box_dims("rudder")
        chord, span = s_r[1], s_r[2]
        self.CHORD_RUDDER = chord
        self.A_RUDDER     = chord * span
        rudder_inertial = root.find(".//link[@name='rudder']/inertial/origin")
        self.CG_rudder = torch.tensor(
            list(map(float, rudder_inertial.get("xyz").split())),
            dtype=torch.float32,
            device=self.device,
        )


        print("[Aero] Parsed surfaces from URDF:")
        print(f"  Fuselage  A={self.A_FUSELAGE:.3f}  chord={self.CHORD_FUSELAGE:.3f}  cg={self.CG_fuselage[0]}")
        print(f"  Wing      A={self.A_WING:.3f}      chord={self.CHORD_WING:.3f}      ")
        print(f"  Elevator  A={self.A_ELEVATOR:.3f}  chord={self.CHORD_ELEVATOR:.3f}  ")
        print(f"  Rudder    A={self.A_RUDDER:.3f}    chord={self.CHORD_RUDDER:.3f}    ")

    @staticmethod
    def axis_angle_to_R_batch(axis, angles):
        B = angles.size(0)
        ax = torch.as_tensor(axis, dtype=torch.float32, device=angles.device)
        ax = (ax / ax.norm()).expand(B, 3)
        cos, sin = torch.cos(angles)[:, None, None], torch.sin(angles)[:, None, None]
        I = torch.eye(3, device=angles.device).expand(B, 3, 3)
        v = ax[..., None]
        outer = v @ v.transpose(1, 2)
        a, b, c = ax.T
        z = torch.zeros_like(a)
        skew = torch.stack(
            [torch.stack([z, -c, b], 1), torch.stack([c, z, -a], 1), torch.stack([-b, a, z], 1)], 1
        )
        return cos * I + sin * skew + (1 - cos) * outer

    # ------------------------------------------------------------------ #
    #  Stall/linear blending sigmoid                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sigmoid(x, x_cut, M):
        deg = x * 180 / math.pi
        cut = x_cut * 180 / math.pi
        num = 1 + torch.exp(-M * (deg - cut)) + torch.exp(M * (deg + cut))
        den = (1 + torch.exp(-M * (deg - cut))) * (1 + torch.exp(M * (deg + cut)))
        return num / den

    # ------------------------------------------------------------------ #
    #  3-D aerodynamic coefficients C_L , C_D                            #
    # ------------------------------------------------------------------ #
    def aero_coefficients(self, frame: str, AR: float, alpha: torch.Tensor):
        AR = torch.as_tensor(AR, dtype=torch.float32, device=alpha.device)
        cl_a = self.CL_ALPHA_2D * AR / (2 + torch.sqrt(AR**2 + 4.0))
        ref = self.ALPHA_0_2D_FUS if frame in {"elevator", "rudder", "fuselage"} else self.ALPHA_0_2D
        cl_lin = cl_a * (alpha - ref)
        cd_lin = self.CD_0 + (cl_lin**2) / (math.pi * AR)

        cl_st = 2 * torch.sin(alpha) * torch.cos(alpha)
        cd_st = 2 * torch.sin(alpha) ** 2
        blend = self._sigmoid(alpha, self.ALPHA_STALL, self.M_SMOOTH)

        cl = (1 - blend) * cl_lin + blend * cl_st
        cd = (1 - blend) * cd_lin + blend * cd_st
        return cl, cd

    # ------------------------------------------------------------------ #
    #  Local CP offset                                                   #
    # ------------------------------------------------------------------ #
    def cp_offset(self, frame: str, alpha: torch.Tensor):
        B = alpha.size(0)
        aa = alpha.abs()
        z0 = torch.zeros(B, device=alpha.device)

        if frame.endswith("wing"):
            cx = (- self.CG_tochord
                + aa / (math.pi / 2) * (self.CP_AERO_END - self.CP_AERO_START)
                + self.CP_AERO_START
            ) * self.CHORD_WING

            return torch.stack((cx, z0, z0), 1)

        if frame == "aero_frame_fuselage":
            cx = self.CG_fuselage[0] + (
                aa / (math.pi / 2) * (self.CP_AERO_END - self.CP_AERO_START)
                + self.CP_AERO_START
            ) * self.CHORD_FUSELAGE

            return torch.stack((cx, z0, z0), 1)

        if frame in {"elev_left", "elev_right"}:
            cx = (- self.CG_tochord
                + aa / (math.pi / 2) * (self.CP_AERO_END - self.CP_AERO_START)
                + self.CP_AERO_START + 0.
            ) * self.CHORD_ELEVATOR
            sign = 1.0 if frame == "elev_left" else -1.0
            cy = torch.full_like(cx, sign * 0.25 * self.A_ELEVATOR / self.CHORD_ELEVATOR)
            cz = z0 + 0.02

            return torch.stack((cx, cy, cz), 1)

        if frame == "rudder":
            cx = (- self.CG_tochord
                + aa / (math.pi / 2) * (self.CP_AERO_END - self.CP_AERO_START)
                + self.CP_AERO_START + 0.
            ) * self.CHORD_RUDDER
            cz = self.CG_rudder[2] + torch.full((B,), 0.5 * self.A_RUDDER / self.CHORD_RUDDER, device=alpha.device)

            return torch.stack((cx, z0, cz), 1)

        return torch.zeros((B, 3), device=alpha.device)

    # ------------------------------------------------------------------ #
    #  Per-link solver                                                   #
    # ------------------------------------------------------------------ #
    def _link_aero(self, frame: str, link, throttle, entity=None) -> Dict[str, Any]:
        v = torch.as_tensor(link.get_vel(), device=self.device, dtype=torch.float32)
        q = torch.as_tensor(link.get_quat(), device=self.device, dtype=torch.float32)
        B = v.size(0)

        # wind in body frame
        R = self.quat_to_R_batch(q)
        vL = (R.transpose(1, 2) @ (-v)[:, :, None]).squeeze(2)
        V = vL.norm(dim=1)
        alpha = torch.atan2(vL[:, 2], vL[:, 0])
        beta = torch.where(V > 1e-6, torch.asin(vL[:, 1] / V), torch.zeros_like(V))

        Ry = self.axis_angle_to_R_batch([0, 1, 0], -alpha)
        Rz = self.axis_angle_to_R_batch([0, 0, 1], beta)
        Rw = Ry @ Rz  # wind → body

        thr = throttle if torch.is_tensor(throttle) else torch.full((B,), float(throttle), device=self.device)

        # ── wings ───────────────────────────────────────────────────────
        if frame in ("aero_frame_left_wing", "aero_frame_right_wing"):
            AR = self.A_WING / self.CHORD_WING**2
            cl, cd = self.aero_coefficients("wing", AR, alpha)
            L = 0.5 * self.AIR_DENSITY * self.A_WING * cl * V**2
            D = 0.5 * self.AIR_DENSITY * self.A_WING * cd * V**2
            cp = self.cp_offset(frame, alpha)

            F_wind = torch.stack((D, torch.zeros_like(D), L), 1)
            Fb = (Rw @ F_wind[:, :, None]).squeeze(2)
            Mb = torch.cross(cp, Fb, 1)
            Fw = (R @ Fb[:, :, None]).squeeze(2)
            Mw = (R @ Mb[:, :, None]).squeeze(2)
            name = "left_wing" if "left" in frame else "right_wing"
            return {name: dict(lift=L, drag=D, cp_offset=cp, alpha=alpha, beta=beta, wrench=(Fw, Mw), wrench_body=(Fb, Mb))}

        # ── fuselage ────────────────────────────────────────────────────
        if frame == "aero_frame_fuselage":
            ARf = self.A_FUSELAGE / self.CHORD_FUSELAGE**2
            cl, cd = self.aero_coefficients(frame, ARf, alpha)
            L = 0.5 * self.AIR_DENSITY * self.A_FUSELAGE * cl * V**2
            D = 0.5 * self.AIR_DENSITY * self.A_FUSELAGE * cd * V**2
            cp = self.cp_offset(frame, alpha)

            F_wind = torch.stack((D, torch.zeros_like(D), L), 1)
            Fb = (Rw @ F_wind[:, :, None]).squeeze(2)
            Mb = torch.cross(cp, Fb, 1)
            Fw = (R @ Fb[:, :, None]).squeeze(2)
            Mw = (R @ Mb[:, :, None]).squeeze(2)
            return {
                "fuselage": dict(lift=L, drag=D, cp_offset=cp, alpha=alpha, beta=beta, wrench=(Fw, Mw), wrench_body=(Fb, Mb))
            }

        # ── elevator (tail) ─────────────────────────────────────────────
        if frame == "aero_frame_elevator":
            # Down-wash from wings
            ARw = self.A_WING / self.CHORD_WING**2
            # left wing
            cl_wl, alpha_wl = self._wing_downwash(entity, "aero_frame_left_wing")
            eps_l = self.K_EPS_TAIL * cl_wl / (math.pi * ARw)
            # right wing
            cl_wr, alpha_wr = self._wing_downwash(entity, "aero_frame_right_wing")
            eps_r = self.K_EPS_TAIL * cl_wr / (math.pi * ARw)

            alpha_eff_l = alpha - eps_l
            alpha_eff_r = alpha - eps_r

            half_S = self.A_ELEVATOR / 2.0
            ar_elevator = self.A_ELEVATOR / self.CHORD_ELEVATOR**2
            # left half-elevator
            cl_el, cd_el = self.aero_coefficients("elevator", ar_elevator, alpha_eff_l)
            L_el = 0.5 * self.AIR_DENSITY * half_S * cl_el * V**2
            D_el = 0.5 * self.AIR_DENSITY * half_S * cd_el * V**2
            # right half-elevator
            cl_er, cd_er = self.aero_coefficients("elevator", ar_elevator, alpha_eff_r)
            L_er = 0.5 * self.AIR_DENSITY * half_S * cl_er * V**2
            D_er = 0.5 * self.AIR_DENSITY * half_S * cd_er * V**2

            # forces wind-frame
            FwL = torch.stack((D_el, torch.zeros_like(D_el), L_el), 1)
            FwR = torch.stack((D_er, torch.zeros_like(D_er), L_er), 1)

            # wind→body rot for each half
            RwL = self.axis_angle_to_R_batch([0, 1, 0], -alpha_eff_l) @ Rz
            RwR = self.axis_angle_to_R_batch([0, 1, 0], -alpha_eff_r) @ Rz

            cp_L = self.cp_offset("elev_left", alpha_eff_l)
            cp_R = self.cp_offset("elev_right", alpha_eff_r)

            FbL = (RwL @ FwL[:, :, None]).squeeze(2)
            FbR = (RwR @ FwR[:, :, None]).squeeze(2)
            MbL = torch.cross(cp_L, FbL, 1)
            MbR = torch.cross(cp_R, FbR, 1)

            FwL = (R @ FbL[:, :, None]).squeeze(2)
            FwR = (R @ FbR[:, :, None]).squeeze(2)
            MwL = (R @ MbL[:, :, None]).squeeze(2)
            MwR = (R @ MbR[:, :, None]).squeeze(2)

            return {
                "elevator_left": dict(
                    lift=L_el, drag=D_el, cp_offset=cp_L, alpha=alpha_eff_l, beta=beta, wrench=(FwL, MwL), wrench_body=(FbL, MbL)
                ),
                "elevator_right": dict(
                    lift=L_er, drag=D_er, cp_offset=cp_R, alpha=alpha_eff_r, beta=beta, wrench=(FwR, MwR), wrench_body=(FbR, MbR)
                ),
            }

        # ── rudder ───────────────────────────────────────────────────────
        if frame == "aero_frame_rudder":
            ar_rudder = self.A_RUDDER / self.CHORD_RUDDER**2
            cl, cd = self.aero_coefficients("rudder", ar_rudder, beta)
            D = 0.5 * self.AIR_DENSITY * self.A_RUDDER * cd * V**2
            Y = 0.5 * self.AIR_DENSITY * self.A_RUDDER * cl * V**2
            cp = self.cp_offset("rudder", beta)

            F_wind = torch.stack((D, Y, torch.zeros_like(D)), 1)
            Fb = (Rw @ F_wind[:, :, None]).squeeze(2)
            Mb = torch.cross(cp, Fb, 1)
            Fw = (R @ Fb[:, :, None]).squeeze(2)
            Mw = (R @ Mb[:, :, None]).squeeze(2)
            return {
                "rudder": dict(
                    drag=D, side_force=Y, cp_offset=cp, beta=beta, alpha=alpha, wrench=(Fw, Mw), wrench_body=(Fb, Mb)
                )
            }

        # ── propeller ────────────────────────────────────────────────────
        if frame == "prop_frame_fuselage_0":
            dirv = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32)
            Fb = dirv * (thr * self.MAX_THRUST)[:, None]
            Mb = dirv * (thr * self.MAX_THRUST * self.RATIO_PROP)[:, None]
            Fw = (R @ Fb[:, :, None]).squeeze(2)
            Mw = (R @ Mb[:, :, None]).squeeze(2)
            return {"prop_frame_fuselage_0": {"wrench": (Fw, Mw), "wrench_body": (Fb, Mb), "cp_offset": torch.zeros_like(Fb)}}

        return {}

    # helper used only by the elevator to get down-wash
    def _wing_downwash(self, entity, link_name: str):
        link = entity.get_link(link_name)
        v = torch.as_tensor(link.get_vel(), device=self.device, dtype=torch.float32)
        q = torch.as_tensor(link.get_quat(), device=self.device, dtype=torch.float32)
        R = self.quat_to_R_batch(q)
        vL = (R.transpose(1, 2) @ (-v)[:, :, None]).squeeze(2)
        alpha_w = torch.atan2(vL[:, 2], vL[:, 0])
        ARw = self.A_WING / self.CHORD_WING**2
        cl_w, _ = self.aero_coefficients("wing", ARw, alpha_w)
        return cl_w, alpha_w

    # ------------------------------------------------------------------ #
    #  Public entry-point (multi-link)                                   #
    # ------------------------------------------------------------------ #
    def compute_aero_details(self, entity, throttle) -> Dict[str, Any]:
        frames = {
            "aero_frame_fuselage": "aero_frame_fuselage",
            "aero_frame_left_wing": "aero_frame_left_wing",
            "aero_frame_right_wing": "aero_frame_right_wing",
            "aero_frame_elevator": "aero_frame_elevator",
            "aero_frame_rudder": "aero_frame_rudder",
            "prop_frame_fuselage_0": "prop_frame_fuselage_0",
        }

        out: Dict[str, Any] = {}
        for frame, name in frames.items():
            out.update(self._link_aero(frame, entity.get_link(name), throttle, entity))

        # ---------- global CG -------------------------------------------
        links = entity.links
        NL = len(links)

        pos_all = torch.stack([torch.as_tensor(L.get_pos(), device=self.device, dtype=torch.float32) for L in links])
        quat_all = torch.stack([torch.as_tensor(L.get_quat(), device=self.device, dtype=torch.float32) for L in links])

        R_links = self.quat_to_R_batch(quat_all.view(-1, 4)).view(NL, -1, 3, 3)
        com_loc = torch.stack([torch.tensor(L.inertial_pos, device=self.device, dtype=torch.float32) for L in links])
        masses = torch.tensor([L.inertial_mass for L in links], device=self.device, dtype=torch.float32)

        com_w = pos_all + torch.einsum("nbij,nj->nbi", R_links, com_loc)
        CG = (com_w * masses[:, None, None]).sum(0) / masses.sum()
        # ---------- total aero force / moment ---------------------------
        F_tot = torch.zeros_like(CG)
        M_tot = torch.zeros_like(CG)
        key2frame = dict(
            fuselage="aero_frame_fuselage",
            left_wing="aero_frame_left_wing",
            right_wing="aero_frame_right_wing",
            elevator_left="aero_frame_elevator",
            elevator_right="aero_frame_elevator",
            rudder="aero_frame_rudder",
            prop_frame_fuselage_0="prop_frame_fuselage_0",
        )

        for key, data in out.items():
            if key.startswith("prop") or "cp_offset" not in data:
                continue

            F_w, M_w = data["wrench"]
            has_cp = "cp_offset" in data
            cp_loc = data["cp_offset"] if has_cp else torch.zeros_like(F_w)

            link = entity.get_link(key2frame[key])
            p = torch.as_tensor(link.get_pos(), device=self.device)
            Rl = self.quat_to_R_batch(torch.as_tensor(link.get_quat(), device=self.device))
            cp_w = p + (Rl @ cp_loc[:, :, None]).squeeze(2)

            r = cp_w - CG
            F_tot += F_w
            M_tot += torch.cross(r, F_w, 1)
            if not has_cp:                                    # ← propeller
                M_tot += M_w  
        # ---------- global aerodynamic centre ---------------------------
        eps = 1e-6

        F_norm2 = (F_tot.norm(dim=1, keepdim=True).clamp_min(eps)) ** 2
        r_AC = torch.cross(F_tot, M_tot, dim=1) / F_norm2
        AC = CG + r_AC
        #  ----- dopo aver calcolato AC -----
        M_res = M_tot - torch.cross(AC - CG, F_tot, dim=1)
        out["M_residual"] = M_res


        out["CG"] = CG
        out["AC"] = AC
        return out
