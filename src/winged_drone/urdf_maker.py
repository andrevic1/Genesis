from __future__ import annotations
"""
Parametric URDF generator for fixed‑wing UAVs – *prop‑aware*, LE‑aware
=====================================================================

Changes vs. previous (2025‑05‑XX)
---------------------------------
1. **`hinge_le_ratio`** – new parameter (default = 0.14).
   It is the distance (in *fraction of chord*) between the wing–fuselage
   hinge and the leading edge (LE).  It is applied to **all lifting
   surfaces** (wing panels, elevator halves, rudder).

   *   The *fixed‑joint* that attaches the surface now shifts **− `hinge_le_ratio·c`**
       along the local **x** so that the child–link origin coincides with
       the LE.
   *   CG and collision offsets are then expressed from that LE:
       – CG = +0.31 c (+ forward for wing, – for elevator & rudder)
       – coll = +0.25 c (idem sign)

2. **Prop‑wash span** still derives from `prop_radius` (introduced in the
   previous revision).
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Union
from dataclasses import fields
import math
# ─────────────────────────────────────────────────────────────────────────────
#  User parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeometryParams:
    wing_span: float
    wing_chord: float
    wing_attach_x: float
    fus_cg_x: float
    fus_length: float
    elevator_span: float
    elevator_chord: float
    rudder_span: float
    rudder_chord: float
    density_scale: float = 1.0
    dihedral_deg: float = 0.0   # ±30°
    prop_radius: float = 0.10    # prop disc radius (m)
    hinge_le_ratio: float = 0.14 # hinge → LE distance (fraction of c)
    sweep_multi: float = 1.0    # es. aileron
    twist_multi: float = 1.0    # es. elevator/rudder
    cl_alpha_2d:    float = 2 * math.pi  # 2π rad⁻¹ tipico profilo sottile
    alpha0_2d:      float = 0.0 * math.pi / 180.0   # rad
# ─────────────────────────────────────────────────────────────────────────────
#  URDF maker
# ─────────────────────────────────────────────────────────────────────────────

class UrdfMaker:
    """Build a *drone_ea*‑compatible URDF from parametric geometry."""

    # reference geometry (constant – from original file) --------------------
    _REF = {
        "tw":   0.017962225,
        "tc":   0.0135567616,
        "tfus": 0.10000000000000005,
        "hfus": 0.10476803794561983,
        "fus_len": 0.7273891148,
        "wing_chord": 0.199988144,
        "wing_span_ref": 0.70,
        "elev_chord": 0.139941293,
        "elev_span": 0.1800018297,
        "rudd_chord": 0.119323096638,
        "rudd_span": 0.16357341166,
    }

    # reference mesh scales -------------------------------------------------
    _REF_SCALE = {
        "fuselage": (0.00097, 0.001, 0.001),
        "wing":     (0.2, 0.7, 0.2),
        "elevator": (0.0007, 0.0012, 0.001),
        "rudder":   (0.0008, 0.001 , 0.001),
    }

    # densities (kg m⁻³) ----------------------------------------------------
    _RHO_FUS_STRUCT  = 20.0
    _FUS_FIXED_MASS  = 0.300           # kg, parte fissa
    _BATTERY_SIZE    = (0.10, 0.05, 0.05)  # m (10×10×5 cm)
    _RHO_WING = 20
    _RHO_RUDD = 20
    _RHO_PROP = 200.0
    _RHO_ELEV = 20

    # ratios from LE --------------------------------------------------------
    _CG_RATIO  = 0.31   # CG @ 31 % c (from LE)
    _COLL_RATIO = 0.25  # box @ 25 % c

    # inertia fudge factor --------------------------------------------------
    _I_FUDGE = 0.6
    _SHRINK = 0.25
    # misc legacy const -----------------------------------------------------
    _MASS_INTER     = 0.1
    _ROOT_Y_OFFSET  = 0.05   # hinge y‑offset (m)

    _RPY_FUSE_COLL  = "-1.5898372930676048 6.123233995736766e-17 -1.5707963267948968"
    _RPY_WING_COLL  = "1.570796327 -1.558922022 0"
    _RPY_ELEV_COLL  = "1.5763749568 -1.5318707859 3.1359944429"
    _RPY_RUDD_COLL  = "0.10626955814138643 0.008780081131392076 1.5717328981686811"

    _REV_LIMIT = {"lower": "-0.35", "upper": "0.35",
                  "effort": "1.5", "velocity": "3.665191429"}
    _REV_DYN   = {"damping": "0.2", "friction": "0.05"}
    _LE_REF = 0.14  # default LE shift ratio (fraction of c)

    # ────────────────────────────────────────────────────────────────────
    def __init__(self, prm: Union[GeometryParams, Sequence[float]], *, out_dir: Union[str, Path] = "urdf_generated") -> None:
        if not isinstance(prm, GeometryParams):
            prm = GeometryParams(*prm)  # type: ignore[arg-type]
        self.p = prm
        self.S = prm.density_scale
        self.LE = prm.hinge_le_ratio
        self.out = Path(out_dir)
        self.dihedral = max(-30.0, min(30.0, prm.dihedral_deg)) * math.pi / 180.0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _indent(elem: ET.Element, lvl: int = 0):
        pad = "\n" + lvl * "  "
        if len(elem):
            if not (elem.text or '').strip():
                elem.text = pad + "  "
            for c in elem:
                UrdfMaker._indent(c, lvl + 1)
            if not (elem.tail or '').strip():
                elem.tail = pad
        elif lvl and not (elem.tail or '').strip():
            elem.tail = pad

    def _origin(self, el: ET.Element, xyz, rpy="0 0 0") -> None:
        ET.SubElement(el, "origin", xyz=" ".join(f"{v:.12g}" for v in xyz), rpy=rpy)

    def _add_inertial(self, link: ET.Element, xyz, m, I):
        inn = ET.SubElement(link, "inertial")
        self._origin(inn, xyz)
        ET.SubElement(inn, "mass", value=f"{m:.6g}")
        ET.SubElement(inn, "inertia", ixx=f"{I[0]:.6g}", ixy="0", ixz="0",
                      iyy=f"{I[1]:.6g}", iyz="0", izz=f"{I[2]:.6g}")

    def _I_box(self, m, sx, sy, sz, *, fudge=None):
        k = self._I_FUDGE if fudge is None else fudge
        return (k * m * (sy**2 + sz**2) / 12.0,
                k * m * (sx**2 + sz**2) / 12.0,
                k * m * (sx**2 + sy**2) / 12.0)

    def _gene_names(self) -> list[str]:
        return [f.name for f in fields(GeometryParams)]

    def _gene_values(self) -> list[float]:
        p = self.p
        return [getattr(p, n) for n in self._gene_names()]

    def _genes_as_string(self) -> str:
        return "[" + ", ".join(f"{v:g}" for v in self._gene_values()) + "]"
    # mesh scale --------------------------------------------------------
    def _scale(self, what: str, dims: Tuple[float, float, float]) -> str:
        rs = self._REF_SCALE[what]

        # ── fusoliera: mapping dedicato ───────────────────────────────
        if what == "fuselage":
            tfus, hfus, fus_len = dims        # ordine (spess., altezza, lunghezza)
            tfus_ref, hfus_ref, fus_len_ref = (
                self._REF["tfus"], self._REF["hfus"], self._REF["fus_len"]
            )

            sx = rs[0] * fus_len / fus_len_ref                     # 1) lunghezza
            sy = rs[1] * hfus / hfus_ref          
            sz = rs[2] * tfus / tfus_ref                          # 3) spessore
            return f"{sx:.6g} {sy:.6g} {sz:.6g}"

        # ── tutti gli altri pezzi rimangono generici ─────────────────
        rd = {
            "wing":     (self._REF["tw"],  self._REF["wing_chord"], self._REF["wing_span_ref"]),
            "elevator": (self._REF["tc"],  self._REF["elev_chord"], self._REF["elev_span"]),
            "rudder":   (self._REF["tc"],  self._REF["rudd_chord"], self._REF["rudd_span"]),
        }[what]
        sx, sy, sz = (rs[i] * dims[i] / rd[i] for i in range(3))
        return f"{sx:.6g} {sy:.6g} {sz:.6g}"


    # ------------------------------------------------------------------
    #  materials & root
    # ------------------------------------------------------------------
    def _materials(self, robot):
        for n, rgba in {"grey": (0.7, 0.7, 0.7, 1), "red": (0.7, 0, 0, 1), "black": (0.2, 0.2, 0.2, 1)}.items():
            m = ET.SubElement(robot, "material", name=n)
            ET.SubElement(m, "color", rgba=" ".join(map(str, rgba)))

    def _root_link(self, robot):
        self._add_inertial(ET.SubElement(robot, "link", name="root_link"), (0, 0, 0), 0, (0, 0, 0))

    # ------------------------------------------------------------------
    #  fuselage (unchanged except bug‑fix on string join)
    # ------------------------------------------------------------------
    def _fuselage(self, robot):

        p   = self.p
        box = (self._REF["tfus"], self._REF["hfus"], p.fus_length)
        vol = math.prod(box)

        # masse
        m_batt   = self._FUS_FIXED_MASS
        m_shell  = vol * self._RHO_FUS_STRUCT * self.S
        m_tot    = m_batt + m_shell

        # inerzie (riferite al CG del link)
        I_shell = self._I_box(m_shell, *box)                 # con fudge
        bx, by, bz = self._BATTERY_SIZE
        I_batt  = self._I_box(m_batt, bx, by, bz, fudge=1.0) # senza fudge
        I_tot   = tuple(a + b for a, b in zip(I_shell, I_batt))

        ln = ET.SubElement(robot, "link", name="fuselage")
        self._add_inertial(ln, (p.fus_cg_x, 0, -0.04), m_tot, I_tot)

        box_coll = (box[0] * self._SHRINK,                # spessore invariato
                    box[1] * self._SHRINK,  # *¼ sull’asse Y
                    box[2] * self._SHRINK)

        coll = ET.SubElement(ln, "collision", name="fuselage_collision_0")
        self._origin(coll, (-0.5 * p.fus_length, 0, 0.0024556534860724526), self._RPY_FUSE_COLL)
        ET.SubElement(ET.SubElement(coll, "geometry"), "box",
                      size=" ".join(f"{v:.12g}" for v in box_coll))

        vis = ET.SubElement(ln, "visual", name="fuselage_visual_0")
        self._origin(vis, (0, 0, 0))
        mesh = ET.SubElement(ET.SubElement(vis, "geometry"), "mesh", filename="package://meshes/fuselage_only.stl")
        mesh.set("scale", self._scale("fuselage", box))
        ET.SubElement(vis, "material", name="grey")

        j0 = ET.SubElement(robot, "joint", name="fixed_joint_base_fuselage", type="fixed")
        ET.SubElement(j0, "parent", link="root_link")
        ET.SubElement(j0, "child",  link="fuselage")
        self._origin(j0, (0, 0, 0), "3.141592653589793 0 0")

        jaf = ET.SubElement(robot, "joint", name="fixed_joint_aero_frame_fuselage", type="fixed")
        ET.SubElement(jaf, "parent", link="fuselage")
        ET.SubElement(jaf, "child",  link="aero_frame_fuselage")
        self._origin(jaf, (p.fus_cg_x, 0, -0.03), "0 3.141592653589793 0")
        self._add_inertial(ET.SubElement(robot, "link", name="aero_frame_fuselage"), (0, 0, 0), 0, (0, 0, 0))

    # ------------------------------------------------------------------
    #  prop frame (radius now = prop_radius)
    # ------------------------------------------------------------------
    def _prop_frame(self, robot):
        p = self.p
        j = ET.SubElement(robot, "joint", name="fixed_joint_prop_frame_fuselage_0", type="fixed")
        ET.SubElement(j, "parent", link="fuselage")
        ET.SubElement(j, "child",  link="prop_frame_fuselage_0")
        self._origin(j, (0, 0, 0), "3.141592653589793 -1.5707963267948966 0")

        r, h = p.prop_radius, 0.005
        m = math.pi * r*r * h * self._RHO_PROP * self.S
        I = self._I_box(m, 2*r, 2*r, h)

        ln = ET.SubElement(robot, "link", name="prop_frame_fuselage_0")
        self._add_inertial(ln, (0, 0, 0), m, I)

        vis = ET.SubElement(ln, "visual", name="prop_frame_fuselage_0_visual_0")
        self._origin(vis, (0, 0, 0))
        ET.SubElement(ET.SubElement(vis, "geometry"), "cylinder", radius=f"{r}", length=f"{h}")
        ET.SubElement(vis, "material", name="black")

        coll = ET.SubElement(ln, "collision", name="prop_frame_collision")
        self._origin(coll, (0, 0, 0))
        ET.SubElement(ET.SubElement(coll, "geometry"), "box", size="0.2 0.2 0.02")

    # ------------------------------------------------------------------
    #  wings (prop span from radius, LE shift from hinge_le_ratio)
    # ------------------------------------------------------------------
    def _wings(self, robot):
        p = self.p
        t = self._REF["tw"]

        # span inside prop disc (per side)
        sp_prop = max(0.0, min(p.wing_span, max(0.0, p.prop_radius - self._ROOT_Y_OFFSET)))
        sp_free = p.wing_span - sp_prop
        rho = self._RHO_WING * self.S

        # ------------------------------------------------------------------
        def svc(name: str):
            ln = ET.SubElement(robot, "link", name=name)
            self._add_inertial(ln, (0, 0, 0), self._MASS_INTER,
                               self._I_box(self._MASS_INTER, t, t, t))

        # ------------------------------------------------------------------
        def panel(side: str, lbl: str, span: float, y_off: float):
            name = f"{side}_wing_{lbl}"
            c = p.wing_chord
            box = (t, c, span)
            m   = math.prod(box) * rho
            I   = self._I_box(m, *box)

            j = ET.SubElement(robot, "joint", name=f"fixed_joint_{name}", type="fixed")
            ET.SubElement(j, "parent", link=f"{side}_wing")
            ET.SubElement(j, "child",  link=name)
            self._origin(j, (0, y_off, 0))

            ln = ET.SubElement(robot, "link", name=name)
            self._add_inertial(ln, ((self._CG_RATIO-self.LE) * c, 0, 0), m, I)

            coll = ET.SubElement(ln, "collision", name=f"{name}_collision")
            self._origin(coll, (self._COLL_RATIO * c, 0, -0.000421042), self._RPY_WING_COLL)
            ET.SubElement(ET.SubElement(coll, "geometry"), "box",
                          size=" ".join(f"{v:.12g}" for v in box))

            ja = ET.SubElement(robot, "joint", name=f"fixed_joint_aero_frame_{name}", type="fixed")
            ET.SubElement(ja, "parent", link=name)
            ET.SubElement(ja, "child",  link=f"aero_frame_{name}")
            self._origin(ja, ((self._CG_RATIO-self.LE) * c, 0, 0), "3.141592654 0 0")
            self._add_inertial(ET.SubElement(robot, "link", name=f"aero_frame_{name}"), (0, 0, 0), 0, (0, 0, 0))

        # ------------------------------------------------------------------
        def wing_side(side: str, sgn: int):
            limit_sweep  = 0.7 / max(self.p.sweep_multi, 0.5)   # rad
            limit_twist  = 0.7 / max(self.p.twist_multi, 0.5)   # rad
            effort_limit_sweep = 1.2 * max(self.p.sweep_multi, 0.5)
            effort_limit_twist = 0.6 * max(self.p.twist_multi, 0.5)
            c = p.wing_chord
            fj = ET.SubElement(robot, "joint", name=f"fixed_joint_{side}_wing", type="fixed")
            ET.SubElement(fj, "parent", link="fuselage")
            ET.SubElement(fj, "child",  link=f"fuselage_{side}_0")
            self._origin(fj, (p.wing_attach_x, self._ROOT_Y_OFFSET*sgn, -0.03),
                         f"{self.dihedral*sgn:.12g} 0 3.141592653589793")

            svc(f"fuselage_{side}_0")
            svc(f"fuselage_{side}_1")

            # 1) SWEEP tra fus0 → fus1  (ora è joint_0)
            js = ET.SubElement(robot, "joint", name=f"joint_0_sweep_{side}_wing", type="revolute")
            ET.SubElement(js, "parent", link=f"fuselage_{side}_0")
            ET.SubElement(js, "child",  link=f"fuselage_{side}_1")
            # origin del twist originariamente era (0,0,0): lo usiamo qui per la sweep
            self._origin(js, (0, 0, 0))  
            ET.SubElement(js, "axis", xyz="0 0 1")
            lim_sweep = {
                "lower": f"{-limit_sweep:.6g}",
                "upper": f"{ limit_sweep:.6g}",
                "effort": f"{effort_limit_sweep:.6g}",
                "velocity": self._REV_LIMIT["velocity"]
            }
            ET.SubElement(js, "limit", **lim_sweep)
            ET.SubElement(js, "dynamics", **self._REV_DYN)

            # 2) TWIST tra fus1 → wing  (ora è joint_1)
            jt = ET.SubElement(robot, "joint", name=f"joint_1_twist_{side}_wing", type="revolute")
            ET.SubElement(jt, "parent", link=f"fuselage_{side}_1")
            ET.SubElement(jt, "child",  link=f"{side}_wing")
            # origin del sweep originariamente era (0,-ROOT_Y_OFFSET,0): lo usiamo qui
            self._origin(jt, (0, -self._ROOT_Y_OFFSET*sgn, 0))
            ET.SubElement(jt, "axis", xyz="0 1 0")
            lim_twist = {
                "lower": f"{-limit_twist:.6g}",
                "upper": f"{ limit_twist:.6g}",
                "effort": f"{effort_limit_twist:.6g}",
                "velocity": self._REV_LIMIT["velocity"]
            }
            ET.SubElement(jt, "limit", **lim_twist)
            ET.SubElement(jt, "dynamics", **self._REV_DYN)

            root = ET.SubElement(robot, "link", name=f"{side}_wing")
            self._add_inertial(root, (0, 0, 0), 0, (0, 0, 0))
            vr = ET.SubElement(root, "visual", name=f"{side}_wing_visual")
            rpy = "3.141592654 0 0" if side == "left" else "0 0 0"
            vis_shift_x = (self.LE - self._LE_REF) * c
            self._origin(vr, (vis_shift_x, 0, 0), rpy)
            mr = ET.SubElement(ET.SubElement(vr, "geometry"), "mesh", filename="package://meshes/wing0009.stl")
            mr.set("scale", self._scale("wing", (t, c, sp_prop + sp_free)))
            ET.SubElement(vr, "material", name="red")


            if sp_prop > 0:
                panel(side, "prop", sp_prop, -sgn*sp_prop/2)
            if sp_free > 0:
                panel(side, "free", sp_free, -sgn*(sp_prop + sp_free/2))

        wing_side("right", +1)
        wing_side("left",  -1)

    # ------------------------------------------------------------------
    #  elevator & rudder (LE shift + new ratios)
    # ------------------------------------------------------------------
    def _elevator(self, robot):
        p = self.p
        box_full = (self._REF["tc"], p.elevator_chord, p.elevator_span)
        box_half = (box_full[0], box_full[1], p.elevator_span/2)
        rho = self._RHO_ELEV * self.S
        m_tot = math.prod(box_full) * rho
        m_half  = math.prod(box_half) * rho          # massa di UNA metà
        I_half  = self._I_box(m_half, *box_half)
        span_half = p.elevator_span/2
        c = p.elevator_chord

        # hinge joint --------------------------------------------------
        jh = ET.SubElement(robot, "joint", name="elevator_pitch_joint", type="revolute")
        ET.SubElement(jh, "parent", link="fuselage")
        ET.SubElement(jh, "child",  link="elevator_hinge")
        self._origin(jh, (- p.fus_length - 0.03, 0, 0))
        ET.SubElement(jh, "axis", xyz="0 1 0")
        ET.SubElement(jh, "limit", lower="-0.35", upper="0.35", effort="1.0", velocity="3.665191429")
        ET.SubElement(jh, "dynamics", damping="0.2", friction="0.05")

        ln_hinge = ET.SubElement(robot, "link", name="elevator_hinge")
        self._add_inertial(ln_hinge, (0, 0, 0), 0, (0, 0, 0))

        # visual complessivo dell’elevatore
        vis_h = ET.SubElement(ln_hinge, "visual")
        self._origin(vis_h, (0, 0, 0))
        mesh_h = ET.SubElement(ET.SubElement(vis_h, "geometry"), "mesh",
                               filename="package://meshes/elevator.stl")
        mesh_h.set("scale", self._scale(
            "elevator", (self._REF["tc"], p.elevator_chord, p.elevator_span)))
        ET.SubElement(vis_h, "material", name="grey")

        for sgn, side in ((+1, "left"), (-1, "right")):
            fj = ET.SubElement(robot, "joint", name=f"fixed_joint_elevator_{side}", type="fixed")
            ET.SubElement(fj, "parent", link="elevator_hinge")
            ET.SubElement(fj, "child",  link=f"elevator_{side}")
            self._origin(fj, (0, span_half*sgn, 0))

            ln = ET.SubElement(robot, "link", name=f"elevator_{side}")
            self._add_inertial(ln, (self._CG_RATIO * c, 0, 0), m_half, I_half)

            coll = ET.SubElement(ln, "collision")
            self._origin(coll, (-self._COLL_RATIO * c, 0, -0.0226291863), self._RPY_ELEV_COLL)
            ET.SubElement(ET.SubElement(coll, "geometry"), "box",
                          size=" ".join(f"{v:.12g}" for v in box_half))

            ja = ET.SubElement(robot, "joint", name=f"fixed_joint_aero_frame_elevator_{side}", type="fixed")
            ET.SubElement(ja, "parent", link=f"elevator_{side}")
            ET.SubElement(ja, "child",  link=f"aero_frame_elevator_{side}")
            self._origin(ja, (-self._CG_RATIO * c, 0, 0), "0 3.141592654 0")
            self._add_inertial(ET.SubElement(robot, "link", name=f"aero_frame_elevator_{side}"), (0, 0, 0), 0, (0, 0, 0))

    def _rudder(self, robot):
        p = self.p
        c = p.rudder_chord
        box = (self._REF["tc"], c, p.rudder_span)
        m   = math.prod(box) * self._RHO_RUDD * self.S
        I   = self._I_box(m, *box)

        jy = ET.SubElement(robot, "joint", name="rudder_yaw_joint", type="revolute")
        ET.SubElement(jy, "parent", link="elevator_hinge")
        ET.SubElement(jy, "child",  link="rudder")
        self._origin(jy, (0, 0, 0))
        ET.SubElement(jy, "axis", xyz="0 0 1")
        ET.SubElement(jy, "limit", lower="-0.35", upper="0.35", effort="1.0", velocity="3.665191429")
        ET.SubElement(jy, "dynamics", damping="0.2", friction="0.05")

        ln = ET.SubElement(robot, "link", name="rudder")
        self._add_inertial(ln, (-self._CG_RATIO * c, 0, -0.08), m, I)

        coll = ET.SubElement(ln, "collision", name="rudder_collision_0")
        self._origin(coll, (-self._COLL_RATIO * c, 0, -0.10499447081596952), self._RPY_RUDD_COLL)
        ET.SubElement(ET.SubElement(coll, "geometry"), "box",
                      size=" ".join(f"{v:.12g}" for v in box))

        vis = ET.SubElement(ln, "visual", name="rudder_visual_0")
        self._origin(vis, (0, 0, 0))
        mesh = ET.SubElement(ET.SubElement(vis, "geometry"), "mesh", filename="package://meshes/rudder.stl")
        mesh.set("scale", self._scale("rudder", box))
        ET.SubElement(vis, "material", name="red")

        ja = ET.SubElement(robot, "joint", name="fixed_joint_aero_frame_rudder", type="fixed")
        ET.SubElement(ja, "parent", link="rudder")
        ET.SubElement(ja, "child",  link="aero_frame_rudder")
        self._origin(ja, (-self._CG_RATIO * c, 0, -0.08), "0 3.141592653589793 0")
        self._add_inertial(ET.SubElement(robot, "link", name="aero_frame_rudder"), (0, 0, 0), 0, (0, 0, 0))

    def _add_metadata(self, robot: ET.Element) -> None:
        meta = ET.SubElement(robot, "metadata", type="genes")
        meta.set("order", " ".join(self._gene_names()))
        meta.set("value", " ".join(f"{v:g}" for v in self._gene_values()))
        # opzionale: copia “umana” in un commento
        robot.append(ET.Comment(f"genes {meta.get('order')} = {meta.get('value')}"))

    # ------------------------------------------------------------------
    def build_tree(self) -> ET.ElementTree:
        r = ET.Element("robot", name="drone_param")
        self._materials(r)
        self._root_link(r)
        self._fuselage(r)
        self._prop_frame(r)
        self._wings(r)
        self._elevator(r)
        self._rudder(r)
        self._add_metadata(r)
        return ET.ElementTree(r)

    def _params_as_string(self) -> str:
        p = self.p
        values = [p.wing_span, p.wing_chord, p.wing_attach_x, p.fus_cg_x,
                  p.fus_length, p.elevator_span, p.elevator_chord,
                  p.rudder_span, p.rudder_chord,
                  p.density_scale, p.dihedral_deg,
                  p.prop_radius, p.hinge_le_ratio, 
                  p.sweep_multi, p.twist_multi, 
                  p.cl_alpha_2d, p.alpha0_2d]
        return self._genes_as_string()

    # ────────────────────────────────────────────────────────────────────
    def create_urdf(self, filename: str | None = None) -> Path:
        """Scrive l’URDF.  
        Se *filename* è None (default) il nome diventa la stringa dei
        parametri, es.:  
        “[0.7, 0.199988, …, 0.14].urdf”
        """
        if filename is None:
            filename = f"{self._params_as_string()}.urdf"

        tree = self.build_tree()
        self._indent(tree.getroot())
        out = (self.out / filename)
        tree.write(out, encoding="utf-8", xml_declaration=True)
        return str(out)

# quick test -------------------------------------------------------------
if __name__ == "__main__":
    ref1 = [0.70, 0.2, -0.275, -0.275, 0.73, 0.18, 0.14, 0.16, 0.12, 1, 0, 0.10, 0.14, 2, 2.5, 2 * math.pi, -3* math.pi / 180.0]
    ref2 = [0.50, 0.15, -0.275, -0.275, 0.6, 0.14, 0.12, 0.12, 0.10, 1, 0, 0.10, 0.14, 2, 2.5, 2 * math.pi, -3* math.pi / 180.0]
    ref3 = [0.45, 0.1125, -0.15, -0.2, 0.4, 0.2, 0.1, 0.08, 0.16, 1, 10, 0.1, 0.18, 2, 2, 3, 0]
    ref4 = [0.5, 0.142857, -0.15, -0.2, 0.4, 0.2, 0.16, 0.16, 0.16, 1, 10, 0.1, 0.16, 2, 1.5, 3, 0]
    ref5 = [0.5, 0.142857, -0.15, -0.2, 0.4, 0.2, 0.16, 0.16, 0.16, 1, 0, 0.1, 0.16, 2.5, 2, 2, -1]
    print("URDF written to", UrdfMaker(ref5).create_urdf())