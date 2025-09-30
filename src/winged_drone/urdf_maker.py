from __future__ import annotations
"""
Parametric URDF generator for fixed-wing UAVs – genome-driven
=============================================================

This module builds a drone URDF from a *structured genome* of **15 parameters**:

 0) wing_span                     [m]
 1) wing_aspect_ratio             = span / chord
 2) fus_length                    [m]
 3) cg_x_ratio                    = fus_cg_x / fus_length
 4) attach_x_ratio                = wing_attach_x / fus_length
 5) elevator_span                 [m]
 6) elevator_aspect_ratio         = span / chord
 7) rudder_span                   [m]
 8) rudder_aspect_ratio           = span / chord
 9) dihedral_deg                  [deg]
10) hinge_le_ratio                [fraction of chord]
11) sweep_multiplier              [-]
12) twist_multiplier              [-]
13) cl_alpha_2d                   [1/rad]
14) alpha0_2d                     [deg] (converted to rad internally)

All *physical* dimensions required by the URDF (chords, x-positions) are
derived internally from the genome above.

Output URDF contains a single XML comment with the **raw genome** (no names).
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Sequence, Tuple, Union

# ─────────────────────────────────────────────────────────────────────────────
#  Physical parameter container (internal shape used by the generator)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeometryParams:
    """Fully resolved physical parameters (derived from the 15-gene genome)."""
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
    dihedral_deg: float = 0.0
    prop_radius: float = 0.10
    hinge_le_ratio: float = 0.14
    sweep_multi: float = 1.0
    twist_multi: float = 1.0
    cl_alpha_2d: float = 2
    alpha0_2d: float = 0.0   # [rad]

# ─────────────────────────────────────────────────────────────────────────────
#  URDF maker
# ─────────────────────────────────────────────────────────────────────────────

class UrdfMaker:
    """Build a *drone_ea*-compatible URDF from the 15-gene genome."""

    # Reference geometry (kept from legacy implementation)
    _REF = {
        "tw":   0.017962225,           # wing thickness box X
        "tc":   0.0135567616,          # control surface thickness box X
        "tfus": 0.10000000000000005,   # fuselage thickness (for scale)
        "hfus": 0.10476803794561983,   # fuselage height    (for scale)
        "fus_len": 0.7273891148,
        "wing_chord": 0.199988144,
        "wing_span_ref": 0.70,
        "elev_chord": 0.139941293,
        "elev_span": 0.1800018297,
        "rudd_chord": 0.119323096638,
        "rudd_span": 0.16357341166,
    }

    # Reference mesh scales
    _REF_SCALE = {
        "fuselage": (0.00097, 0.001, 0.001),
        "wing":     (0.2, 0.7, 0.2),
        "elevator": (0.0007, 0.0012, 0.001),
        "rudder":   (0.0008, 0.001 , 0.001),
    }

    # “Mass model” densities (kg/m³) and constants
    _RHO_FUS_STRUCT  = 20.0
    _FUS_FIXED_MASS  = 0.250              # e.g. battery, avionics
    _BATTERY_SIZE    = (0.10, 0.05, 0.05) # box for inertia (10×5×5 cm)
    _RHO_WING = 20
    _RHO_ELEV = 20
    _RHO_RUDD = 20
    _RHO_PROP = 500.0

    # Lever arms as fraction of chord (from LE reference frame)
    _CG_RATIO   = 0.31
    _COLL_RATIO = 0.25

    # Inertia fudge factor + collision shrink
    _I_FUDGE = 0.6
    _SHRINK  = 0.25

    # Root offsets and fixed RPYs (legacy values kept)
    _MASS_INTER     = 0.0
    _ROOT_Y_OFFSET  = 0.05

    _RPY_FUSE_COLL  = "-1.5898372930676048 6.123233995736766e-17 -1.5707963267948968"
    _RPY_WING_COLL  = "1.570796327 -1.558922022 0"
    _RPY_ELEV_COLL  = "1.5763749568 -1.5318707859 3.1359944429"
    _RPY_RUDD_COLL  = "0.10626955814138643 0.008780081131392076 1.5717328981686811"

    _REV_LIMIT = {"lower": "-0.35", "upper": "0.35", "effort": "1.5", "velocity": "3.665191429"}
    _REV_DYN   = {"damping": "0.2", "friction": "0.05"}
    _LE_REF    = 0.25  # historical LE shift used by legacy meshes

    # Servo masses (kg) e dimensioni box (m) per inerzia
    _SERVO_WING_SWEEP_MASS = 0.040   # servo sweep ala (per lato)
    _SERVO_WING_TWIST_MASS = 0.030   # servo twist ala (per lato)
    _SERVO_TAIL_ELEV_MASS  = 0.02   # servo elevatore
    _SERVO_TAIL_RUDD_MASS  = 0.02   # servo timone
    _SERVO_SIZE            = (0.03, 0.012, 0.03)  # (sx, sy, sz) ~ 30×12×30 mm

    # ────────────────────────────────────────────────────────────────────
    # Construction
    # ────────────────────────────────────────────────────────────────────
    def __init__(self, genome_or_params: Union[GeometryParams, Sequence[float]],
                 *, out_dir: Union[str, Path] = "urdf_generated") -> None:
        """
        Accept either:
          • a 15-value genome (Sequence[float]) as specified at top, or
          • a pre-resolved GeometryParams instance.
        """
        self._raw_genome: list[float] | None = None

        if isinstance(genome_or_params, GeometryParams):
            prm = genome_or_params
        else:
            seq = list(genome_or_params)
            if len(seq) != 15:
                raise ValueError("Expected a 15-value genome sequence.")
            # Unpack genome
            (wing_span, wing_AR, fus_length, cg_ratio, attach_ratio,
             elev_span, elev_AR, rudd_span, rudd_AR,
             dihedral_deg, hinge_le_ratio, sweep_multi, twist_multi,
             cl_alpha_2d, alpha0_2d_deg) = seq

            # Derive physical quantities
            wing_chord     = wing_span / max(wing_AR, 1e-6)
            elevator_chord = elev_span / max(elev_AR, 1e-6)
            rudder_chord   = rudd_span / max(rudd_AR, 1e-6)
            fus_cg_x       = cg_ratio    * fus_length
            wing_attach_x  = attach_ratio * fus_length
            density_scale  = 1.0
            prop_radius    = 0.10
            alpha0_2d_rad  = alpha0_2d_deg * math.pi / 180.0
            cl_alpha_2d = cl_alpha_2d * math.pi

            prm = GeometryParams(
                wing_span=wing_span,
                wing_chord=wing_chord,
                wing_attach_x=wing_attach_x,
                fus_cg_x=fus_cg_x,
                fus_length=fus_length,
                elevator_span=elev_span,
                elevator_chord=elevator_chord,
                rudder_span=rudd_span,
                rudder_chord=rudder_chord,
                density_scale=density_scale,
                dihedral_deg=dihedral_deg,
                prop_radius=prop_radius,
                hinge_le_ratio=hinge_le_ratio,
                sweep_multi=sweep_multi,
                twist_multi=twist_multi,
                cl_alpha_2d=cl_alpha_2d,
                alpha0_2d=alpha0_2d_rad
            )
            self._raw_genome = seq

        self.p = prm
        self.out = Path(out_dir)
        self.S = prm.density_scale
        self.LE = prm.hinge_le_ratio
        self.dihedral = max(-30.0, min(30.0, prm.dihedral_deg)) * math.pi / 180.0

    # ────────────────────────────────────────────────────────────────────
    # Utilities
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _indent(elem: ET.Element, lvl: int = 0) -> None:
        """Pretty-print XML indentation."""
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

    def _origin(self, el: ET.Element, xyz, rpy: str = "0 0 0") -> None:
        ET.SubElement(el, "origin", xyz=" ".join(f"{v:.12g}" for v in xyz), rpy=rpy)

    def _add_inertial(self, link: ET.Element, xyz, m, I) -> None:
        inn = ET.SubElement(link, "inertial")
        self._origin(inn, xyz)
        ET.SubElement(inn, "mass", value=f"{m:.6g}")
        ET.SubElement(inn, "inertia", ixx=f"{I[0]:.6g}", ixy="0", ixz="0",
                      iyy=f"{I[1]:.6g}", iyz="0", izz=f"{I[2]:.6g}")

    def _I_box(self, m, sx, sy, sz, *, fudge: float | None = None):
        """Box inertia with a scaling fudge (empirical)."""
        k = self._I_FUDGE if fudge is None else fudge
        return (k * m * (sy**2 + sz**2) / 12.0,
                k * m * (sx**2 + sz**2) / 12.0,
                k * m * (sx**2 + sy**2) / 12.0)

    def _raw_genome_values(self) -> list[float]:
        """Return the 15-value genome (re-derived if needed)."""
        if self._raw_genome is not None:
            return self._raw_genome
        p = self.p
        # Reconstruct ratios if only GeometryParams were provided
        return [
            p.wing_span,
            p.wing_span / p.wing_chord if p.wing_chord else float("nan"),
            p.fus_length,
            p.fus_cg_x / p.fus_length if p.fus_length else float("nan"),
            p.wing_attach_x / p.fus_length if p.fus_length else float("nan"),
            p.elevator_span,
            p.elevator_span / p.elevator_chord if p.elevator_chord else float("nan"),
            p.rudder_span,
            p.rudder_span / p.rudder_chord if p.rudder_chord else float("nan"),
            p.dihedral_deg,
            p.hinge_le_ratio,
            p.sweep_multi,
            p.twist_multi,
            p.cl_alpha_2d,
            math.degrees(p.alpha0_2d),
        ]

    # Mesh scale mapping -----------------------------------------------------
    def _scale(self, what: str, dims: Tuple[float, float, float]) -> str:
        rs = self._REF_SCALE[what]

        if what == "fuselage":
            tfus, hfus, fus_len = dims
            tfus_ref, hfus_ref, fus_len_ref = (self._REF["tfus"], self._REF["hfus"], self._REF["fus_len"])
            sx = rs[0] * fus_len / fus_len_ref
            sy = rs[1] * hfus / hfus_ref
            sz = rs[2] * tfus / tfus_ref
            return f"{sx:.6g} {sy:.6g} {sz:.6g}"

        ref_dims = {
            "wing":     (self._REF["tw"],  self._REF["wing_chord"], self._REF["wing_span_ref"]),
            "elevator": (self._REF["tc"],  self._REF["elev_chord"], self._REF["elev_span"]),
            "rudder":   (self._REF["tc"],  self._REF["rudd_chord"], self._REF["rudd_span"]),
        }[what]
        sx, sy, sz = (rs[i] * dims[i] / ref_dims[i] for i in range(3))
        return f"{sx:.6g} {sy:.6g} {sz:.6g}"

    # ────────────────────────────────────────────────────────────────────
    # Materials & root
    # ────────────────────────────────────────────────────────────────────
    def _materials(self, robot: ET.Element) -> None:
        for n, rgba in {"grey": (0.7, 0.7, 0.7, 1), "red": (0.7, 0, 0, 1), "black": (0.2, 0.2, 0.2, 1)}.items():
            m = ET.SubElement(robot, "material", name=n)
            ET.SubElement(m, "color", rgba=" ".join(map(str, rgba)))

    def _root_link(self, robot: ET.Element) -> None:
        self._add_inertial(ET.SubElement(robot, "link", name="root_link"), (0, 0, 0), 0, (0, 0, 0))

    def _add_fixed_mass(self, robot: ET.Element, *,
                        name: str, parent: str, xyz, rpy: str = "0 0 0",
                        m: float,
                        size: Tuple[float, float, float] | None = None,
                        sphere_radius: float = 0.015):
        """
        Crea un link con massa m fissato al parent.
        Se 'size' è dato -> inerzia box; altrimenti -> inerzia sferica con raggio 'sphere_radius'.
        """
        if m <= 0:
            I = (0.0, 0.0, 0.0)
        else:
            if size is not None:
                sx, sy, sz = size
                I = self._I_box(m, sx, sy, sz, fudge=1.0)
            else:
                r = sphere_radius
                Ixx = Iyy = Izz = (2.0/5.0) * m * (r*r)  # sfera piena
                I = (Ixx, Iyy, Izz)

        ln = ET.SubElement(robot, "link", name=name)
        self._add_inertial(ln, (0, 0, 0), m, I)

        j = ET.SubElement(robot, "joint", name=f"fixed_joint_{name}", type="fixed")
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child",  link=name)
        self._origin(j, xyz, rpy)



    # ────────────────────────────────────────────────────────────────────
    # Fuselage
    # ────────────────────────────────────────────────────────────────────
    def _fuselage(self, robot: ET.Element) -> None:
        p = self.p
        box = (self._REF["tfus"], self._REF["hfus"], p.fus_length)
        vol = math.prod(box)

        # Mass model
        m_batt   = self._FUS_FIXED_MASS
        m_shell  = vol * self._RHO_FUS_STRUCT * self.S
        m_tot    = m_batt + m_shell

        # Inertia (shell + battery)
        I_shell = self._I_box(m_shell, *box)
        bx, by, bz = self._BATTERY_SIZE
        I_batt  = self._I_box(m_batt, bx, by, bz, fudge=1.0)
        I_tot   = tuple(a + b for a, b in zip(I_shell, I_batt))

        ln = ET.SubElement(robot, "link", name="fuselage")
        self._add_inertial(ln, (p.fus_cg_x, 0, -0.04), m_tot, I_tot)

        box_coll = (box[0] * self._SHRINK, box[1] * self._SHRINK, box[2] * self._SHRINK)
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

    # ────────────────────────────────────────────────────────────────────
    # Prop frame
    # ────────────────────────────────────────────────────────────────────
    def _prop_frame(self, robot: ET.Element) -> None:
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
        ET.SubElement(ET.SubElement(coll, "geometry"), "box", size="0.2 0.2 0.005")

    # ────────────────────────────────────────────────────────────────────
    # Wings (with sweep & twist joints + LE shift for visuals/frames)
    # ────────────────────────────────────────────────────────────────────
    def _wings(self, robot: ET.Element) -> None:
        p = self.p
        t = self._REF["tw"]

        # Span region within prop disc (per side)
        sp_prop = max(0.0, min(p.wing_span, max(0.0, p.prop_radius - self._ROOT_Y_OFFSET)))
        sp_free = p.wing_span - sp_prop
        rho = self._RHO_WING * self.S
        c   = p.wing_chord

        # helper: make a small service link with inertia
        def svc(name: str):
            ln = ET.SubElement(robot, "link", name=name)
            self._add_inertial(ln, (0, 0, 0), self._MASS_INTER,
                               self._I_box(self._MASS_INTER, t, t, t))

        # helper: create a wing panel link (prop / free)
        def panel(side: str, lbl: str, span: float, y_off: float):
            name = f"{side}_wing_{lbl}"
            box = (t, c, span)
            m   = math.prod(box) * rho
            I   = self._I_box(m, *box)

            j = ET.SubElement(robot, "joint", name=f"fixed_joint_{name}", type="fixed")
            ET.SubElement(j, "parent", link=f"{side}_wing")
            ET.SubElement(j, "child",  link=name)
            self._origin(j, (0, y_off, 0))

            ln = ET.SubElement(robot, "link", name=name)
            # Inertial CG from LE shifted by hinge_le_ratio
            self._add_inertial(ln, ((self._CG_RATIO - self.LE) * c, 0, 0), m, I)

            coll = ET.SubElement(ln, "collision", name=f"{name}_collision")
            self._origin(coll, (self._COLL_RATIO * c, 0, -0.000421042), self._RPY_WING_COLL)
            ET.SubElement(ET.SubElement(coll, "geometry"), "box",
                          size=" ".join(f"{v:.12g}" for v in box))

            # Aero frame anchored at CG (LE-based frame)
            ja = ET.SubElement(robot, "joint", name=f"fixed_joint_aero_frame_{name}", type="fixed")
            ET.SubElement(ja, "parent", link=name)
            ET.SubElement(ja, "child",  link=f"aero_frame_{name}")
            self._origin(ja, ((self._CG_RATIO - self.LE) * c, 0, 0), "3.141592654 0 0")
            self._add_inertial(ET.SubElement(robot, "link", name=f"aero_frame_{name}"), (0, 0, 0), 0, (0, 0, 0))

        # one wing side (right or left)
        def wing_side(side: str, sgn: int):
            # limits scale with sweep/twist multipliers (kept compatible)
            limit_sweep  = 0.7 / max(self.p.sweep_multi, 0.5)   # [rad]
            limit_twist  = 0.7 / max(self.p.twist_multi, 0.5)   # [rad]
            effort_sweep = 1.0 * max(self.p.sweep_multi, 0.5)
            effort_twist = 0.5 * max(self.p.twist_multi, 0.5)

            # attach the side "root" to fuselage with dihedral
            fj = ET.SubElement(robot, "joint", name=f"fixed_joint_{side}_wing", type="fixed")
            ET.SubElement(fj, "parent", link="fuselage")
            ET.SubElement(fj, "child",  link=f"fuselage_{side}_0")
            self._origin(fj, (p.wing_attach_x, self._ROOT_Y_OFFSET*sgn, -0.03),
                         f"{self.dihedral*sgn:.12g} 0 3.141592653589793")

            svc(f"fuselage_{side}_0")
            svc(f"fuselage_{side}_1")

            # 1) SWEEP joint: fus0 → fus1
            js = ET.SubElement(robot, "joint", name=f"joint_0_sweep_{side}_wing", type="revolute")
            ET.SubElement(js, "parent", link=f"fuselage_{side}_0")
            ET.SubElement(js, "child",  link=f"fuselage_{side}_1")
            self._origin(js, (0, 0, 0))
            ET.SubElement(js, "axis", xyz="0 0 1")
            ET.SubElement(js, "limit",
                          lower=f"{-limit_sweep:.6g}", upper=f"{limit_sweep:.6g}",
                          effort=f"{effort_sweep:.6g}", velocity=self._REV_LIMIT["velocity"])
            ET.SubElement(js, "dynamics", **self._REV_DYN)

            self._add_fixed_mass(robot,
                name=f"{side}_wing_servo_sweep",
                parent=f"fuselage_{side}_0",
                xyz=(0, 0, 0),
                m=self._SERVO_WING_SWEEP_MASS,
                size=None,                 # niente box
                sphere_radius=0.015        # ~15 mm
            )

            # 2) TWIST joint: fus1 → {side}_wing
            jt = ET.SubElement(robot, "joint", name=f"joint_1_twist_{side}_wing", type="revolute")
            ET.SubElement(jt, "parent", link=f"fuselage_{side}_1")
            ET.SubElement(jt, "child",  link=f"{side}_wing")
            self._origin(jt, (0, -self._ROOT_Y_OFFSET*sgn, 0))
            ET.SubElement(jt, "axis", xyz="0 1 0")
            ET.SubElement(jt, "limit",
                          lower=f"{-limit_twist:.6g}", upper=f"{limit_twist:.6g}",
                          effort=f"{effort_twist:.6g}", velocity=self._REV_LIMIT["velocity"])
            ET.SubElement(jt, "dynamics", **self._REV_DYN)

            self._add_fixed_mass(robot,
                name=f"{side}_wing_servo_twist",
                parent=f"fuselage_{side}_1",
                xyz=(0, -self._ROOT_Y_OFFSET*sgn, 0),
                m=self._SERVO_WING_TWIST_MASS,
                size=None,
                sphere_radius=0.015
            )


            # Root link for the side (visual is the whole half-wing mesh)
            root = ET.SubElement(robot, "link", name=f"{side}_wing")
            self._add_inertial(root, (0, 0, 0), 0, (0, 0, 0))
            vr = ET.SubElement(root, "visual", name=f"{side}_wing_visual")
            rpy = "3.141592654 0 0" if side == "left" else "0 0 0"
            vis_shift_x = (self.LE - self._LE_REF) * c
            self._origin(vr, (vis_shift_x, 0, 0), rpy)
            mr = ET.SubElement(ET.SubElement(vr, "geometry"), "mesh", filename="package://meshes/wing0009.stl")
            mr.set("scale", self._scale("wing", (t, c, sp_prop + sp_free)))
            ET.SubElement(vr, "material", name="red")

            # Two boxes for collision/aero frames: prop-wash panel + free panel
            if sp_prop > 0:
                panel(side, "prop", sp_prop, -sgn*sp_prop/2)
            if sp_free > 0:
                panel(side, "free", sp_free, -sgn*(sp_prop + sp_free/2))

        wing_side("right", +1)
        wing_side("left",  -1)

    # ────────────────────────────────────────────────────────────────────
    # Elevator (pitch) – LE-based frames for aero, legacy signs for coll
    # ────────────────────────────────────────────────────────────────────
    def _elevator(self, robot: ET.Element) -> None:
        p = self.p
        box_full = (self._REF["tc"], p.elevator_chord, p.elevator_span)
        box_half = (box_full[0], box_full[1], p.elevator_span/2)
        rho = self._RHO_ELEV * self.S
        m_half  = math.prod(box_half) * rho
        I_half  = self._I_box(m_half, *box_half)
        span_half = p.elevator_span / 2
        c = p.elevator_chord

        # Main revolute hinge (pitch)
        jh = ET.SubElement(robot, "joint", name="elevator_pitch_joint", type="revolute")
        ET.SubElement(jh, "parent", link="fuselage")
        ET.SubElement(jh, "child",  link="elevator_hinge")
        self._origin(jh, (- p.fus_length - 0.03, 0, 0))
        ET.SubElement(jh, "axis", xyz="0 1 0")
        ET.SubElement(jh, "limit", lower="-0.35", upper="0.35", effort="0.5", velocity="3.665191429")
        ET.SubElement(jh, "dynamics", damping="0.2", friction="0.05")

        # Servo elevatore
        self._add_fixed_mass(robot,
            name="elevator_servo",
            parent="fuselage",
            xyz=(-p.fus_length - 0.03, 0, 0),
            m=self._SERVO_TAIL_ELEV_MASS,
            size=None
        )

        ln_hinge = ET.SubElement(robot, "link", name="elevator_hinge")
        self._add_inertial(ln_hinge, (0, 0, 0), 0, (0, 0, 0))

        # Whole visual mesh (for convenience)
        vis_h = ET.SubElement(ln_hinge, "visual")
        self._origin(vis_h, (0, 0, 0))
        mesh_h = ET.SubElement(ET.SubElement(vis_h, "geometry"), "mesh",
                               filename="package://meshes/elevator.stl")
        mesh_h.set("scale", self._scale("elevator", (self._REF["tc"], p.elevator_chord, p.elevator_span)))
        ET.SubElement(vis_h, "material", name="grey")

        for sgn, side in ((+1, "left"), (-1, "right")):
            fj = ET.SubElement(robot, "joint", name=f"fixed_joint_elevator_{side}", type="fixed")
            ET.SubElement(fj, "parent", link="elevator_hinge")
            ET.SubElement(fj, "child",  link=f"elevator_{side}")
            self._origin(fj, (0, span_half*sgn, 0))

            ln = ET.SubElement(robot, "link", name=f"elevator_{side}")
            # CG from LE (+ forward); aero frame will mirror this
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

    # ────────────────────────────────────────────────────────────────────
    # Rudder (yaw)
    # ────────────────────────────────────────────────────────────────────
    def _rudder(self, robot: ET.Element) -> None:
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
        ET.SubElement(jy, "limit", lower="-0.35", upper="0.35", effort="0.5", velocity="3.665191429")
        ET.SubElement(jy, "dynamics", damping="0.2", friction="0.05")

        # Servo timone
        self._add_fixed_mass(robot,
            name="rudder_servo",
            parent="elevator_hinge",
            xyz=(0, 0, 0),
            m=self._SERVO_TAIL_RUDD_MASS,
            size=None
        )


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

    # ────────────────────────────────────────────────────────────────────
    # Metadata & export
    # ────────────────────────────────────────────────────────────────────
    def _add_metadata(self, robot: ET.Element) -> None:
        """Write a single XML comment with the raw 15-gene genome."""
        comment_text = "[" + ", ".join(f"{v:g}" for v in self._raw_genome_values()) + "]"
        robot.append(ET.Comment(comment_text))

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
        """Genome as filename-friendly string."""
        return "[" + ", ".join(f"{v:g}" for v in self._raw_genome_values()) + "]"

    def create_urdf(self, filename: str | None = None) -> str:
        """
        Write the URDF to disk.
        If *filename* is None, the genome list is used as the filename.
        """
        if filename is None:
            filename = f"{self._params_as_string()}.urdf"

        self.out.mkdir(parents=True, exist_ok=True)
        tree = self.build_tree()
        self._indent(tree.getroot())
        out = (self.out / filename)
        tree.write(out, encoding="utf-8", xml_declaration=True)
        return str(out)

# Quick self-test ---------------------------------------------------------
if __name__ == "__main__":
    # A simple genome example (values chosen for sanity, not performance)
    genome = [
        0.70, 3.5,     # wing span, AR -> chord ≈ 0.20
        0.73, -0.38,   # fus length, cg_x_ratio -> cg_x ≈ -0.277
        -0.38,         # attach_x_ratio -> attach_x ≈ -0.277
        0.18, 1.3,     # elevator span, AR -> chord ≈ 0.138
        0.16, 1.3,     # rudder   span, AR -> chord ≈ 0.123
        0.0,           # dihedral [deg]
        0.25,          # hinge_le_ratio
        2.0, 2.5,      # sweep, twist multipliers
        2, -3.0 # cl_alpha_2d, alpha0_2d [deg]
    ]
    print("URDF written to:", UrdfMaker(genome).create_urdf())
