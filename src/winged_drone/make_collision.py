#!/usr/bin/env python3
"""
make_collision.py
───────────────────────────────────────────────────────────────────────────────
Crea OBB di collisione per ogni link e assicura che QUALSIASI shape
(box, mesh, cylinder, sphere) abbia un attributo name distinto.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
from trimesh.transformations import euler_from_matrix

# ─── paths ───────────────────────────────────────────────────────────────────
URDF_IN  = Path("/home/andrea/Documents/genesis/Genesis/genesis/assets/urdf/mydrone/bix3.urdf")
URDF_OUT = Path("/home/andrea/Documents/genesis/Genesis/genesis/assets/urdf/mydrone/bix3_new.urdf")
MESHROOT = Path("/home/andrea/Documents/genesis/Genesis/genesis/assets/urdf/mydrone")
# -----------------------------------------------------------------------------

def _mesh_path(fname: str) -> Path:
    return MESHROOT / fname[len("package://"):] if fname.startswith("package://") else Path(fname)

def _origin(elem):
    xyz = np.fromstring(elem.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(elem.get("rpy",  "0 0 0"), sep=" ")
    return xyz, rpy

# ────────────────────────── naming helpers ───────────────────────────────────
def _unique(base: str, used: set) -> str:
    i = 0
    while f"{base}_{i}" in used:
        i += 1
    name = f"{base}_{i}"
    used.add(name)
    return name

def _ensure_shape_name(geom_elem, base_name: str, used: set):
    """Aggiunge name al nodo shape interno (<mesh>/<box>/…) se assente."""
    # individua il child che definisce lo shape
    for tag in ("mesh", "box", "cylinder", "sphere"):
        shape = geom_elem.find(tag)
        if shape is not None:
            if not shape.get("name", "").strip():
                shape.set("name", _unique(f"{base_name}_shape", used))
            break

# ─────────────────────── collision OBB creator ──────────────────────────────
def make_collision(link, used_names: set):
    """Restituisce un <collision> con box OBB + nomi univoci (collision e shape)."""
    verts = []
    for vis in link.findall("visual"):
        geom = vis.find("geometry")
        mesh = geom.find("mesh")
        if mesh is None:
            continue
        path = _mesh_path(mesh.get("filename"))
        try:
            m = trimesh.load_mesh(path, process=False)
        except Exception:
            print(" ⚠ mesh mancante:", path)
            continue
        m.apply_scale(np.fromstring(mesh.get("scale", "1 1 1"), sep=" "))
        org = vis.find("origin")
        if org is not None:
            xyz, rpy = _origin(org)
            T = trimesh.transformations.euler_matrix(*rpy)
            T[:3, 3] = xyz
            m.apply_transform(T)
        verts.append(m.vertices)

    if not verts:
        return None

    cloud = trimesh.Trimesh(vertices=np.vstack(verts), process=False)
    obb   = cloud.bounding_box_oriented
    size  = obb.primitive.extents
    T     = obb.primitive.transform
    center = T[:3, 3]
    roll, pitch, yaw = euler_from_matrix(T[:3, :3], axes="sxyz")

    # nodo collision
    coll_name = _unique(f"{link.attrib['name']}_collision", used_names)
    coll = ET.Element("collision", name=coll_name)
    ET.SubElement(coll, "origin",
                  xyz=" ".join(map(str, center)),
                  rpy=f"{roll} {pitch} {yaw}")
    geom = ET.SubElement(coll, "geometry")
    box  = ET.SubElement(geom, "box",
                         size=" ".join(map(str, size)),
                         name=_unique(f"{coll_name}_box", used_names))

    return coll


def main():
    used = set()
    tree = ET.parse(URDF_IN)
    root = tree.getroot()

    for link in root.findall("link"):
        # 1) nomi ai visual e ai loro shape
        for vis_idx, vis in enumerate(link.findall("visual")):
            if not vis.get("name"):
                vis.set("name", _unique(f"{link.attrib['name']}_visual", used))
            _ensure_shape_name(vis.find("geometry"), vis.attrib["name"], used)

        # 2) remove & rebuild collision OBB se ci sono visual
        for old in link.findall("collision"):
            link.remove(old)
        new_coll = make_collision(link, used)
        if new_coll is not None:
            link.append(new_coll)

    tree.write(URDF_OUT, encoding="utf-8", xml_declaration=True)
    print(f"✔ URDF aggiornato → {URDF_OUT}")

if __name__ == "__main__":
    main()
