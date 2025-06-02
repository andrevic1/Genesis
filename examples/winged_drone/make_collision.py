#!/usr/bin/env python3
"""
auto_collision_boxes.py
───────────────────────────────────────────────────────────────────────────────
Genera un parallelepipedo AABB per ogni link e sostituisce i tag <collision>
direttamente nell’XML dell’URDF. Non servono librerie esterne per il parsing.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
from trimesh.transformations import euler_from_matrix
# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE: sostituisci questi path assoluti col tuo URDF e cartella mesh
URDF_INPUT    = Path("/home/andrea/Documents/genesis/Genesis/genesis/assets/urdf/mydrone/mydrone.urdf")
URDF_OUTPUT   = Path("/home/andrea/Documents/genesis/Genesis/genesis/assets/urdf/mydrone/mydrone_new.urdf")
MESH_ROOT     = Path("/home/andrea/Documents/genesis/Genesis/genesis/assets/urdf/mydrone")
# ──────────────────────────────────────────────────────────────────────────────

def resolve_mesh_path(filename: str, mesh_root: Path) -> Path:
    if filename.startswith("package://"):
        return mesh_root / filename[len("package://"):]
    return Path(filename)

def parse_origin(origin_elem):
    """Ritorna (xyz: np.array(3), rpy: np.array(3)) da <origin>."""
    xyz = origin_elem.get("xyz", "0 0 0")
    rpy = origin_elem.get("rpy", "0 0 0")
    return np.fromstring(xyz, sep=" "), np.fromstring(rpy, sep=" ")

def make_collision_element(link_elem):
    """
    Costruisce il parallelepipedo OBB per il link:
    - trasforma tutte le mesh nel frame link
    - raccoglie tutti i vertici
    - calcola bounding_box_oriented
    - restituisce <collision> con origin e box size
    """
    all_vertices = []

    for vis in link_elem.findall("visual"):
        geom = vis.find("geometry")
        mesh_elem = geom.find("mesh")
        if mesh_elem is None:
            continue

        # carica mesh
        mesh_path = resolve_mesh_path(mesh_elem.get("filename"), MESH_ROOT)
        try:
            mesh = trimesh.load_mesh(mesh_path, process=False)
        except Exception:
            print(f"⚠️  Mesh non trovata o invalida: {mesh_path}")
            continue

        # scala
        scale = np.fromstring(mesh_elem.get("scale", "1 1 1"), sep=" ")
        mesh.apply_scale(scale)

        # trasforma con l’origin del visual, se c’è
        origin_elem = vis.find("origin")
        if origin_elem is not None:
            xyz, rpy = parse_origin(origin_elem)
            T = trimesh.transformations.euler_matrix(rpy[0], rpy[1], rpy[2])
            T[:3,3] = xyz
            mesh.apply_transform(T)

        all_vertices.append(mesh.vertices)

    if not all_vertices:
        return None

    # unisco tutti i vertici
    verts = np.vstack(all_vertices)

    # calcolo OBB: parallelepipedo orientato minimo
    cloud = trimesh.Trimesh(vertices=verts, process=False)
    obb = cloud.bounding_box_oriented   # OBB mesh
    size = obb.primitive.extents        # [lx, ly, lz]
    transform = obb.primitive.transform # 4×4 matrix

    # decomponi transform in xyz e rpy
    center = transform[:3,3]
    R = transform[:3,:3]
    roll, pitch, yaw = euler_from_matrix(R, axes='sxyz')

    # costruisci element collision
    coll = ET.Element("collision")
    origin = ET.SubElement(coll, "origin")
    origin.set("xyz", f"{center[0]} {center[1]} {center[2]}")
    origin.set("rpy", f"{roll} {pitch} {yaw}")

    geom = ET.SubElement(coll, "geometry")
    box  = ET.SubElement(geom, "box")
    box.set("size", f"{size[0]} {size[1]} {size[2]}")

    return coll

def main():
    tree = ET.parse(URDF_INPUT)
    root = tree.getroot()

    for link in root.findall("link"):
        # rimuovi collisioni esistenti
        for old in link.findall("collision"):
            link.remove(old)
        # genera e inserisci nuova collisione OBB
        new_coll = make_collision_element(link)
        if new_coll is not None:
            link.append(new_coll)

    tree.write(URDF_OUTPUT, xml_declaration=True, encoding="utf-8")
    print(f"✔  Collision boxes OBB generate e salvate in:\n   {URDF_OUTPUT}")

if __name__ == "__main__":
    main()