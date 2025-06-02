import numpy as np
import os
import genesis as gs
import xml.etree.ElementTree as ET

import torch

def print_urdf_collision_boxes(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links = root.findall("link")
    print(f"DEBUG: Found {len(links)} link in {urdf_path}", flush=True)

    print("=== Collision‐box in URDF ===", flush=True)
    for link in links:
        link_name = link.get("name")
        collisions = link.findall("collision")
        print(f"  → Link '{link_name}' has {len(collisions)} collision tag", flush=True)
        for i, coll in enumerate(collisions):
            orig = coll.find("origin")
            xyz = orig.get("xyz", "0 0 0")
            rpy = orig.get("rpy", "0 0 0")
            box = coll.find("geometry/box")
            size = box.get("size")
            print(f"    [{i}] size=({size}), origin=({xyz}), rpy=({rpy})", flush=True)


def main():


    # Resolve absolute paths and validate they exist
    drone1_path = "/home/andrea/Documents/genesis/Genesis/genesis/assets/urdf/mydrone/mydrone_new.urdf"
    drone2_path = "/home/andrea/Documents/genesis/Genesis/genesis/assets/urdf/mydrone/bix3_new.urdf"
    for p in (drone1_path, drone2_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"URDF not found: {p}")
        
    print_urdf_collision_boxes(drone1_path)
    print_urdf_collision_boxes(drone2_path)
    # ------------------------------------------------------------------
    # 1. Initialise Genesis
    # ------------------------------------------------------------------
    gs.init()

    # 2. Scene & viewer ---------------------------------------------------------------
    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 960),
            camera_pos=(3.5, 0.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.5),
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            plane_reflection=True,
            show_link_frame=False,
            show_cameras=False,
        )
    )

    # Add a ground plane for reference
    scene.add_entity(gs.morphs.Plane(pos=(0, 0, 0)))

    # 3. Load drones as generic URDFs -------------------------------------------------
    drone1 = scene.add_entity(gs.morphs.URDF(file=drone1_path,
                                    pos=(-0, -1.0, 0.25),
                                    fixed=True),
                                    surface = gs.surfaces.Collision(), 
                                    #vis_mode="collision"
                                    )

    drone2 = scene.add_entity(gs.morphs.URDF(file=drone2_path,
                                    pos=(0, 1.0, 0.25),
                                    fixed=True),
                                    surface = gs.surfaces.Collision(),
                                    #vis_mode="collision"
                                    )
    # 4. Run the viewer ---------------------------------------------------------------
    scene.build()

    servo_joint_names = [
        "joint_0_twist_left_wing",
        "joint_0_twist_right_wing",
        "elevator_pitch_joint",
        "rudder_yaw_joint",
    ]
    servo_dof_idx = [drone1.get_joint(n).dof_idx_local for n in servo_joint_names]

    N_SERVO = len(servo_joint_names)
    kp = torch.full((N_SERVO,), 500.0, dtype=torch.float32)
    kv = torch.full((N_SERVO,),   10.0, dtype=torch.float32)
    max_tau = 5.0  # [Nm], scegli in base ai carichi aerodinamici

    # 1) crea un vettore di indici LONG sullo stesso device della simulazione
    servo_idx = torch.tensor(servo_dof_idx, dtype=torch.long)

    # 2) usa SEMPRE quel tensor per impostare guadagni / range
    drone1.set_dofs_kp(kp, servo_idx)
    drone1.set_dofs_kv(kv, servo_idx)
    drone1.set_dofs_force_range(
        upper=torch.full((N_SERVO,),  max_tau),
        lower=torch.full((N_SERVO,), -max_tau),
        dofs_idx_local=servo_idx
    )

    # ---------------- Stato iniziale --------------------------------------------
    # 3) imposta la posizione iniziale dei servomeccanismi
    servo_pos = torch.zeros((N_SERVO), dtype=torch.float32)
    servo_pos[:2] = -0.                            # -0.1 rad pre-twist
    servo_pos[2] = 0.0                            # -0.1 rad pre-twist
    servo_pos[3] = 0.0                            # -0.1 rad pre-twist
    drone1.set_dofs_position(servo_pos, servo_idx)

    print("Viewer running … ESC to quit.")
    try:
        while True:
            drone1.set_dofs_position(servo_pos, servo_idx)
            scene.step()


    except KeyboardInterrupt:
        print("\nSimulazione terminata.")


if __name__ == "__main__":
    main()
