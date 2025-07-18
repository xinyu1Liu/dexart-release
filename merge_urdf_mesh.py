import pybullet as p
import pybullet_data
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load URDF
body_id = p.loadURDF("assets/sapien/101463/mobility.urdf", useFixedBase=True)

# Step to allow FK to update
p.stepSimulation()

# Get all visual shapes
mesh_list = []
num_joints = p.getNumJoints(body_id)

for link_idx in range(-1, num_joints):  # -1 for base link
    visual_shapes = p.getVisualShapeData(body_id)
    for shape in visual_shapes:
        if shape[1] == link_idx:
            mesh_file = shape[4].decode("utf-8")
            mesh = trimesh.load_mesh(mesh_file)
            translation = p.getLinkState(body_id, link_idx)[4]
            rotation = p.getLinkState(body_id, link_idx)[5]
            
            r = R.from_quat(rotation)
            rot_matrix = r.as_matrix()

            transform = np.eye(4)
            transform[:3, :3] = rot_matrix
            transform[:3, 3] = translation

            # mesh.apply_transform(transform)
            mesh_list.append(mesh)

# Combine meshes
combined = trimesh.util.concatenate(mesh_list)

rot_matrix = np.array([
                [0,0,-1],[-1,0,0],[0,1,0]
            ])
transform = np.eye(4)
transform[:3, :3] = rot_matrix
combined.apply_transform(transform)

combined.export('combined.obj')
axes = trimesh.creation.axis(origin_size=0.02)
scene = trimesh.Scene([combined, axes])
scene.show()