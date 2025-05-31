import numpy as np
import open3d as o3d
import time
import os

file_path = 'demo_data/laptop/demo_0.npy'
demo = np.load(file_path)

vis = o3d.visualization.Visualizer()
vis.create_window()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(demo[0][:, :3])
pcd.paint_uniform_color([0.1, 0.7, 0.1])

vis.add_geometry(pcd)

for i, frame in enumerate(demo):
    points = frame[:, :3]
    pcd.points = o3d.utility.Vector3dVector(points)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.10)  # control the frame rate

vis.destroy_window()


# demo_dir = 'demo_data/laptop'
# for file in sorted(os.listdir(demo_dir)):
#     if file.endswith('.npy'):
#         path = os.path.join(demo_dir, file)
#         demo = np.load(path)
#         print(f"{file}: {len(demo)} frames")