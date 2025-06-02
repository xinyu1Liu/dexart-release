
import open3d as o3d
import numpy as np
import pickle
import time
import os

with open("/data/xinyu/demo_DexArt_1036/laptop/demo_100.pkl", "rb") as f:
    demo_data = pickle.load(f)

output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)


vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Demo Playback', width=1920, height=1080)
coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
vis.add_geometry(coord)

ctr = vis.get_view_control()
ctr.set_zoom(0.05)

# Initialize with first frame
obs_pc = o3d.geometry.PointCloud()
obs_pc.points = o3d.utility.Vector3dVector(demo_data[0]['obs']['observed_point_cloud'])
obs_pc.paint_uniform_color([0, 0.6, 1])
vis.add_geometry(obs_pc)

imagine_pc = o3d.geometry.PointCloud()
imagine_pc.points = o3d.utility.Vector3dVector(demo_data[0]['obs']['imagined_robot_point_cloud'])
imagine_pc.paint_uniform_color([1.0, 0.6, 0])
vis.add_geometry(imagine_pc)

vis.poll_events()
vis.update_renderer()
first_frame_path = os.path.join(output_dir, "frame_0000.png")
vis.capture_screen_image(first_frame_path)


for i, observed in enumerate(demo_data[1:], start=1):
    obs = observed['obs']
    obs_pc.points = o3d.utility.Vector3dVector(obs['observed_point_cloud'])
    imagine_pc.points = o3d.utility.Vector3dVector(obs['imagined_robot_point_cloud'])
    
    vis.update_geometry(obs_pc)
    vis.update_geometry(imagine_pc)
    
    vis.poll_events()
    vis.update_renderer()

    image_path = os.path.join(output_dir, f"frame_{i:04d}.png")
    vis.capture_screen_image(image_path)

    time.sleep(0.15)  # playback speed

print("Playback finished. Close window to exit.")
vis.run()
vis.destroy_window()

