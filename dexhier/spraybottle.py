import sapien
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
import torch
import numpy as np
import gym
import open3d as o3d
import random

import dexhier.xarm6_leap
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
import mani_skill.envs.utils.randomization as randomization
from typing import Any, Dict, Union, Optional
from mani_skill.utils.building import articulations
from mani_skill.utils.sapien_utils import set_articulation_render_material
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion, euler_angles_to_matrix, _axis_angle_rotation
from mani_skill.envs.scene import ManiSkillScene
from pytorch3d.ops import sample_farthest_points
import os
from pathlib import Path
import json

SPRAY_BOTTLE_CONFIGS = {
    "xarm6_leap": {
        "obj_half_length": 0.1,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.05,
        "cube_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        "sensor_cam_eye_pos": [
            0.3,
            0,
            0.6,
        ],  # sensor cam is the camera used for visual observation generation
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [
            0.6,
            0.7,
            0.6,
        ],  # human cam is the camera used for human rendering (i.e. eval videos)
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
}

SCALE_DICT = {'101463': 0.15}
OBJ_LIST = [101463]

@register_env("SprayBottle-v1", max_episode_steps=500)
class SprayBottleEnv(BaseEnv):
    def __init__(self, *args, robot_uids="xarm6_leap", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in SPRAY_BOTTLE_CONFIGS:
            cfg = SPRAY_BOTTLE_CONFIGS[robot_uids]
        else:
            raise NotImplementedError
        self.obj_half_length = cfg["obj_half_length"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))
    
    def setup_instance_annotation(self):
        current_dir = Path(__file__).parent
        self.joint_dicts = dict()
        for instance_index in OBJ_LIST:  
            joint_json_path = current_dir.parent / "assets"  / "sapien" / str(
                instance_index) / "mobility_v2.json"
            with open(joint_json_path, 'r') as load_f:
                load_dict = json.load(load_f)
            self.joint_dicts[instance_index] = load_dict
    
    def load_instance(self, index):
        self.setup_instance_annotation()
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.load_multiple_collisions_from_file = True
        loader.fix_root_link = False
        loader.set_material(static_friction=1.0, dynamic_friction=0.8, restitution=0.0)
        current_dir = Path(__file__).parent
        urdf_path = str(current_dir.parent / "assets" / "sapien" / str(index) / "mobility.urdf")
        loader.scale = SCALE_DICT[str(index)] if SCALE_DICT.__contains__(str(index)) else 1

        instance: sapien.Articulation = loader.load(urdf_path)
        # for joint in instance.get_joints():
        #     joint.set_friction(5)

        load_dict = self.joint_dicts[index]

        # in sapien, it will auto add the fixed base joint, so the loaded joint_size need to plus 1
        # assert len(laptop.get_joints()) == joint_size + 1
        dof = 0
        revolute_joint = None
        for i, joint_entry in enumerate(load_dict):
            if joint_entry['joint'] == 'free':
                dof += 1
            if joint_entry['joint'] == 'hinge':
                revolute_joint_index = dof - 1
                revolute_joint = instance.get_active_joints()[revolute_joint_index]
        
        assert (dof == instance.dof).all(), "dof parse error, index={}, calculate_dof={}, real_dof={}".format(index, dof,
                                                                                                      instance.dof)
        assert revolute_joint, "revolue_joint can not be None!"
        return instance, revolute_joint, revolute_joint_index
    
    def load_goal_instance(self, index):
        builder = self.scene.create_actor_builder()
        builder.set_initial_pose(sapien.Pose())
        current_dir = Path(__file__).parent
        mesh_path = str(current_dir.parent / "assets" / "sapien" / str(index) / "combined.obj")
        builder.add_visual_from_file(filename=mesh_path, scale=SCALE_DICT[str(index)] * np.array([1,1,1]), material=[0, 1, 0])
        mesh = builder.build_kinematic(name="goal_site")
        return mesh
    
    def _load_scene(self, options: dict):
        # load table scene
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # load object instance
        model_id = random.choice(OBJ_LIST)  
        self.instance, self.revolute_joint, self.revolute_joint_index = self.load_instance(index=model_id)
        self.handle_link = self.revolute_joint.get_child_link()
        self.instance_links = self.instance.get_links()
        # set up revolute joint
        joint = self.instance.get_active_joints()[0]
        joint.set_drive_properties(
            stiffness=3, damping=1
        )
        
        # load goal instance
        self.goal_site = self.load_goal_instance(index=model_id)
        self._hidden_objects.append(self.goal_site)
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.last_openness = self.instance.get_qpos()[self.revolute_joint_index]
        joint = self.instance.get_active_joints()[0]
        joint.set_drive_target(0.2)

        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]

            xyz[:, 2] = self.obj_half_length
            qs = self.generate_random_quat(b, lock_x=True, lock_y=True,
                                bounds=((0, np.pi/3), (-np.pi/3, 0), (np.pi / 2, 3 * np.pi / 2)))
            self.instance.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
            goal_qs = self.generate_random_quat(b, lock_x=True, 
                bounds=((0, np.pi/3), (-np.pi/3, 0), (np.pi / 2, 3 * np.pi / 2))
            )
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz, goal_qs))

    def generate_random_quat(
        self,
        n: int,
        device = None,
        lock_x: bool = False,
        lock_y: bool = False,
        lock_z: bool = False,
        bounds = (
            (0, 2 * np.pi),
            (0, 2 * np.pi),
            (0, 2 * np.pi),
        ),
    ):
        xyz_angles = torch.zeros((n, 3), device=device)

        for i, (lock, (low, high)) in enumerate(zip(
            (lock_x, lock_y, lock_z),
            bounds
        )):
            if lock:
                xyz_angles[:, i] = 0.0
            else:
                xyz_angles[:, i] = torch.rand(n, device=device) * (high - low) + low

        # rot_mats = euler_angles_to_matrix(xyz_angles, convention="XYZ")  # this is intrinsics, not correct
        # we implement extrinsic rotation ourselves here.
        matrices = [
            _axis_angle_rotation(c, e)
            for c, e in zip("XYZ", torch.unbind(xyz_angles, -1))
        ]
        rot_mats = torch.matmul(torch.matmul(matrices[2], matrices[1]), matrices[0])
        return matrix_to_quaternion(rot_mats)

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.instance.pose.raw_pose,
                tcp_to_obj_pos=self.instance.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.instance.pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.instance.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.instance)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }
    
    def get_obs(self, info: Optional[Dict] = None, unflattened: bool = False):
        obs = super().get_obs(info=info, unflattened=unflattened)
        # post processing point cloud information
        if self._obs_mode == "pointcloud":
            B = obs['pointcloud']['xyzw'].shape[0]
            processed_pcd = []
            for env_id in range(B):
                points = obs["pointcloud"]["xyzw"][env_id, :, :3].cpu().numpy()  # [N, 3]
                xyz_min = np.array([-10, -10, 0.01])
                xyz_max = np.array([0.5, 10, 1])
                mask = (points >= xyz_min) & (points <= xyz_max)
                mask = mask.all(axis=-1)
                cropped_pcd = points[mask]

                seg = obs['pointcloud']['segmentation'][env_id, mask, 0].cpu().numpy()
                seg_obj = seg >= 26
                seg_hand = (seg < 26) & (seg >= 8)
                onehot = np.zeros_like(cropped_pcd)
                onehot[seg_obj, 0] = 1
                onehot[seg_hand, 1] = 1
                onehot[~(seg_hand | seg_obj), 2] = 1
                cropped_pcd = np.concatenate([cropped_pcd, onehot], axis=-1)  # [N, 6]
                
                if len(cropped_pcd) < 1024:
                    # random upsample to 1024
                    num_pad = 1024 - len(cropped_pcd)
                    indices = np.random.choice(len(cropped_pcd), num_pad)
                    padded_xyz = cropped_pcd[indices]
                    cropped_pcd = np.concatenate([cropped_pcd, padded_xyz], 0)
                points = torch.tensor(cropped_pcd, dtype=torch.float32).unsqueeze(0)  # [1, N, 3]
                sampled_pts, _ = sample_farthest_points(points, K=1024)  # [1, 1024, 3]
                processed_pcd.append(sampled_pts.squeeze(0))
            processed_pcd = torch.stack(processed_pcd, dim=0)  # [B, 1024, 3]
            obs["pointcloud"]["xyzw"] = processed_pcd
        return obs
        


    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.instance.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.instance.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel = self.agent.robot.get_qvel()
        if self.robot_uids in ["panda", "widowxai"]:
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5