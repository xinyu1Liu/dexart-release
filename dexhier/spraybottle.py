import sapien
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
import torch
import numpy as np
import gym
import open3d as o3d

import dexhier.xarm6_leap
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
import mani_skill.envs.utils.randomization as randomization
from typing import Any, Dict, Union, Optional
from mani_skill.utils.building import articulations
from mani_skill.envs.scene import ManiSkillScene
from pytorch3d.ops import sample_farthest_points
import os
from pathlib import Path
import json

@register_env("SprayBottle-v1", max_episode_steps=500)
class SprayBottleEnv(BaseEnv):
    cube_half_size = 0.1
    goal_thresh = 0.025
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)

    def __init__(self, *args, robot_uids="xarm6_leap", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        # self.cube_half_size = cfg["cube_half_size"]
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

    def get_object_loader(self, scene: ManiSkillScene, id: str, fix_root_link=False, urdf_config: dict = dict()):
        loader = scene.create_urdf_loader()
        loader.fix_root_link = fix_root_link
        loader.scale = 0.15
        loader.load_multiple_collisions_from_file = True
        urdf_path = os.path.join('/home/yingyuan/dexart-release/assets/sapien', id, 'mobility.urdf')
        applied_urdf_config = sapien_utils.parse_urdf_config(
            dict(
                material=dict(static_friction=1, dynamic_friction=1, restitution=0),
            )
        )
        applied_urdf_config.update(**urdf_config)
        sapien_utils.apply_urdf_config(loader, applied_urdf_config)
        articulation_builders = loader.parse(str(urdf_path))["articulation_builders"]
        builder = articulation_builders[0]
        return builder
    
    def setup_instance_annotation(self):
        current_dir = Path(__file__).parent
        # self.scale_path = current_dir.parent / "assets" / "annotation" / "laptop_scale.json"
        # if os.path.exists(self.scale_path):
        #     with open(self.scale_path, "r") as f:
        #         self.scale_dict = json.load(f)
        # else:
        self.scale_dict = {'101463': 0.15}  # spraybottle scale
        self.joint_dicts = dict()
        for instance_index in [101463]:  # TASK_CONFIG['laptop']:
            joint_json_path = current_dir.parent / "assets"  / "sapien" / str(
                instance_index) / "mobility_v2.json"
            with open(joint_json_path, 'r') as load_f:
                load_dict = json.load(load_f)
            self.joint_dicts[instance_index] = load_dict
        # self.joint_limits_dict_path = current_dir.parent / "assets" / "annotation" /"laptop_joint_annotation.json"
        # self.joint_limits_dict = dict()
        # if os.path.exists(self.joint_limits_dict_path):
        #     with open(self.joint_limits_dict_path, "r") as f:
        #         self.joint_limits_dict = json.load(f)
    
    def load_instance(self, index):
        self.setup_instance_annotation()
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.load_multiple_collisions_from_file = True
        loader.fix_root_link = False
        current_dir = Path(__file__).parent
        urdf_path = str(current_dir.parent / "assets" / "sapien" / str(index) / "mobility.urdf")
        loader.scale = self.scale_dict[str(index)] if self.scale_dict.__contains__(str(index)) else 1

        instance: sapien.Articulation = loader.load(urdf_path)
        for joint in instance.get_joints():
            joint.set_friction(5)
            # joint.set_drive_property(stiffness=1e6, damping=1e2)
            # joint.set_drive_target(0.2)
            # joint.set_drive_property(0, 5)

        load_dict = self.joint_dicts[index]
        joint_size = len(load_dict)
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
    
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # self.cube = actors.build_cube(
        #     self.scene,
        #     half_size=self.cube_half_size,
        #     color=[1, 0, 0, 1],
        #     name="cube",
        #     initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        # )
        model_id = 101463
        # builder = self.get_object_loader(self.scene, str(model_id), fix_root_link=False)
        # builder.initial_pose = sapien.Pose(p=[0, 0, self.cube_half_size])
        # self.instance = builder.build(name="spraybottle")
        
        self.instance, self.revolute_joint, self.revolute_joint_index = self.load_instance(index=model_id)
        self.handle_link = self.revolute_joint.get_child_link()
        self.instance_links = self.instance.get_links()
        # self.instance_collision_links = [link for link in self.instance.get_links() if
        #                                     len(link.get_collision_shapes()) > 0]
        # self.handle_id = self.handle_link.get_id()
        # self.instance_ids_without_handle = [link.get_id() for link in self.instance_links]
        # self.instance_ids_without_handle.remove(self.handle_id)
        self.last_openness = self.instance.get_qpos()[self.revolute_joint_index]
        joint = self.instance.get_active_joints()[0]  # get the joint you want to control

        joint.set_drive_properties(
            stiffness=3, damping=1
        )  # set the drive properties
        joint.set_drive_target(0.2)  # set the target position of the joint
        
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
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

            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.instance.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

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
            assert obs["pointcloud"]["xyzw"].shape[0] == 1
            points = obs["pointcloud"]["xyzw"][0, :, :3].cpu().numpy()
            xyz_min = np.array([-10, -10, 0.01])
            xyz_max = np.array([0.5, 10, 1])
            mask = (points >= xyz_min) & (points <= xyz_max)
            mask = mask.all(axis=-1)
            cropped_pcd = points[mask]

            seg = obs['pointcloud']['segmentation'][0, mask, 0].cpu().numpy()
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
            sampled_pts, _ = sample_farthest_points(points, K=1024)
            obs["pointcloud"]["xyzw"] = sampled_pts
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