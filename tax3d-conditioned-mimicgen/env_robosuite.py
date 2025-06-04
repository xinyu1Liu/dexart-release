"""
This file contains the robosuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import json
import numpy as np
import torch
from copy import deepcopy
import open3d as o3d

import robosuite
from robosuite.utils.camera_utils import get_real_depth_map, get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_camera_segmentation
try:
    # this is needed for ensuring robosuite can find the additional mimicgen environments (see https://mimicgen.github.io)
    import mimicgen_envs
except ImportError:
    pass

import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB
from pytorch3d.transforms import quaternion_to_matrix

import mujoco
import pickle

# protect against missing mujoco-py module, since robosuite might be using mujoco-py or DM backend
try:
    import mujoco_py
    MUJOCO_EXCEPTIONS = [mujoco_py.builder.MujocoException]
except ImportError:
    MUJOCO_EXCEPTIONS = []

def depth2fgpcd(depth, mask, cam_params):
    # depth: (h, w)
    # fgpcd: (n, 3)
    # mask: (h, w)
    h, w = depth.shape
    mask = np.logical_and(mask, depth > 0)
    # mask = (depth <= 0.599/0.8)
    fgpcd = np.zeros((mask.sum(), 3))
    fx, fy, cx, cy = cam_params
    pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
    fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
    fgpcd[:, 2] = depth[mask]
    return fgpcd

def np2o3d(pcd, color=None):
    # pcd: (n, 3)
    # color: (n, 3)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None and color.shape[0] > 0:
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1
        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d

class EnvRobosuite(EB.EnvBase):
    """Wrapper class for robosuite environments (https://github.com/ARISE-Initiative/robosuite)"""
    def __init__(
        self, 
        env_name, 
        render=False, 
        render_offscreen=False, 
        use_image_obs=False, 
        postprocess_visual_obs=True, 
        **kwargs,
    ):
        """
        Args:
            env_name (str): name of environment. Only needs to be provided if making a different
                environment from the one in @env_meta.

            render (bool): if True, environment supports on-screen rendering

            render_offscreen (bool): if True, environment supports off-screen rendering. This
                is forced to be True if @env_meta["use_images"] is True.

            use_image_obs (bool): if True, environment is expected to render rgb image observations
                on every env.step call. Set this to False for efficiency reasons, if image
                observations are not required.

            postprocess_visual_obs (bool): if True, postprocess image observations
                to prepare for learning. This should only be False when extracting observations
                for saving to a dataset (to save space on RGB images for example).
        """
        self.postprocess_visual_obs = postprocess_visual_obs

        # robosuite version check
        self._is_v1 = (robosuite.__version__.split(".")[0] == "1")
        if self._is_v1:
            assert (int(robosuite.__version__.split(".")[1]) >= 2), "only support robosuite v0.3 and v1.2+"

        kwargs = deepcopy(kwargs)

        # update kwargs based on passed arguments
        update_kwargs = dict(
            has_renderer=render,
            has_offscreen_renderer=(render_offscreen or use_image_obs),
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=use_image_obs,
            camera_depths=True,
        )
        kwargs.update(update_kwargs)

        if self._is_v1:
            if kwargs["has_offscreen_renderer"]:
                # ensure that we select the correct GPU device for rendering by testing for EGL rendering
                # NOTE: this package should be installed from this link (https://github.com/StanfordVL/egl_probe)
                import egl_probe
                valid_gpu_devices = egl_probe.get_available_devices()
                if len(valid_gpu_devices) > 0:
                    kwargs["render_gpu_device_id"] = valid_gpu_devices[0]
        else:
            # make sure gripper visualization is turned off (we almost always want this for learning)
            kwargs["gripper_visualization"] = False
            del kwargs["camera_depths"]
            kwargs["camera_depth"] = False # rename kwarg

        self._env_name = env_name
        self._init_kwargs = deepcopy(kwargs)
        self.env = robosuite.make(self._env_name, **kwargs)

        if self._is_v1:
            # Make sure joint position observations and eef vel observations are active
            for ob_name in self.env.observation_names:
                if ("joint_pos" in ob_name) or ("eef_vel" in ob_name):
                    self.env.modify_observable(observable_name=ob_name, attribute="active", modifier=True)

        voxel_center = np.array([0, 0, 0.7])
        pc_center = np.array([0, 0, 0.7])
        if hasattr(self.env, 'table_offset'):
            voxel_center[:2] = self.env.table_offset[:2]
            pc_center = np.array(self.env.table_offset)
            pc_center[2] = pc_center[2] + 0.02
        self.ws_size = 0.6
        if env_name.startswith('Kitchen_'):
            self.ws_size = 0.7
            pc_center = self.env.table_offset
        elif env_name.startswith('PickPlace_'):
            pc_center = np.array([0, 0, 0.83])
            self.ws_size = 1.1

        self.voxel_workspace = np.array([
            [voxel_center[0] - self.ws_size/2, voxel_center[0] + self.ws_size/2],
            [voxel_center[1] - self.ws_size/2, voxel_center[1] + self.ws_size/2],
            [voxel_center[2], voxel_center[2] + self.ws_size]
        ])
        self.pc_workspace = np.array([
            [pc_center[0] - self.ws_size/2, pc_center[0] + self.ws_size/2],
            [pc_center[1] - self.ws_size/2, pc_center[1] + self.ws_size/2],
            [pc_center[2], pc_center[2] + self.ws_size]
        ])

        # loading gripper mesh
        with open('eef_pointcloud_wiz_pose_info.pkl', 'rb') as f:
            reference_eef_data = pickle.load(f)
        self.reference_eef_data = reference_eef_data

        self.robot_eef_pointcloud = torch.tensor(self.reference_eef_data['pointcloud']).cuda()
        reference_quat = self.reference_eef_data['quat']
        reference_quat = torch.tensor([reference_quat[3], reference_quat[0], reference_quat[1], reference_quat[2]]).cuda()  # xyzw -> wxyz
        self.reference_rot = quaternion_to_matrix(reference_quat).float()
        self.reference_pos = torch.tensor(self.reference_eef_data['pos']).cuda()


    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, r, done, info = self.env.step(action)
        obs = self.get_observation(obs)
        return obs, r, self.is_done(), info

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        di = self.env.reset()
        self.gt_goal_gripper_pcd = None
        self.gt_goal_eef_pos = None
        self.gt_goal_eef_quat = None
        # self.compute_robot_eef_pointcloud()
        return self.get_observation(di)

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment
                - model (str): mujoco scene xml
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        should_ret = False

        if "model" in state:
            self.reset()
            robosuite_version_id = int(robosuite.__version__.split(".")[1])
            if robosuite_version_id <= 3:
                from robosuite.utils.mjcf_utils import postprocess_model_xml
                xml = postprocess_model_xml(state["model"])
            else:
                # v1.4 and above use the class-based edit_model_xml function
                xml = self.env.edit_model_xml(state["model"])
            self.env.reset_from_xml_string(xml)
            self.env.sim.reset()
            if not self._is_v1:
                # hide teleop visualization after restoring from model
                self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array([0., 0., 0., 0.])
                self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
        if "states" in state:
            if isinstance(state["states"], dict):
                self.env.sim.set_state_from_flattened(state["states"]["states"])

                if 'goal_gripper_pcd' in state["states"]:
                    self.gt_goal_gripper_pcd = state["states"]["goal_gripper_pcd"].cpu().numpy()
                else:
                    self.gt_goal_gripper_pcd = None
                if 'goal_eef_pos' in state["states"]:
                    self.gt_goal_eef_pos = state["states"]["goal_eef_pos"].cpu().numpy()
                    self.gt_goal_eef_quat = state["states"]["goal_eef_quat"].cpu().numpy()
                else:
                    self.gt_goal_eef_pos = None
                    self.gt_goal_eef_quat = None
                self.cur_goal_gripper_idx = 0

            else:
                self.env.sim.set_state_from_flattened(state["states"])
                self.gt_goal_gripper_pcd = None
                self.gt_goal_eef_pos = None
                self.gt_goal_eef_quat = None
                self.cur_goal_gripper_idx = None
            self.env.sim.forward()
            should_ret = True

        if "goal" in state:
            self.set_goal(**state["goal"])

        if should_ret:
            # only return obs if we've done a forward call - otherwise the observations will be garbage
            return self.get_observation()

        return None

    def render(self, mode="human", height=None, width=None, camera_name="agentview"):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
            camera_name (str): camera name to use for rendering
        """
        if mode == "human":
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            self.env.viewer.set_camera(cam_id)
            return self.env.render()
        elif mode == "rgb_array":
            return self.env.sim.render(height=height, width=width, camera_name=camera_name)[::-1]
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    def get_observation(self, di=None):
        """
        Get current environment observation dictionary.

        Args:
            di (dict): current raw observation dictionary from robosuite to wrap and provide 
                as a dictionary. If not provided, will be queried from robosuite.
        """
        if di is None:
            di = self.env._get_observations(force_update=True) if self._is_v1 else self.env._get_observation()
        ret = {}
        for k in di:
            if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb"):
                ret[k] = di[k][::-1]
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)
            if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="depth"):
                depth_map = di[k][::-1]
                depth_map = np.clip(depth_map, 0, 1)
                ret[k] = get_real_depth_map(self.env.sim, depth_map)
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)

        # "object" key contains object information
        ret["object"] = np.array(di["object-state"])

        if self.env.use_camera_obs:
            workspace = self.voxel_workspace

            voxel_bound = workspace.T
            voxel_size = 64

            all_pcds = o3d.geometry.PointCloud()
            for cam_idx, camera_name in enumerate(self.env.camera_names):
                cam_height = self.env.camera_heights[cam_idx]
                cam_width = self.env.camera_widths[cam_idx]
                ext_mat = get_camera_extrinsic_matrix(self.env.sim, camera_name)
                int_mat = get_camera_intrinsic_matrix(self.env.sim, camera_name, cam_height, cam_width)
                depth = di[f'{camera_name}_depth'][::-1]
                depth = np.clip(depth, 0, 1)
                depth = get_real_depth_map(self.env.sim, depth)
                depth = depth[:, :, 0]
                segmentations = get_camera_segmentation(self.env.sim, camera_name, cam_height, cam_width)[:, :, -1]

                color = di[f'{camera_name}_image'][::-1]

                cam_param = [int_mat[0, 0], int_mat[1, 1], int_mat[0, 2], int_mat[1, 2]]
                mask = np.ones_like(depth, dtype=bool)
                pcd = depth2fgpcd(depth, mask, cam_param)
                action_mask = np.where(segmentations >= 93, 1, 0)
                action_pcd = depth2fgpcd(depth, action_mask, cam_param)
                anchor_mask = np.where(segmentations == 14, 1, 0)
                anchor_pcd = depth2fgpcd(depth, anchor_mask, cam_param)
                gripper_mask_cond = np.logical_and(segmentations > 14, segmentations < 93)
                gripper_mask = np.where(gripper_mask_cond, 1, 0)
                gripper_pcd = depth2fgpcd(depth, gripper_mask, cam_param)

                pose = ext_mat
                
                trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
                trans_pcd = trans_pcd[:3, :].T
                action_trans_pcd = pose @ np.concatenate([action_pcd.T, np.ones((1, action_pcd.shape[0]))], axis=0)
                action_trans_pcd = action_trans_pcd[:3, :].T
                anchor_trans_pcd = pose @ np.concatenate([anchor_pcd.T, np.ones((1, anchor_pcd.shape[0]))], axis=0)
                anchor_trans_pcd = anchor_trans_pcd[:3, :].T
                gripper_trans_pcd = pose @ np.concatenate([gripper_pcd.T, np.ones((1, gripper_pcd.shape[0]))], axis=0)
                gripper_trans_pcd = gripper_trans_pcd[:3, :].T

                mask = (trans_pcd[:, 0] > workspace[0, 0]) * (trans_pcd[:, 0] < workspace[0, 1]) * (trans_pcd[:, 1] > workspace[1, 0]) * (trans_pcd[:, 1] < workspace[1, 1]) * (trans_pcd[:, 2] > workspace[2, 0]) * (trans_pcd[:, 2] < workspace[2, 1])
                action_mask = (action_trans_pcd[:, 0] > workspace[0, 0]) * (action_trans_pcd[:, 0] < workspace[0, 1]) * (action_trans_pcd[:, 1] > workspace[1, 0]) * (action_trans_pcd[:, 1] < workspace[1, 1]) * (action_trans_pcd[:, 2] > workspace[2, 0]) * (action_trans_pcd[:, 2] < workspace[2, 1])
                anchor_mask = (anchor_trans_pcd[:, 0] > workspace[0, 0]) * (anchor_trans_pcd[:, 0] < workspace[0, 1]) * (anchor_trans_pcd[:, 1] > workspace[1, 0]) * (anchor_trans_pcd[:, 1] < workspace[1, 1]) * (anchor_trans_pcd[:, 2] > workspace[2, 0]) * (anchor_trans_pcd[:, 2] < workspace[2, 1])
                gripper_mask = (gripper_trans_pcd[:, 0] > workspace[0, 0]) * (gripper_trans_pcd[:, 0] < workspace[0, 1]) * (gripper_trans_pcd[:, 1] > workspace[1, 0]) * (gripper_trans_pcd[:, 1] < workspace[1, 1]) * (gripper_trans_pcd[:, 2] > workspace[2, 0]) * (gripper_trans_pcd[:, 2] < workspace[2, 1])

                pcd_o3d = np2o3d(trans_pcd[mask], color.reshape(-1, 3)[mask].astype(np.float64) / 255)
                
                action_pcd = action_trans_pcd[action_mask]
                action_onehot = np.zeros_like(action_pcd)
                action_onehot[:, 0] = 1

                anchor_pcd = anchor_trans_pcd[anchor_mask]
                anchor_onehot = np.zeros_like(anchor_pcd)
                anchor_onehot[:, 1] = 1

                gripper_pcd = gripper_trans_pcd[gripper_mask]
                gripper_onehot = np.zeros_like(gripper_pcd)
                gripper_onehot[:, 2] = 1

                action_pcd_o3d = np2o3d(action_pcd, action_onehot)
                anchor_pcd_o3d = np2o3d(anchor_pcd, anchor_onehot)
                gripper_pcd_o3d = np2o3d(gripper_pcd, gripper_onehot)

                all_pcds += action_pcd_o3d
                all_pcds += anchor_pcd_o3d
                all_pcds += gripper_pcd_o3d

            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(all_pcds, voxel_size=self.ws_size/voxel_size+1e-4, min_bound=voxel_bound[0], max_bound=voxel_bound[1])
            voxels = voxel_grid.get_voxels()  # returns list of voxels
            if len(voxels) == 0:
                np_voxels = np.zeros([4, voxel_size, voxel_size, voxel_size], dtype=np.uint8)
            else:
                indices = np.stack(list(vx.grid_index for vx in voxels))
                colors = np.stack(list(vx.color for vx in voxels))

                mask = (indices > 0) * (indices < voxel_size)
                indices = indices[mask.all(axis=1)]
                colors = colors[mask.all(axis=1)]

                np_voxels = np.zeros([4, voxel_size, voxel_size, voxel_size], dtype=np.uint8)
                np_voxels[0, indices[:, 0], indices[:, 1], indices[:, 2]] = 1
                np_voxels[1:, indices[:, 0], indices[:, 1], indices[:, 2]] = colors.T * 255

            ret['voxels'] = np_voxels

            bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.pc_workspace.T[0], self.pc_workspace.T[1])
            cropped_pcd = all_pcds.crop(bounding_box)
            if len(cropped_pcd.points) == 0:
                # create fake points
                cropped_pcd.points = o3d.utility.Vector3dVector(np.array([[0., 0., 0.]]))
                cropped_pcd.colors = o3d.utility.Vector3dVector(np.array([[0., 0., 0.]]))
            if len(cropped_pcd.points) < 1024:
                # random upsample to 1024
                num_pad = 1024 - len(cropped_pcd.points)
                indices = np.random.choice(len(cropped_pcd.points), num_pad)
                padded_xyz = np.asarray(cropped_pcd.points)[indices]
                padded_color = np.asarray(cropped_pcd.colors)[indices]
                xyz = np.concatenate([np.asarray(cropped_pcd.points), padded_xyz], 0)
                color = np.concatenate([np.asarray(cropped_pcd.colors), padded_color], 0)
                cropped_pcd = o3d.geometry.PointCloud()
                cropped_pcd.points = o3d.utility.Vector3dVector(xyz)
                cropped_pcd.colors = o3d.utility.Vector3dVector(color)
            sampled_pcds = cropped_pcd.farthest_point_down_sample(1024)
            xyz = np.asarray(sampled_pcds.points)
            color = np.asarray(sampled_pcds.colors)


            ret['point_cloud'] = np.concatenate([xyz, color], 1)
            # ret['gripper_mesh_origin'] = self.robot_eef_pointcloud
            # import pickle
            # print(ret['gripper_mesh_origin'].shape)
            # with open('debug_pointcloud_eef.pkl', 'wb') as f:
            #     pickle.dump(ret['gripper_mesh_origin'], f)
            # exit()

        if self._is_v1:
            for robot in self.env.robots:
                # add all robot-arm-specific observations. Note the (k not in ret) check
                # ensures that we don't accidentally add robot wrist images a second time
                pf = robot.robot_model.naming_prefix
                for k in di:
                    if k.startswith(pf) and (k not in ret) and \
                            (not k.endswith("proprio-state")):
                        ret[k] = np.array(di[k])
        else:
            # minimal proprioception for older versions of robosuite
            ret["proprio"] = np.array(di["robot-state"])
            ret["eef_pos"] = np.array(di["eef_pos"])
            ret["eef_quat"] = np.array(di["eef_quat"])
            ret["gripper_qpos"] = np.array(di["gripper_qpos"])
        
        # geom_rot = self.compute_robot_eef_pointcloud()
        # adding gripper mesh pcd
        if self.env.use_camera_obs:
            n_batch = ret['point_cloud'].shape[0]
            eef_pos = torch.tensor(ret['robot0_eef_pos']).cuda()
            eef_quat = torch.tensor(ret['robot0_eef_quat']).cuda()
            eef_quat = torch.cat([eef_quat[..., 3].unsqueeze(-1), eef_quat[..., 0].unsqueeze(-1), eef_quat[..., 1].unsqueeze(-1), eef_quat[..., 2].unsqueeze(-1)], dim=-1)
            eef_rot = quaternion_to_matrix(eef_quat).float()

            tmp1 = (self.robot_eef_pointcloud.cuda() - self.reference_pos).float()
            tmp2 = (eef_rot @ self.reference_rot.T @ tmp1.T).T
            global_vertices = tmp2 + eef_pos

            eef_one_hot = torch.zeros_like(global_vertices)
            eef_pcd = torch.cat([global_vertices, eef_one_hot], dim=-1).cpu().numpy()
            ret['point_cloud'] = np.concatenate([ret['point_cloud'], eef_pcd], axis=-2)
        
            # adding goal gripper pcd
            if self.gt_goal_gripper_pcd is not None:
                # update current goal gripper idx based on stage
                current_goal = self.gt_goal_gripper_pcd[self.cur_goal_gripper_idx]
                if np.linalg.norm(eef_pcd[..., :3] - current_goal[..., :3]) <= 0.1 and \
                        self.cur_goal_gripper_idx < len(self.gt_goal_gripper_pcd) - 1:
                    self.cur_goal_gripper_idx += 1
                ret['goal_gripper_pcd'] = self.gt_goal_gripper_pcd[self.cur_goal_gripper_idx]
                if self.gt_goal_eef_pos is not None:
                    ret['goal_eef_pos'] = self.gt_goal_eef_pos[self.cur_goal_gripper_idx]
                    ret['goal_eef_quat'] = self.gt_goal_eef_quat[self.cur_goal_gripper_idx]
                # ret['goal_gripper_pcd'] = self.gt_goal_gripper_pcd[0]
                # ret['goal_gripper_pcd'] = np.concatenate([self.gt_goal_gripper_pcd[0], self.gt_goal_gripper_pcd[2]], axis=0)
        # self.get_state()
        return ret

    def get_robot_eef_pointcloud(self):
        return self.robot_eef_pointcloud
    
    def compute_robot_eef_pointcloud(self):
        # This is a test function for obtaining reference point cloud and pose information of the end effector.

        point_cloud = []
        for geom_id in range(self.env.sim.model.ngeom):
            if self.env.sim.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
                # Get the mesh ID associated with this geometry
                mesh_id = self.env.sim.model.geom_dataid[geom_id]
                name_start = self.env.sim.model.name_meshadr[mesh_id]
                mesh_name = self.env.sim.model.names[name_start:].decode('utf-8').split('\x00', 1)[0]
                if mesh_name not in ['gripper0_hand', 'gripper0_finger']:
                    continue
                # Get the address and number of vertices
                vert_start = self.env.sim.model.mesh_vertadr[mesh_id]
                vert_count = self.env.sim.model.mesh_vertnum[mesh_id]

                # Extract vertices in the local coordinate frame
                local_vertices = self.env.sim.model.mesh_vert[vert_start:vert_start + vert_count].reshape(-1, 3)

                geom_pos = self.env.sim.data.geom_xpos[geom_id]
                geom_rot = self.env.sim.data.geom_xmat[geom_id].reshape(3, 3)

                if mesh_name == 'gripper0_hand':
                    import robosuite.utils.transform_utils as T
                    eef_body_rot = T.convert_quat(self.env.sim.data.get_body_xquat('robot0_right_hand'), to="xyzw")
                    eef_body_xmat = self.env.sim.data.get_body_xmat('robot0_right_hand').reshape(3, 3)
                    eef_body_xpos = self.env.sim.data.site_xpos[2]  # self.env.sim.data.get_body_xpos('robot0_right_hand')

                # Transform local vertices to the global frame
                global_vertices = (geom_rot @ local_vertices.T).T + geom_pos

                # print("eef body rot", eef_body_rot)
                # print("eef body xmat", eef_body_xmat)
                # print("eef body xpos", eef_body_xpos)
                
                # print("geom pos", geom_pos)
                # print("geom rot", geom_rot)

                point_cloud.append(global_vertices)
        point_cloud = np.vstack(point_cloud)

        import pickle
        with open('eef_pointcloud_wiz_pose_info.pkl', 'wb') as f:
            pickle.dump({'pointcloud': point_cloud, 'quat': eef_body_rot, 'pos': eef_body_xpos}, f)
        exit()
        self.robot_eef_pointcloud = torch.tensor(point_cloud)
        return torch.tensor(geom_rot)

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        xml = self.env.sim.model.get_xml() # model xml file
        state = np.array(self.env.sim.get_state().flatten()) # simulator state
        return dict(model=xml, states=state)

    def get_reward(self):
        """
        Get current reward.
        """
        return self.env.reward()

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        return self.get_observation(self.env._get_goal())

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        return self.env.set_goal(**kwargs)

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """

        # Robosuite envs always rollout to fixed horizon.
        return False

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        succ = self.env._check_success()
        if isinstance(succ, dict):
            assert "task" in succ
            return succ
        return { "task" : succ }

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return self.env.action_spec[0].shape[0]

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.ROBOSUITE_TYPE

    @property
    def version(self):
        """
        Returns version of robosuite used for this environment, eg. 1.2.0
        """
        return robosuite.__version__

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(
            env_name=self.name,
            env_version=self.version,
            type=self.type,
            env_kwargs=deepcopy(self._init_kwargs)
        )

    @classmethod
    def create_for_data_processing(
        cls, 
        env_name, 
        camera_names, 
        camera_height, 
        camera_width, 
        reward_shaping, 
        **kwargs,
    ):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. 

        Args:
            env_name (str): name of environment
            camera_names (list of str): list of camera names that correspond to image observations
            camera_height (int): camera height for all cameras
            camera_width (int): camera width for all cameras
            reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
        """
        is_v1 = (robosuite.__version__.split(".")[0] == "1")
        has_camera = (len(camera_names) > 0)

        new_kwargs = {
            "reward_shaping": reward_shaping,
        }

        if has_camera:
            if is_v1:
                new_kwargs["camera_names"] = list(camera_names)
                new_kwargs["camera_heights"] = camera_height
                new_kwargs["camera_widths"] = camera_width
            else:
                assert len(camera_names) == 1
                if has_camera:
                    new_kwargs["camera_name"] = camera_names[0]
                    new_kwargs["camera_height"] = camera_height
                    new_kwargs["camera_width"] = camera_width

        kwargs.update(new_kwargs)

        # also initialize obs utils so it knows which modalities are image modalities
        image_modalities = list(camera_names)
        if is_v1:
            image_modalities = ["{}_image".format(cn) for cn in camera_names]
            depth_modalities = ["{}_depth".format(cn) for cn in camera_names]
        elif has_camera:
            # v0.3 only had support for one image, and it was named "rgb"
            assert len(image_modalities) == 1
            image_modalities = ["rgb"]
        obs_modality_specs = {
            "obs": {
                "low_dim": [], # technically unused, so we don't have to specify all of them
                "rgb": image_modalities,
                "depth": depth_modalities,
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        # note that @postprocess_visual_obs is False since this env's images will be written to a dataset
        return cls(
            env_name=env_name,
            render=False, 
            render_offscreen=has_camera, 
            use_image_obs=has_camera, 
            postprocess_visual_obs=False,
            **kwargs,
        )

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return tuple(MUJOCO_EXCEPTIONS)

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)
