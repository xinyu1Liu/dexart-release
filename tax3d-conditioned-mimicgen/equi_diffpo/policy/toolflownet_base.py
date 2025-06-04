from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import pytorch3d.ops as torch3d_ops
from pytorch3d.transforms import (
    Transform3d, Rotate, rotation_6d_to_matrix, axis_angle_to_matrix,
    matrix_to_quaternion, matrix_to_euler_angles, quaternion_to_axis_angle,
    quaternion_to_matrix
)
import pickle

from equi_diffpo.model.common.module_attr_mixin import ModuleAttrMixin
from equi_diffpo.model.common.normalizer import LinearNormalizer
from equi_diffpo.model.diffusion.dp3_conditional_unet1d import ConditionalUnet1D
from equi_diffpo.model.diffusion.mask_generator import LowdimMaskGenerator
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.model.vision.pointnet2_segmentation import PointNet2_Segm
from equi_diffpo.model.common.rotation_transformer import RotationTransformer


def align_point_clouds(P, Q):
    """
    Align two point clouds using SVD to compute the rotation matrix and translation vector.

    Parameters:
    P (torch.tensor): Reference point cloud of shape (B, N, 3).
    Q (torch.tensor): Transformed point cloud of shape (B, N, 3).

    Returns:
    R (torch.tensor): Rotation matrix of shape (B, 3, 3).
    T (torch.tensor): Translation vector of shape (B, 3,).
    """
    # Step 1: Compute centroids
    centroid_P = torch.mean(P, dim=1, keepdims=True)
    centroid_Q = torch.mean(Q, dim=1, keepdims=True)
    
    # Step 2: Center the point clouds
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # Step 3: Compute the covariance matrix
    H = P_centered.transpose(1, 2) @ Q_centered
    
    # Step 4: Perform SVD
    U, S, Vt = torch.linalg.svd(H)
    
    # Step 5: Compute the rotation matrix
    R = Vt.transpose(1, 2) @ U.transpose(1, 2)
    
    # Handle reflection case (ensure a proper rotation matrix with det(R) = 1)
    for i in range(len(R)):
        if torch.linalg.det(R[i]) < 0:
            Vt[i, -1, :] *= -1
            R[i] = Vt[i].T @ U[i].T
    
    # Step 6: Compute the translation vector
    T = centroid_Q - (R @ centroid_P.transpose(1, 2)).transpose(1, 2)
    
    return R, T


def symmetric_orthogonalization(M):
    """Maps arbitrary input matrices onto SO(3) via symmetric orthogonalization.
    (modified from https://github.com/amakadia/svd_for_pose)

    M: should have size [batch_size, 3, 3]

    Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    U, _, Vh = torch.linalg.svd(M)
    det = torch.det(torch.bmm(U, Vh)).view(-1, 1, 1)
    Vh = torch.cat((Vh[:, :2, :], Vh[:, -1:, :] * det), 1)
    R = U @ Vh
    return R


def flow2pose(xyz, flow, weights=None, return_transform3d=False,
        return_quaternions=False, world_frameify=True):
    """Flow2Pose via SVD.

    Operates on minibatches of `B` point clouds, each with `N` points. Assumes
    all point clouds have `N` points, but in practice we only call this with
    minibatch size 1 and we get rid of non-tool points before calling this.

    Outputs a rotation with the origin at the tool centroid. This is not the
    origin of the frame where actions are expressed, which is at the tool tip.

    Parameters
    ----------
    xyz: point clouds of shape (B,N,3). This gets zero-centered so it's OK if it
        is not already centered.
    flow: corresponding flow of shape (B,N,3). As with xyz, it gets zero-centered.
    weights: weights for the N points, set to None for uniform weighting, for now
        I don't think we want to weigh certain points more than others, and it
        could be tricky when points can technically be any order in a PCL.
    return_transform3d: Used if we want to return a transform, for which we apply
        on a set of point clouds. This is what Brian/Chuer use to compute losses,
        by applying this on original points and comparing point-wise MSEs.
    return_quaternions: Use if we want to convert rotation matrices to quaternions.
        Uses format of (wxyz) format, so the identity quanternion is (1,0,0,0).
    world_frameify: Use if we want to correct the translation vector so that the
        transformation is expressed w.r.t. the world frame.
    """
    if weights is None:
        weights = torch.ones(xyz.shape[:-1], device=xyz.device)
    ww = (weights / weights.sum(dim=-1, keepdims=True)).unsqueeze(-1)

    # xyz_mean shape: ((B,N,1), (B,N,3)) mult -> (B,N,3) -> sum -> (B,1,3)
    xyz_mean = (ww * xyz).sum(dim=1, keepdims=True)
    xyz_demean = xyz - xyz_mean  # broadcast `xyz_mean`, still shape (B,N,3)

    # As with xyz positions, find (weighted) mean of flow, shape (B,1,3).
    flow_mean = (ww * flow).sum(dim=1, keepdims=True)

    # Zero-mean positions plus zero-mean flow to find new points.
    xyz_trans = xyz_demean + flow - flow_mean  # (B,N,3)

    # Batch matrix-multiply, get X: (B,3,3), each (3x3) matrix is in SO(3).
    X = torch.bmm(xyz_demean.transpose(-2,-1),  # (B,3,N)
                  ww * xyz_trans)               # (B,N,3)

    # Rotation matrix in SO(3) for each mb item, (B,3,3).
    R = symmetric_orthogonalization(X)

    # 3D translation vector for eacb mb item, (B,3) due to squeezing.
    if world_frameify:
        t = (flow_mean + xyz_mean - torch.bmm(xyz_mean, R)).squeeze(1)
    else:
        t = flow_mean.squeeze(1)

    if return_transform3d:
        return Rotate(R).translate(t)
    if return_quaternions:
        quats = matrix_to_quaternion(matrix=R)
        return quats, t
    return R, t


class BasePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
    
class ToolFlowNetBase(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])


        # obs_encoder = DP3Encoder(observation_space=obs_dict,
        #                                            img_crop_shape=crop_shape,
        #                                         out_channel=encoder_output_dim,
        #                                         pointcloud_encoder_cfg=pointcloud_encoder_cfg,
        #                                         use_pc_color=use_pc_color,
        #                                         pointnet_type=pointnet_type,
        #                                         )
        self.use_pc_color = use_pc_color
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        
        if not self.use_pc_color:
            self.in_dim = 3
        else:
            self.in_dim = 6

        obs_encoder = PointNet2_Segm(
                in_dim=self.in_dim,
                flow_dim=3*horizon,  # We are here predicting 16 steps of action flow
                encoder_type='pointnet',
                scale_pcl_val=None,
                separate_MLPs_R_t=False,
        )
        obs_feature_dim = obs_encoder.output_shape()

        self.obs_encoder = obs_encoder
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # loading gripper mesh
        with open('eef_pointcloud_wiz_pose_info.pkl', 'rb') as f:
            reference_eef_data = pickle.load(f)
        self.reference_eef_data = reference_eef_data

        self.robot_eef_pointcloud = torch.tensor(self.reference_eef_data['pointcloud']).cuda()
        reference_quat = self.reference_eef_data['quat']
        reference_quat = torch.tensor([reference_quat[3], reference_quat[0], reference_quat[1], reference_quat[2]]).cuda()  # xyzw -> wxyz
        self.reference_rot = quaternion_to_matrix(reference_quat).float()
        self.reference_pos = torch.tensor(self.reference_eef_data['pos']).cuda()

        # print_params(self)


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        if 'robot0_eye_in_hand_image' in obs_dict:
            del obs_dict['robot0_eye_in_hand_image']
        if 'agentview_image' in obs_dict:
            del obs_dict['agentview_image']
        
        # incorporate gripper mesh point cloud into the whole point cloud.
        if obs_dict['point_cloud'].shape[-2] <= 1024:
            eef_pcd = self.get_gripper_mesh_pointcloud(obs_dict)
            obs_dict['point_cloud'] = torch.cat([obs_dict['point_cloud'], eef_pcd], dim=-2)
        else:
            eef_pcd = obs_dict['point_cloud'][..., 1024:, :]
        
        # normalize input
        nobs = obs_dict  # self.normalizer.normalize(obs_dict)

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        flow_per_pt, finger_output = self.obs_encoder(this_nobs)
        
        # compute svd
        eef_pcd_origin = torch.tile(eef_pcd[..., :3], (1, self.horizon, 1, 1)).reshape(-1, 138, 3)
        flow = flow_per_pt[..., 1024:, :].reshape(-1, 138, 3)
        goal = eef_pcd_origin + flow

        tmp1 = (self.robot_eef_pointcloud.unsqueeze(0) - self.reference_pos).float()
        original_point_cloud = (self.reference_rot.T @ tmp1.transpose(1, 2)).transpose(1, 2)

        abs_R, abs_T = align_point_clouds(torch.tile(original_point_cloud.float(), (goal.shape[0], 1, 1)),
                                  goal.float())
        z_axis_offset = torch.tensor([[0, -1, 0],
                                    [1, 0, 0],
                                    [0, 0, 1]]).float().cuda()
        abs_R = z_axis_offset @ abs_R
        rot = RotationTransformer('matrix', 'rotation_6d').forward(abs_R)
        abs_T = abs_T.reshape(B, -1, 3)
        rot = rot.reshape(B, -1, 6)
        action_pred = torch.cat([abs_T, rot, finger_output.unsqueeze(-1)], dim=-1)

        info = {'total': nobs['point_cloud'].cpu(), 
                'pred': (flow + eef_pcd[..., :3].squeeze(1)).cpu(),
                'flow': flow.cpu(),
                'finger_pred': finger_output,
                'action_pred': action_pred}

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction


        result = {
            'action': action,
            'action_pred': action_pred,
            'info': info,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_gripper_mesh_pointcloud(self, obs):
        n_batch = obs['point_cloud'].shape[0]
        eef_pos = obs['robot0_eef_pos']
        eef_quat = obs['robot0_eef_quat']
        eef_quat = torch.cat([eef_quat[..., 3].unsqueeze(-1), eef_quat[..., 0].unsqueeze(-1), eef_quat[..., 1].unsqueeze(-1), eef_quat[..., 2].unsqueeze(-1)], dim=-1)
        eef_rot = quaternion_to_matrix(eef_quat).float()

        tmp1 = (self.robot_eef_pointcloud.unsqueeze(0).unsqueeze(0).cuda() - self.reference_pos).float()
        tmp2 = (eef_rot @ self.reference_rot.T @ tmp1.transpose(2, 3)).transpose(2, 3)
        global_vertices = tmp2 + eef_pos.unsqueeze(2)

        eef_one_hot = torch.zeros_like(global_vertices)
        eef_pcd = torch.cat([global_vertices, eef_one_hot], dim=-1)

        return eef_pcd

    def get_gripper_goal_pointcloud(self, action):
        # action is in shape [128, 16, 10]
        # [0:3] - pos
        # [3:9] - rotation_6d
        # [9:] - gripper
        z_axis_offset = torch.tensor([[0, 1, 0],
                                    [-1, 0, 0],
                                    [0, 0, 1]]).float().cuda()
        eef_pos = action[..., :3]
        eef_rot = RotationTransformer('rotation_6d', 'matrix').forward(action[..., 3:9])
        tmp1 = (self.robot_eef_pointcloud.unsqueeze(0).unsqueeze(0).cuda() - self.reference_pos).float()
        tmp2 = (z_axis_offset @ eef_rot @ self.reference_rot.T @ tmp1.transpose(2, 3)).transpose(2, 3)
        global_vertices = tmp2 + eef_pos.unsqueeze(2)
        return global_vertices
    
    def compute_loss(self, batch):
        if 'robot0_eye_in_hand_image' in batch['obs']:
            del batch['obs']['robot0_eye_in_hand_image']

        # incorporate gripper mesh point cloud into the whole point cloud.
        if batch['obs']['point_cloud'].shape[-2] <= 1024:
            eef_pcd = self.get_gripper_mesh_pointcloud(batch['obs'])
            batch['obs']['point_cloud'] = torch.cat([batch['obs']['point_cloud'], eef_pcd], dim=-2)
        else:
            eef_pcd = batch['obs']['point_cloud'][..., 1024:, :]

        # with open('debug_pointcloud_toolflownet.pkl', 'wb') as f:
        #     pickle.dump(batch['obs']['point_cloud'].cpu(), f)
        # exit()

        # normalize input
        nobs = batch['obs']  # self.normalizer.normalize(batch['obs'])
        nactions = batch['action']  # self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))

        flow_per_pt, finger_output = self.obs_encoder(this_nobs)  # should be in shape [128*1162, 1, 48]
        pred = flow_per_pt[:, :, 1024:, :]  # should be in shape [128, 16, 1162-1024, 3]
        
        # compute target flow
        goal_eef_pcd = self.get_gripper_goal_pointcloud(nactions)
        target = goal_eef_pcd - eef_pcd[..., :3]

        # with open('debug_pointcloud_toolflownet.pkl', 'wb') as f:
        #     pickle.dump({'total': batch['obs']['point_cloud'].cpu(), 
        #                  'goal': goal_eef_pcd.cpu(), 
        #                  'action': nactions.cpu(),
        #                  'pred': (pred + eef_pcd[..., :3]).cpu()}, f)
        info = {'total': batch['obs']['point_cloud'].cpu(), 
                'goal': goal_eef_pcd.cpu(), 
                'action': nactions.cpu(),
                'pred': (pred + eef_pcd[..., :3]).cpu(),
                'finger_pred': finger_output}

        loss = F.mse_loss(pred, target, reduction='none')
        finger_loss = F.mse_loss(finger_output, nactions[..., -1], reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean() + finger_loss.mean()
        

        loss_dict = {
                'bc_loss': loss.item(),
            }

        # print(f"t2-t1: {t2-t1:.3f}")
        # print(f"t3-t2: {t3-t2:.3f}")
        # print(f"t4-t3: {t4-t3:.3f}")
        # print(f"t5-t4: {t5-t4:.3f}")
        # print(f"t6-t5: {t6-t5:.3f}")
        
        return loss, loss_dict, info