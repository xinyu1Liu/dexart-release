# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py
import torch
import numpy as np
from pytorch3d.transforms import (
    Rotate, matrix_to_quaternion, quaternion_to_axis_angle,
    axis_angle_to_matrix,
)
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius, knn_interpolate
torch.set_printoptions(sci_mode=False, precision=6)


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FPModule(torch.nn.Module):
    """This module is used in segmentation (but not classification) with PointNet++."""

    def __init__(self, k, nn, use_skip=True):
        super().__init__()
        self.k = k
        self.nn = nn
        self.use_skip = use_skip

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if (x_skip is not None) and self.use_skip:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


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


class PointNet2_Segm(torch.nn.Module):
    """PointNet++ architectue for flow prediction.

    Should be the same as the segmentation version, except as with the analogue
    to the classification case, we don't use log softmax at the end. It directly
    predicts flow, returning (N,flow_dim) where typically `flow_dim=3` and N is
    all the points in the minibatch (might not be same # of points per PCL).
    There may be further processing such as differentiable, parameter-less SVD.

    04/18/2022: changing radius hyperparameter to make it consistent w/Regression.
    04/20/2022: can output flow, or compress the flow into a pose.
    05/10/2022: minibatches can have non-equal numbers of points per PCL.
        Requires each data to have the `ptr` to indicate idxs for each PCL.
    05/16/2022: support 6D flow.
    05/21/2022: support dense transformation policy.
    06/02/2022: slight improvement to baseline with clear masks.
    06/05/2022: support removing skip connections. Technically this will use skip
        for the interpolation stage, but it just won't do the torch.cat([x,x_skip])
        which seems like that's more in the spirit of testing this ablation.
    """

    def __init__(self, in_dim, flow_dim=3, encoder_type=None, scale_pcl_val=None,
            separate_MLPs_R_t=False, dense_transform=False,
            remove_skip_connections=False, diff_mode=None):
        super().__init__()
        self.in_dim = in_dim  # 3 for pos, then rest for segmentation
        self.flow_dim = flow_dim
        self.diff_mode = diff_mode
        if self.diff_mode == None:
            self.horizon = int(flow_dim / 3)
        self.encoder_type = encoder_type
        self.scale_pcl_val = scale_pcl_val
        self.separate_MLPs_R_t = separate_MLPs_R_t
        self.state_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
        if 'dense_tf_3D_' in self.encoder_type:
            assert self.flow_dim == 3, self.flow_dim
        if self.separate_MLPs_R_t or ('dense_tf_6D_' in self.encoder_type):
            assert self.flow_dim == 6, self.flow_dim
        self.dense_transform = dense_transform
        self.remove_skip_connections = remove_skip_connections
        self._mask = None

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([in_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        # The `in_dim-3` is because we only want the `data.x` part, not `data.pos`.
        # If removing skip connections, change # of nodes in the first MLP layer.
        if self.remove_skip_connections:
            self.fp3_module = FPModule(1, MLP([1024, 256, 256]), use_skip=False)
            self.fp2_module = FPModule(3, MLP([ 256, 256, 128]), use_skip=False)
            self.fp1_module = FPModule(3, MLP([ 128, 128, 128, 128]), use_skip=False)
        else:
            self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
            self.fp2_module = FPModule(3, MLP([ 256 + 128, 256, 128]))
            self.fp1_module = FPModule(3, MLP([ 128 + (in_dim-3), 128, 128, 128]))

        if self.separate_MLPs_R_t:
            self.mlp_t = MLP([128, 128, 128, 3], dropout=0.5, batch_norm=False)
            self.mlp_R = MLP([128, 128, 128, 3], dropout=0.5, batch_norm=False)
        else:
            self.mlp = MLP([128, 128, 128, self.flow_dim], dropout=0.5, batch_norm=False)

        if self.diff_mode == None:
            state_size = 3 + 4 + 2
            self.state_mlp = MLP([state_size, 64, 64], dropout=0.5, batch_norm=False)
            self.finger_mlp = MLP([128+64, self.horizon], dropout=0.5, batch_norm=False)

    def assign_clear_mask(self, mask):
        """
        For pouring or scooping, in case we have a 6D vector and need to extract
        a transformation from it here (and want to clear out unused components).
        """
        self._mask = mask

    def forward(self, data, info=None, epoch=None, rgt=None):
        """Forward pass, store the flow, potentially do further processing."""

        # Special case if we scaled everything (e.g., meter -> millimeter), need
        # to downscale the observation because otherwise PN++ produces NaNs. :(

        # Here we transform our original data format so that it fits pointnet++
        # Keys in 'data':
        # 'whole_point_cloud' - [256, 1024+102, 6]
        # 'robot0_eef_pos' - [256, 3]
        # 'robot0_eef_quat' - [256, 4]
        # 'robot0_gripper_qpos' - [256, 2]

        # import pickle
        # with open('debug_pointcloud_toolflownet.pkl', 'wb') as f:
        #     pickle.dump(data['point_cloud'].cpu(), f)
        # exit()

        points = data['point_cloud']

        # re-write one-hot vector for point cloud
        # points = points[..., :1300, :]

        # # goal version 1 -- goal gripper pcd
        # points[..., :1024, 3:] = torch.tensor([1, 0, 0])
        # points[..., 1024:1162, 3:] = torch.tensor([0, 1, 0])
        # points[..., 1162:, 3:] = torch.tensor([0, 0, 1])

        # goal version 2 -- goal gripper flow
        points[..., :1024, 3:] = torch.tensor([0, 0, 0])
        points[..., 1024:1162, 3:] = points[..., 1162:, :3] - points[..., 1024:1162, :3]
        points_used = points[..., :1162, :]

        # # goal version 3 -- truncated gripper flow
        # # max flow ~ 1.5, min flow ~ 0, mean flow ~ 0.45
        # threshold = 0.2
        # points[..., :1024, 3:] = torch.tensor([0, 0, 0])
        # flow = points[..., 1162:, :3] - points[..., 1024:1162, :3]
        # flow_norm = torch.norm(flow, p=2, dim=-1, keepdim=True)
        # scaling_factor = torch.minimum(threshold / flow_norm, torch.ones_like(flow_norm))
        # flow_truncated = flow * scaling_factor
        # points[..., 1024:1162, 3:] = flow_truncated
        # points_used = points[..., :1162, :]

        eef_pos = data['robot0_eef_pos']
        eef_quat = data['robot0_eef_quat']
        gripper_qpos = data['robot0_gripper_qpos']

        n_batch = points_used.shape[0]
        n_points = points_used.shape[1]

        x = points_used.reshape(-1, points_used.shape[-1])[:, 3:]
        pos = points_used.reshape(-1, points_used.shape[-1])[:, :3]

        batch = np.tile(np.arange(n_batch)[:, np.newaxis], (1, n_points)).reshape(-1)
        batch = torch.tensor(batch).cuda()

        if self.scale_pcl_val is not None:
            data.pos /= self.scale_pcl_val

        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        if self.diff_mode == None:
            state = torch.cat([data[key] for key in self.state_keys], dim=-1)
            state_feat = self.state_mlp(state)

            finger_input = torch.cat([x.reshape(n_batch, 1162, 128).mean(dim=-2), state_feat], dim=-1)
            finger_output = self.finger_mlp(finger_input)

        # Unlike in segmentation, don't apply a `log_softmax(dim=-1)`.
        if self.separate_MLPs_R_t:
            flow_t = self.mlp_t(x)
            flow_R = self.mlp_R(x)
        else:
            flow_per_pt = self.mlp(x)  # this is per-point, of size (N,flow_dim)
            self.flow_per_pt = flow_per_pt  # use for flow visualizations
            self.flow_per_pt_r = None # only used for 6D flow

        # Only used for 6D flow consistency loss, detect otherwise with `None`
        self.means = None
        self.rot_flows = None

        # Must revert `data.pos` back to the original scale before SVD!!
        if self.scale_pcl_val is not None:
            data.pos *= self.scale_pcl_val

        if self.encoder_type == 'pointnet':
            # Per-point flow, with NON-differentiable averaging or SVD later.
            if self.diff_mode == None:
                return flow_per_pt.reshape(n_batch, self.horizon, -1, 3), finger_output
            else:
                return flow_per_pt
        else:
            raise ValueError(self.encoder_type)

    def output_shape(self):
        return self.flow_dim