import torch
import torch.nn.functional as F
from torch import nn
from stable_baselines3.networks.pretrain_nets import PointNet, PointNetMedium, PointNetLarge
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.policies import MlpExtractor
from einops import rearrange, reduce
import sys
sys.path.append('tax3d-conditioned-mimicgen')
from equi_diffpo.model.common.normalizer import LinearNormalizer

class DexArt_IL_Wrapper(nn.Module):
    def __init__(self, extractor_name="smallpn", state_dim=23, state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU, features_dim=320, final_dim=22, device="cuda"):
        super().__init__()
        if extractor_name == "smallpn":
            self.extractor = PointNet()
        elif extractor_name == "mediumpn":
            self.extractor = PointNetMedium()
        elif extractor_name == "largepn":
            self.extractor = PointNetLarge()
        else:
            raise NotImplementedError(f"Extractor {extractor_name} not implemented. Available:\
             smallpn, mediumpn, largepn")

        self.device = device
        self.state_dim = state_dim
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]
        self.state_mlp = nn.Sequential(*create_mlp(self.state_dim, output_dim, net_arch, state_mlp_activation_fn))
        self.state_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']

        self.features_dim = features_dim
        self.net_arch = [{'pi': [64, 64, final_dim], 'vf': [64, 64, final_dim]}]
        self.activation_fn = nn.ReLU
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )
        self.normalizer = LinearNormalizer()

    def compute_loss(self, batch):
        batch['obs']['point_cloud'] = torch.cat([batch['obs']['point_cloud'], batch['obs']['goal_gripper_pcd']], dim=-2)

        obs = batch["obs"]
        obs_clean = {k: v for k, v in obs.items() if k not in ['point_cloud', 'imagin_robot', 'goal_gripper_pcd', 'observed_pc_seg-gt', 'imagined_robot_pc_seg-gt']}
        nobs = self.normalizer.normalize(obs_clean)
        device = next(self.extractor.parameters()).device  # get model device

        nobs["point_cloud"] = obs["point_cloud"]
        nobs["imagin_robot"] = obs["imagin_robot"]
        nobs["goal_gripper_pcd"] = obs["goal_gripper_pcd"]

        if "observed_pc_seg-gt" in obs:
            nobs['observed_pc_seg-gt'] = obs['observed_pc_seg-gt']
            nobs['imagined_robot_pc_seg-gt'] = obs['imagined_robot_pc_seg-gt']

        # Only use current observation, action, state
        pcd = obs["point_cloud"][:,-1,:]
        nactions = self.normalizer['action'].normalize(batch['action']).to(batch['action'].device)
        target = nactions[:,0,:].to(device)
        state = torch.cat([obs[key] for key in self.state_keys], dim=-1)
        state = state.to(pcd.device)[:,-1,:]

        pred = self.forward(pcd,  state)
        loss = F.mse_loss(pred, target)
        info = {}
        loss_dict = {'bc_loss': loss.item()}
        return loss, loss_dict, info

    def forward(self, pcd, state):
        pn_feat = self.extractor(pcd)
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        pred, _ = self.mlp_extractor(final_feat) # [policy, value] - discarding value
        return pred

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def predict_action(self, obs_dict):
        # incorporate gripper mesh point cloud into the whole point cloud.
        obs_dict['point_cloud'] = torch.cat([obs_dict['point_cloud'], obs_dict['goal_gripper_pcd']], dim=-2)
        obs_clean = {k: v for k, v in obs_dict.items() if k not in ['point_cloud', 'imagin_robot', 'goal_gripper_pcd', 'observed_pc_seg-gt', 'imagined_robot_pc_seg-gt']}
        # normalize input
        nobs = self.normalizer.normalize(obs_clean)
        nobs["point_cloud"] = obs_dict["point_cloud"]
        nobs["imagin_robot"] = obs_dict["imagin_robot"]
        nobs["goal_gripper_pcd"] = obs_dict["goal_gripper_pcd"]
        if "observed_pc_seg-gt" in obs_dict:
            nobs['observed_pc_seg-gt'] = obs_dict['observed_pc_seg-gt']
            nobs['imagined_robot_pc_seg-gt'] = obs_dict['imagined_robot_pc_seg-gt']
        this_n_point_cloud = nobs['point_cloud']

        # Only use current observation, action, state
        pcd = obs_dict["point_cloud"][:,-1,:]
        state = torch.cat([obs_dict[key] for key in self.state_keys], dim=-1)
        state = state.to(pcd.device)[:,-1,:]

        # unnormalize prediction
        naction_pred = self.forward(pcd, state)
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        result = {
            'action': action_pred,
            'action_pred': {},
            'info': {},
        }

        return result
