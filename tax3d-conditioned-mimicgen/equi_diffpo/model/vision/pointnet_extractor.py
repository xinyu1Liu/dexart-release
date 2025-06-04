import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint

from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import einops
from equi_diffpo.model.vision.layers import RelativeCrossAttentionModule, RotaryPositionEncoding3D

NUM_SCENE_PCD = 1024

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d) or isinstance(x, nn.BatchNorm1d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


class Act3dEncoder(nn.Module):
    def __init__(self, 
                 in_channels=6, 
                 encoder_output_dim=60, 
                 num_gripper_points=4, 
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 observation_space=None,
                 goal_mode='None',
                 mode=None,
                 use_mlp=False,
                 self_attention=False,
                 use_attn_for_point_features=False,
                 pointcloud_backbone='mlp',
                 use_lightweight_unet=False,
                 final_attention=False,
                 attention_num_heads=3,
                 attention_num_layers=2,
                 use_repr_10d=False,
                 **kwargs
                 ):
        super(Act3dEncoder, self).__init__()
        hidden_layer_dim = encoder_output_dim
        self.goal_mode = goal_mode
        vision_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, encoder_output_dim)
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)

        attn_layers = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
        attn_layers = replace_bn_with_gn(attn_layers)

        self.nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'relative_pe_layer': RotaryPositionEncoding3D(encoder_output_dim),
            'attn_layers': attn_layers,
        })

        position_embedding_mlp = nn.Sequential(
            nn.Linear(9, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, encoder_output_dim // 3),
        )
        
        self.nets['gripper_pcd_position_embedding_mlp'] = position_embedding_mlp
        self.nets['embed'] = nn.Embedding(1, encoder_output_dim // 3 * 2)

        goal_attn_layers = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
        goal_attn_layers = replace_bn_with_gn(goal_attn_layers)
        self.nets['goal_attn_layers'] = goal_attn_layers
        self.nets['goal_pcd_position_embedding_mlp'] = copy.deepcopy(position_embedding_mlp)
        self.nets['goal_embed'] = nn.Embedding(1, encoder_output_dim // 3 * 2)

    def forward(self, x):
        # x shape: [B, hor, N, input_dim]

        if self.goal_mode == 'None':
            x[..., NUM_SCENE_PCD+138:, :] = 0

        # scene point cloud
        chosen_four_point_idx = torch.tensor([30, 40, 110, 135])
        point_cloud = x[..., :NUM_SCENE_PCD, :]

        B, N, C = point_cloud.shape
        point_cloud_flatten = point_cloud.reshape(-1, C)
        point_cloud_features_flatten = self.nets['vision_encoder'](point_cloud_flatten)
        point_cloud_features = point_cloud_features_flatten.reshape(B, N, -1)
        point_cloud_features = einops.rearrange(point_cloud_features, "B N encoder_output_dim -> N B encoder_output_dim")  # N B encoder_output_dim
        point_cloud_rel_pos_embedding = self.nets['relative_pe_layer'](point_cloud)  # B N encoder_output_dim

        # attention between gripper pcd and scene pcd
        gripper_pcd = x[..., NUM_SCENE_PCD + chosen_four_point_idx, :]
        gripper_pcd_rel_pos_embedding = self.nets['relative_pe_layer'](gripper_pcd)  # B num_gripper_points encoder_output_dim
        gripper_pcd_features = self.nets['embed'].weight.unsqueeze(0).repeat(4, B, 1)  # num_gripper_points B encoder_output_dim

        displacement_to_goal = x[..., NUM_SCENE_PCD + 138 + chosen_four_point_idx, :3] - x[..., NUM_SCENE_PCD + chosen_four_point_idx, :3]
        input_to_position_embedding = torch.cat([gripper_pcd, displacement_to_goal], dim=-1)  # B num_gripper_points 9
        gripper_pcd_position_embedding = self.nets['gripper_pcd_position_embedding_mlp'](input_to_position_embedding)
        gripper_pcd_position_embedding = einops.rearrange(gripper_pcd_position_embedding, "B N encoder_output_dim -> N B encoder_output_dim")  # N B encoder_output_dim

        gripper_pcd_features = torch.cat([gripper_pcd_features, gripper_pcd_position_embedding], dim=-1)

        attn_output = self.nets['attn_layers'](
            query=gripper_pcd_features, value=point_cloud_features,
            query_pos=gripper_pcd_rel_pos_embedding, value_pos=point_cloud_rel_pos_embedding,
        )[-1]  # N B encoder_output_dim

        # goal gripper
        goal_gripper_pcd = x[..., NUM_SCENE_PCD + 138 + chosen_four_point_idx, :]
        goal_gripper_pcd_rel_pos_embedding = self.nets['relative_pe_layer'](goal_gripper_pcd)
        goal_gripper_pcd_features = self.nets['goal_embed'].weight.unsqueeze(0).repeat(4, B, 1)

        displacement_to_goal = goal_gripper_pcd[..., :3] - gripper_pcd[..., :3]
        input_to_position_embedding = torch.cat([goal_gripper_pcd, displacement_to_goal], dim=-1)
        goal_gripper_pcd_position_embedding = self.nets['goal_pcd_position_embedding_mlp'](input_to_position_embedding)
        goal_gripper_pcd_position_embedding = einops.rearrange(goal_gripper_pcd_position_embedding, "B N encoder_output_dim -> N B encoder_output_dim")  # N B encoder_output_dim

        goal_gripper_pcd_features = torch.cat([goal_gripper_pcd_features, goal_gripper_pcd_position_embedding], dim=-1)

        goal_attn_output = self.nets['goal_attn_layers'](query=gripper_pcd_features, value=goal_gripper_pcd_features,
                    query_pos=gripper_pcd_rel_pos_embedding, value_pos=goal_gripper_pcd_rel_pos_embedding,
                )[-1]

        obs_feature = torch.cat([attn_output, goal_attn_output], dim=-1)
        obs_feature = einops.rearrange(obs_feature, "N B encoder_output_dim -> B N encoder_output_dim")

        return obs_feature.flatten(start_dim=1)


class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()

    


class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 state_keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
                 goal_mode='None',
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_keys = state_keys
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_size = sum([observation_space[key][0] for key in self.state_keys])
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_size}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        elif pointnet_type == "act3d":
            self.extractor = Act3dEncoder(goal_mode=goal_mode)
            self.n_output_channels = 480
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_size, output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        
        # re-write one-hot vector for point cloud
        points = points[..., :NUM_SCENE_PCD+138*2, :]

        # goal version 1 -- goal gripper pcd

        # probably set NUM_SCENE_PCD = 512 here? And four fingers seperately by 96? CONFIRM

        points[..., :NUM_SCENE_PCD, 3:] = torch.tensor([1, 0, 0])
        points[..., NUM_SCENE_PCD:NUM_SCENE_PCD+138, 3:] = torch.tensor([0, 1, 0])
        points[..., NUM_SCENE_PCD+138:, 3:] = torch.tensor([0, 0, 1])

        # # goal version 2 -- goal gripper flow
        # points[..., :1024, 3:] = torch.tensor([0, 0, 0])
        # points[..., 1024:1162, 3:] = points[..., 1162:, :3] - points[..., 1024:1162, :3]
        # points_used = points[..., :1162, :]

        # # goal version 3 -- truncated gripper flow
        # # max flow ~ 1.5, min flow ~ 0, mean flow ~ 0.45
        # threshold = 0.2
        # points[..., :1024, 3:] = torch.tensor([0, 0, 0])imagined_robot_
        # flow = points[..., 1162:, :3] - points[..., 1024:1162, :3]
        # flow_norm = torch.norm(flow, p=2, dim=-1, keepdim=True)
        # scaling_factor = torch.minimum(threshold / flow_norm, torch.ones_like(flow_norm))
        # flow_truncated = flow * scaling_factor
        # points[..., 1024:1162, 3:] = flow_truncated
        # points_used = points[..., :1162, :]

        # import pickle
        # with open('debug_pointcloud.pkl', 'wb') as f:
        #     pickle.dump(points, f)
        # exit()

        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        pn_feat = self.extractor(points)    # B * out_channel
        
        state = torch.cat([observations[key] for key in self.state_keys], dim=-1)
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels