# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py

import os.path as osp

import torch
import torch.nn.functional as F
import numpy as np

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
from torch_geometric.typing import WITH_TORCH_CLUSTER
# import torch_cluster

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")


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


class PointNet2_Cls(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([6, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, 1], dropout=0.5, norm=None)

    def forward(self, data):
        # preprocess data
        assert len(data['scene'].shape) == 3
        scene = data['scene']
        pc = data['pc']

        scene_onehot = torch.zeros_like(scene).to(scene.device)
        scene_onehot[..., 0] = 1
        pc_onehot = torch.zeros_like(pc).to(pc.device)
        pc_onehot[..., 1] = 1

        scene = torch.cat([scene, scene_onehot], dim=-1)
        pc = torch.cat([pc, pc_onehot], dim=-1)
        points_used = torch.cat([scene, pc], dim=1)

        n_batch = points_used.shape[0]
        n_points = points_used.shape[1]

        x = points_used.reshape(-1, points_used.shape[-1])[:, 3:]
        pos = points_used.reshape(-1, points_used.shape[-1])[:, :3]

        batch = np.tile(np.arange(n_batch)[:, np.newaxis], (1, n_points)).reshape(-1)
        batch = torch.tensor(batch).cuda()

        sa0_out = (x.float(), pos.float(), batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        x = self.mlp(x)
        x = F.sigmoid(x)
        return x  # self.mlp(x).log_softmax(dim=-1)


def train(epoch):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__':
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..',
                    'data/ModelNet10')
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    test_dataset = ModelNet(path, '10', False, transform, pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 201):
        train(epoch)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Test: {test_acc:.4f}')