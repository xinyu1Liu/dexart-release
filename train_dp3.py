import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import sys
sys.path.append('tax3d-conditioned-mimicgen')
from equi_diffpo.policy.dp3 import DP3
from equi_diffpo.model.common.normalizer import LinearNormalizer
from diffusers.schedulers import DDPMScheduler
import hydra
import collections


def set_random_quaternion():
    quat = torch.randn(4)
    quat = quat / quat.norm()
    return quat


class DP3DexArtDataset(Dataset):
    def __init__(self, data_dir, n_obs_steps=2, n_action_steps=8):
        self.samples = []
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        for fname in os.listdir(data_dir):
            if fname.endswith(".pkl"):
                with open(os.path.join(data_dir, fname), "rb") as f:
                    traj = pickle.load(f)
                T = len(traj)
                max_start = T - (n_obs_steps + n_action_steps) + 1
                for t in range(max_start):
                    self.samples.append((traj, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj, start_idx = self.samples[idx]
        obs_seq = traj[start_idx : start_idx + self.n_obs_steps]
        act_seq = traj[start_idx + self.n_obs_steps : start_idx + self.n_obs_steps + self.n_action_steps]
        

        obs = {
            'point_cloud':         torch.stack([torch.tensor(o["obs"]["observed_point_cloud"], dtype=torch.float32) for o in obs_seq]),
            'goal_gripper_pcd':    torch.stack([torch.tensor(o["obs"]['imagined_robot_point_cloud'], dtype=torch.float32) for o in obs_seq]),
            'robot0_eef_pos':      torch.stack([torch.tensor(o["obs"]['palm_pose.p'], dtype=torch.float32) for o in obs_seq]),
            #'robot0_eef_quat':     torch.stack([torch.tensor(o["obs"]['palm_pose.q'], dtype=torch.float32) for o in obs_seq]),
            'robot0_eef_quat':     torch.stack([set_random_quaternion() for _ in obs_seq]),
            'robot0_gripper_qpos': torch.stack([torch.tensor(o["obs"]['robot_qpos_vec'][-16:], dtype=torch.float32) for o in obs_seq]),
        }

        action = torch.stack([torch.tensor(a["action"], dtype=torch.float32) for a in act_seq])

        return {
            'obs': obs,
            'action': action
        }


def build_normalizer(dataset):
    obs_accum = collections.defaultdict(list)
    action_accum = []

    for sample in dataset:
        obs = sample["obs"]
        action = sample["action"]

        # Exclude point cloud fields from normalization
        obs_clean = {k: v for k, v in obs.items() if k not in ['point_cloud', 'goal_gripper_pcd']}

        for k, v in obs_clean.items():
            # v is (n_obs_steps, dim); flatten across time
            obs_accum[k].append(v.reshape(-1, v.shape[-1]))

        # action is (n_action_steps, dim); flatten across time
        action_accum.append(action.reshape(-1, action.shape[-1]))

    obs_stacked = {k: torch.cat(v_list, dim=0) for k, v_list in obs_accum.items()}
    actions_stacked = torch.cat(action_accum, dim=0)

    normalizer = LinearNormalizer()
    normalizer.fit(obs_stacked)

    action_normalizer = LinearNormalizer()
    action_normalizer.fit({"action": actions_stacked})
    normalizer["action"] = action_normalizer["action"]

    return normalizer


@hydra.main(version_base="1.1", config_path="tax3d-conditioned-mimicgen/equi_diffpo/config", config_name="dp3")
def main(cfg):

    data_dir = "/data/xinyu/demo_DexArt_1036/laptop"
    batch_size = 128
    num_epochs = 100
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DP3DexArtDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)


    shape_meta = {
        'obs': {
            'point_cloud': {'shape': (512, 3)},
            'goal_gripper_pcd': {'shape': (96, 3)},
            'robot0_eef_pos': {'shape': (3,)},
            'robot0_eef_quat': {'shape': (4,)},        
            'robot0_gripper_qpos': {'shape': (16,)}
        },
        'action': {'shape': (22,)}
    }


    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    horizon = 16
    n_action_steps = 8
    n_obs_steps = 2

    pointcloud_encoder_cfg = cfg.policy.get("pointcloud_encoder_cfg", None)

    model = DP3(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        pointcloud_encoder_cfg=pointcloud_encoder_cfg,
        pointnet_type="act3d",
    ).to(device)


    normalizer = build_normalizer(dataset)
    model.set_normalizer(normalizer)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            
            #print("batch shape:")
            #print(batch['obs']['point_cloud'].shape)
            #print(batch['action'].shape)

            obs_batch = batch["obs"]
            action_batch = batch["action"].to(device)

            obs_batch = {k: v.to(device) for k, v in batch["obs"].items()}
            action_batch = batch["action"].to(device)
            #print("obs batch:")
            #print(obs_batch)

            model_input = {"obs": obs_batch, "action": action_batch}
            loss, loss_dict, _ = model.compute_loss(model_input)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}")
        torch.save(model.state_dict(), f"dp3_epoch_{epoch+1}.pt")


if __name__ == "__main__":
    main()