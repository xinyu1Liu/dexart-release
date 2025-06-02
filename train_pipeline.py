import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DemoDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []

        for fname in os.listdir(data_dir):
            if fname.endswith('.pkl'):
                with open(os.path.join(data_dir, fname), 'rb') as f:
                    demo_data = pickle.load(f)
                    for entry in demo_data:
                        obs = entry['obs']
                        action = entry['action']
                        if obs['observed_point_cloud'].shape[0] != 512 or obs['imagined_robot_point_cloud'].shape[0] != 96:
                            continue

                        obs_vec = np.concatenate([
                            obs['robot_qpos_vec'],                   # [22]
                            obs['palm_v'],                           # [3]
                            obs['palm_w'],                           # [3]
                            obs['palm_pose.p'],                      # [3]
                            obs['observed_point_cloud'].flatten(),   # [512*3]
                            obs['imagined_robot_point_cloud'].flatten()  # [96*3]
                        ])
                        self.samples.append((obs_vec, action))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_vec, action = self.samples[idx]
        return torch.tensor(obs_vec, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=22):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def train(data_dir, num_epochs=10):
    dataset = DemoDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = 22 + 3 + 3 + 3 + 512*3 + 96*3
    model = SimpleMLP(input_dim=input_dim, output_dim=22)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for obs_batch, action_batch in dataloader:
            pred_action = model(obs_batch)
            loss = criterion(pred_action, action_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * obs_batch.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    return model


if __name__ == "__main__":
    data_path = "/data/xinyu/demo_DexArt_1036/laptop"
    model = train(data_path)
