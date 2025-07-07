import matplotlib.pyplot as plt
import torch
from torch import optim
from dp3_dexart_dataset import DP3DexArtDataset, get_dataloaders
import pickle
import sys
from tqdm import tqdm
sys.path.append('tax3d-conditioned-mimicgen')
from equi_diffpo.model.vision.articubot import PointNet2_super
from train_high_level import compute_weighted_displacement
import hydra

def extract_model_input(obs_batch, device):
    pcd = torch.from_numpy(obs_batch["observed_point_cloud"])
    imagined_pcd = torch.from_numpy(obs_batch["imagined_robot_point_cloud"])
    obs = torch.cat([pcd, imagined_pcd], axis=0)[None].permute(0,2,1).to(device).float()
    return obs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_file = "outputs/2025-07-06/00-44-41/dp3_epoch_11.pt"
model_type = "pn_plus_plus"

eval_file = "/data/xinyu/demo_dexart_Jun18/laptop/demo_3707.pkl"
eval_data = pickle.load(open(eval_file, "rb"))
output_file = "demo_3707_with_pred.pkl"
output_data = []

if model_type == "pn_plus_plus":
    model = PointNet2_super(num_classes=13, input_channel=3).to(device)
    model.load_state_dict(torch.load(ckpt_file))
else:
    raise NotImplementedError

for step in tqdm(eval_data):
    obs = extract_model_input(step["obs"], device)
    with torch.no_grad():
        pred = model(obs)
    pred_points = compute_weighted_displacement(obs, pred)

    step["pred_goal"] = pred_points.squeeze().cpu().numpy()
    output_data.append(step)


with open(output_file, 'wb') as f:
    pickle.dump(output_data, f)
