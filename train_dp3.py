from dexart_il_wrapper import DexArt_IL_Wrapper
from dp3_dexart_dataset import DP3DexArtDataset, get_dataloaders
import torch
import torch.optim as optim

import sys
sys.path.append('tax3d-conditioned-mimicgen')
from equi_diffpo.policy.dp3 import DP3
from equi_diffpo.model.common.normalizer import LinearNormalizer
from diffusers.schedulers import DDPMScheduler
import os
import hydra
import collections
import open3d as o3d

#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

def build_normalizer(dataset):
    obs_accum = collections.defaultdict(list)
    action_accum = []

    for sample in dataset:
        obs = sample["obs"]
        action = sample["action"]

        # Exclude point cloud fields from normalization? CONFIRM
        obs_clean = {k: v for k, v in obs.items() if k not in ['point_cloud', 'imagin_robot', 'goal_gripper_pcd', 'observed_pc_seg-gt', 'imagined_robot_pc_seg-gt']}

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
    data_dir = "/data/xinyu/demo_dexart_Jun18/laptop"
    batch_size = 128
    num_epochs = 500

    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DP3DexArtDataset(data_dir, goal_mode=cfg.policy.goal_mode, with_scene_seg=cfg.with_scene_seg)
    train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size)

    shape_meta = {
        'obs': {
            'point_cloud': {'shape': (512, 3)},
            'imagin_robot': {'shape': (96, 3)},
            'goal_gripper_pcd': {'shape': (96, 3)},
            'robot0_eef_pos': {'shape': (3,)},
            'robot0_eef_quat': {'shape': (4,)},        
            'robot0_gripper_qpos': {'shape': (16,)},
            'observed_pc_seg-gt': {'shape': (512, 4)},
            'imagined_robot_pc_seg-gt': {'shape': (96, 4)},
        },
        'action': {'shape': (22,)}
    }


    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    horizon = 16
    n_action_steps = 8
    n_obs_steps = 2

    pointcloud_encoder_cfg = cfg.policy.get("pointcloud_encoder_cfg", None)

    if cfg.policy.type == "dp3":
        model = DP3(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            pointnet_type=cfg.policy.pointnet_type,
            goal_mode=cfg.policy.goal_mode,
        ).to(device)
    elif cfg.policy.type == "dexart":
        model = DexArt_IL_Wrapper().to(device)
    else:
        raise NotImplementedError

    normalizer = build_normalizer(dataset)
    model.set_normalizer(normalizer)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    with open("loss_log.txt", "w") as f:
        f.write("Epoch,TrainLoss,ValLoss\n")

    avg_train_losses = []
    avg_val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_val_loss = 0.0
        train_count = 0
        val_count = 0

        for batch in train_loader:
            
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

            total_train_loss += loss.item()
            train_count += 1


        avg_train_loss = total_train_loss / train_count
        avg_train_losses.append(avg_train_loss)

        # ===== Validation =====
        model.eval()
        total_val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                obs_batch = {k: v.to(device) for k, v in batch["obs"].items()}
                action_batch = batch["action"].to(device)

                model_input = {"obs": obs_batch, "action": action_batch}
                loss, _, _ = model.compute_loss(model_input)

                total_val_loss += loss.item()
                val_count += 1

        avg_val_loss = total_val_loss / val_count
        avg_val_losses.append(avg_val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
            torch.save(model.state_dict(), f"dp3_epoch_{epoch+1}.pt")
        
        with open("loss_log.txt", "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f}\n")


    # ===== Final Test Evaluation =====
    model.eval()
    total_test_loss = 0.0
    test_count = 0

    with torch.no_grad():
        for batch in test_loader:
            obs_batch = {k: v.to(device) for k, v in batch["obs"].items()}
            action_batch = batch["action"].to(device)

            model_input = {"obs": obs_batch, "action": action_batch}
            loss, _, _ = model.compute_loss(model_input)

            total_test_loss += loss.item()
            test_count += 1

    avg_test_loss = total_test_loss / test_count
    print(f"Final Test Loss: {avg_test_loss:.4f}")


    if avg_train_losses:
        try:
            plt.figure()
            plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Training Loss')
            plt.title('Training Loss vs. Epochs')
            plt.grid(True)
            plt.show()
            #plt.savefig("training_loss_plot.png")
            #plt.close()
            print("Plot saved successfully.")
        except Exception as e:
            print("Failed to plot:", e)
    else:
        print("avg_train_losses is empty — skipping plot.")

if __name__ == "__main__":
    main()
