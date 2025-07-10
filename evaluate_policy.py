import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('tax3d-conditioned-mimicgen')

import argparse
import hydra
from hydra.core.config_store import ConfigStore
import hydra.utils as utils
from omegaconf import OmegaConf  # For config handling if needed
from dexart.env.task_setting import TRAIN_CONFIG, RANDOM_CONFIG
import numpy as np
from dexart.env.create_env import create_env
from stable_baselines3 import PPO
from examples.train import get_3d_policy_kwargs
from tqdm import tqdm
import pickle
import random
import torch
from equi_diffpo.policy.dp3 import DP3
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from dexart_il_wrapper import DexArt_IL_Wrapper
import collections
from collections import deque
#from train_dp3 import set_random_quaternion

def retrieve_goal(demo_id):
    folder_path = '/data/xinyu/demo_dexart_unseen/laptop'
    prefix = f'demo_{demo_id}_'

    matched_file = None
    for fname in os.listdir(folder_path):
        if fname.startswith(prefix):
            matched_file = os.path.join(folder_path, fname)
            break

    if matched_file:
        if matched_file.endswith('.pkl'):
            with open(matched_file, 'rb') as f:
                traj = pickle.load(f)
    else:
        return None
    
    progress_array = np.array([i['obs']['progress'] for i in traj])
    subgoal_idx = np.where(progress_array > 1e-5)[0][0]

    return np.array([traj[subgoal_idx]["obs"]["imagined_robot_point_cloud"], traj[-1]["obs"]["imagined_robot_point_cloud"]])



def get_obs(obs):
    """Observation for saving"""
    state = obs['state']  # shape (32,)
    obs_dict = {
        'robot_qpos_vec': state[:22],
        'palm_v': state[22: 25],
        'palm_w': state[25: 28],
        'palm_pose.p': state[28: 31],
        "palm_pose.q": obs['quat_obs'],
        'observed_point_cloud': obs['instance_1-point_cloud'],  # (512, 3)
        "observed_pc_seg-gt": obs['instance_1-seg_gt'],         # (512, 4)
        'imagined_robot_point_cloud': obs['imagination_robot'][:, :3],  # (96, 3)
        'imagined_robot_pc_seg-gt': obs['imagination_robot'][:, 3:],    # (96, 4)
        'progress': obs['progress']
    }
    return obs_dict

N_OBS_STEPS = 2

def get_dp3_obs(obs_dict, obs, oracle_goal, device, n_obs_steps, eval_with_scene_seg, goal_mode):
    """Observation for DP3 inference"""
    state = obs['state'].squeeze()  # shape (32,)
    robot_qpos_vec = state[:22]
    if goal_mode == 'None':
        goal_gripper_pcd = torch.tensor(obs['imagination_robot'][:, :, :3], dtype=torch.float32).to(device)
    elif goal_mode == 'pointcloud_oracle':
        if (obs['progress'] > 10e-6):
            goal_gripper_pcd = torch.tensor(oracle_goal[1], dtype=torch.float32).to(device)
        else:
            goal_gripper_pcd = torch.tensor(oracle_goal[0], dtype=torch.float32).to(device)
        goal_gripper_pcd = goal_gripper_pcd[None]
    
    point_cloud = torch.tensor(obs['instance_1-point_cloud'], dtype=torch.float32).to(device)
    imagin_robot = torch.tensor(obs['imagination_robot'][:, :, :3], dtype=torch.float32).to(device)
    robot0_eef_pos = torch.tensor(state[28:31], dtype=torch.float32).to(device)[None]
    robot0_eef_quat = torch.tensor(obs['quat_obs'], dtype=torch.float32).to(device)
    robot0_gripper_qpos = torch.tensor(robot_qpos_vec[-16:], dtype=torch.float32).to(device)[None]
    observed_pc_seg_gt = torch.tensor(obs['instance_1-seg_gt'], dtype=torch.float32).to(device)
    imagined_robot_pc_seg_gt = torch.tensor(obs['imagination_robot'][:, :, 3:], dtype=torch.float32).to(device)

    if obs_dict is None:  # First step
        obs_dict = {
            'point_cloud': torch.cat([point_cloud] * n_obs_steps, dim=0)[None],
            'imagin_robot': torch.cat([imagin_robot] * n_obs_steps, dim=0)[None],
            'goal_gripper_pcd': torch.cat([goal_gripper_pcd] * n_obs_steps, dim=0)[None],
            'robot0_eef_pos': torch.cat([robot0_eef_pos] * n_obs_steps, dim=0)[None],
            'robot0_eef_quat': torch.cat([robot0_eef_quat] * n_obs_steps, dim=0)[None],
            'robot0_gripper_qpos': torch.cat([robot0_gripper_qpos] * n_obs_steps, dim=0)[None],
        }
        if eval_with_scene_seg:
            obs_dict['observed_pc_seg-gt'] = torch.cat([observed_pc_seg_gt] * n_obs_steps, dim=0)[None]
            obs_dict['imagined_robot_pc_seg-gt'] = torch.cat([imagined_robot_pc_seg_gt] * n_obs_steps, dim=0)[None]

    else:  # Succeeding steps
        new_values = {
            'point_cloud': point_cloud,
            'imagin_robot': imagin_robot,
            'goal_gripper_pcd': goal_gripper_pcd,
            'robot0_eef_pos': robot0_eef_pos,
            'robot0_eef_quat': robot0_eef_quat,
            'robot0_gripper_qpos': robot0_gripper_qpos,
        }
        if eval_with_scene_seg:
            new_values['observed_pc_seg-gt'] = observed_pc_seg_gt
            new_values['imagined_robot_pc_seg-gt'] = imagined_robot_pc_seg_gt
        
        obs, n_points, _ = new_values["point_cloud"].shape
        obs_dict["point_cloud"] = obs_dict["point_cloud"][:,:,:n_points,:] # remove imagin pcd concatenation

        for key in obs_dict.keys():
            obs_dict[key] = torch.cat((obs_dict[key][:, 1:], new_values[key].unsqueeze(0)), dim=1)  # Slide window
    
    return obs_dict

def prepare_dp3(cfg, device, checkpoint_path):
    # seed = cfg.training.seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    model: DP3 = hydra.utils.instantiate(cfg.policy).to(device)

    checkpoint = torch.load(f"{utils.get_original_cwd()}/{checkpoint_path}", map_location=device)

    ## Checkpoint top-level keys: dict_keys(['cfg', 'state_dicts', 'pickles'])
    ## Keys in checkpoint['state_dicts']: dict_keys(['model', 'ema_model', 'optimizer'])

    if cfg.training.use_ema and 'ema_model' in checkpoint['state_dicts']:
        print("Loading EMA weights for evaluation")
        model.load_state_dict(checkpoint['state_dicts']['ema_model'])
    else:
        print("Loading standard model weights")
        model.load_state_dict(checkpoint['state_dicts']['model'])
    
    model.eval()  # set model to evaluation mode
    
    return model


def prepare_dexart(device, checkpoint_path):
    policy = DexArt_IL_Wrapper().to(device)
    # Load the checkpoint
    state_dict = torch.load(f"{utils.get_original_cwd()}/{checkpoint_path}", map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()  # Set to evaluation mode
    return policy

@hydra.main(version_base="1.1", config_path="tax3d-conditioned-mimicgen/equi_diffpo/config", config_name="eval_dexart")
def main(cfg):
    """
    python evaluate_policy.py eval.task_name=laptop eval.checkpoint_path=checkpoints_remote/100.ckpt eval.eval_per_instance=10 eval.model=dp3
    """
    eval_cfg = cfg.eval

    task_name = eval_cfg.task_name  # Now use cfg, which includes argparse overrides
    use_test_set = eval_cfg.use_test_set
    checkpoint_path = eval_cfg.checkpoint_path

    random.seed(eval_cfg.seed)
    np.random.seed(eval_cfg.seed)
    torch.manual_seed(eval_cfg.seed)

    device = "cuda:0"

    if use_test_set:
        indeces = TRAIN_CONFIG[task_name]['unseen']
        print(f"using unseen instances {indeces}")
    else:
        indeces = TRAIN_CONFIG[task_name]['seen']
        print(f"using seen instances {indeces}")

    rand_pos = RANDOM_CONFIG[task_name]['rand_pos']
    rand_degree = RANDOM_CONFIG[task_name]['rand_degree']
    env = create_env(task_name=task_name,
                     use_visual_obs=True,
                     use_gui=True,
                     is_eval=True,
                     pc_noise=True,
                     pc_seg=True,
                     index=indeces,
                     img_type='robot',
                     rand_pos=rand_pos,
                     rand_degree=rand_degree)
    env.seed(eval_cfg.seed)

    if eval_cfg.model == "ppo":
        policy = PPO.load(f"{utils.get_original_cwd()}/{checkpoint_path}", env, device,
                            policy_kwargs=get_3d_policy_kwargs(extractor_name='smallpn'),
                            check_obs_space=False, force_load=True)
        policy.set_random_seed(eval_cfg.seed)
    elif eval_cfg.model == "dp3":
        policy = prepare_dp3(cfg, device, checkpoint_path)
    elif eval_cfg.model == "dexart":
        policy = prepare_dexart(device, checkpoint_path)
    else:
        raise NotImplementedError

    eval_instances = len(env.instance_list)
    eval_per_instance = eval_cfg.eval_per_instance
    eval_with_scene_seg = cfg.with_scene_seg
    goal_mode = cfg.policy.goal_mode
    success_list = list()
    reward_list = list()
    progress_list = list()
    stage_list = list()

    # demo_save_dir_success = os.path.join('demo_dp3', task_name, 'success_demo')
    # demo_save_dir_failure = os.path.join('demo_dp3', task_name, 'failure_demo')
    # os.makedirs(demo_save_dir_success, exist_ok=True)
    # os.makedirs(demo_save_dir_failure, exist_ok=True)

    # demo_save_dir = os.path.join('demo_DexArt_unseen', task_name)
    # os.makedirs(demo_save_dir, exist_ok=True)
    
    success_id = 0
    demo_id = 0
    num = 0
    with tqdm(total=eval_per_instance * eval_instances) as pbar:
        for _ in range(eval_per_instance):       # Loop over number of episodes per instance
            for _ in range(eval_instances):
                
                episode_seed = eval_cfg.seed + demo_id * 1399
                env.seed(episode_seed)
                np.random.seed(episode_seed)
                torch.manual_seed(episode_seed)
                random.seed(episode_seed)

                obs = env.reset()
                eval_success = False
                progress = 0
                stage = 1
                reward_sum = 0
                demo_data = []
                obs_dict = None # Initialize at None
                action_queue = deque([])
                # flag = False

                # first_obs = get_obs(obs)
                # with open(os.path.join(demo_save_dir_success, f'first_obs_{demo_id}.pkl'), 'wb') as f:
                #     pickle.dump(first_obs, f)

                if goal_mode:
                    oracle_goal = retrieve_goal(demo_id)
                    if oracle_goal is None:
                        demo_id += 1
                        pbar.update(1)
                        # print("flag is true")
                        continue
                else:
                    oracle_goal = None

                for j in range(env.horizon):         # Loop for max steps
                    if isinstance(obs, dict):
                        for key, value in obs.items():
                            if isinstance(value, np.ndarray):
                                obs[key] = value[np.newaxis, :]
                            else:
                                obs[key] = np.array([[value]]) 
                    else:
                        obs = obs[np.newaxis, :]

                    if eval_cfg.model == "ppo":
                        action = policy.predict(observation=obs, deterministic=True)[0]
                    elif eval_cfg.model == "dp3":
                        obs_dict = get_dp3_obs(obs_dict, obs, oracle_goal, device, N_OBS_STEPS, eval_with_scene_seg, goal_mode)
                        
                        # Receding horizon control
                        if len(action_queue) == 0:
                            with torch.no_grad():
                                result = policy.predict_action(obs_dict)
                                actions = result['action'].squeeze()
                                action_queue.extend(actions.tolist())
                        action = np.array(action_queue.popleft()).astype(np.float32)
                    elif eval_cfg.model == "dexart":
                        obs_dict = get_dp3_obs(obs_dict, obs, device, N_OBS_STEPS)
                        with torch.no_grad():
                            result = policy.predict_action(obs_dict)
                        action = result['action'].squeeze().cpu().numpy().astype(np.float32)
                    else:
                        raise NotImplementedError

                    obs, reward, done, _ = env.step(action)
                    reward_sum += reward

                    observed = {
                        'obs': get_obs(obs),
                        'action': action,  # shape (22,)
                        'reward': reward
                    }
                    demo_data.append(observed)

                    # Observation Structure:
                    #     instance_1-seg_gt: shape=(512, 4), dtype=float64
                    #     instance_1-point_cloud: shape=(512, 3), dtype=float64
                    #     imagination_robot: shape=(96, 7), dtype=float64
                    #     state: shape=(32,), dtype=float64
                    #     oracle_state: shape=(32,), dtype=float64

                    
                    # print("Observation Structure:")
                    # for key, value in obs.items():
                    #      print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                    # break
                    
                    # # Extract and concatenate point cloud data from obs
                    # observed_pc = np.concatenate([S
                    #     obs['instance_1-point_cloud'],    # (512, 3) point cloud
                    #     obs['instance_1-seg_gt']          # (512, 4) segmentation label
                    # ], axis=1)                             # (512, 7) 
                    # observed_pc = np.concatenate([
                    #     observed_pc,
                    #     obs['imagination_robot']           # (96, 7)
                    # ], axis=0)                             # => (608, 7)
                    # assert observed_pc.shape == (608, 7)
                    # demo_data.append(observed_pc)  # Append to tpbar.update(1)rajectory list 

                    progress = env.progress
                    stage = env.state

                    if env.is_eval_done:
                        eval_success = True
                    
                    if done:
                        break
                
                # breakpoint()
                # print(flag)

                reward_list.append(reward_sum)
                success_list.append(int(eval_success))
                progress_list.append(progress)
                stage_list.append(stage)
                pbar.update(1)
                
                # if eval_success:
                #     #print(demo_data)         
                #     with open(os.path.join(demo_save_dir, f'demo_{demo_id}_{success_id}.pkl'), 'wb') as f:
                #         pickle.dump(demo_data, f)
                    
                #     success_id +=1
                #     if success_id == 600:
                #         print("600!!!")
    
                # if eval_success:
                #     #print(demo_data)         
                #     with open(os.path.join(demo_save_dir_success, f'demo_{demo_id}.pkl'), 'wb') as f:
                #         pickle.dump(demo_data, f)
                # else:
                #     with open(os.path.join(demo_save_dir_failure, f'demo_{demo_id}.pkl'), 'wb') as f:
                #         pickle.dump(demo_data, f)
                
                demo_id += 1
                num += 1

    print(num)
    print(progress_list)
    print(stage_list)
    print(f"checkpoint in {checkpoint_path} success rate = {np.mean(success_list)}")
    # print(f"Saved {demo_id} demos to: {demo_save_dir_success} and {demo_save_dir_failure}")

if __name__ == "__main__":
    main()
