"""
Script to extract observations from low-dimensional simulation states in a robosuite dataset.

Args:
    dataset (str): path to input hdf5 dataset

    output_name (str): name of output hdf5 dataset

    n (int): if provided, stop after n trajectories are processed

    shaped (bool): if flag is set, use dense rewards

    camera_names (str or [str]): camera name(s) to use for image observations. 
        Leave out to not use image observations.

    camera_height (int): height of image observation.

    camera_width (int): width of image observation

    done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a success state.
        If 1, done is 1 at the end of each trajectory. If 2, both.

    copy_rewards (bool): if provided, copy rewards from source file instead of inferring them

    copy_dones (bool): if provided, copy dones from source file instead of inferring them

Example usage:
    
    # extract low-dimensional observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name low_dim.hdf5 --done_mode 2
    
    # extract 84x84 image observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # (space saving option) extract 84x84 image observations with compression and without 
    # extracting next obs (not needed for pure imitation learning algos)
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
        --compress --exclude-next-obs

    # use dense rewards, and only annotate the end of trajectories with done signal
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image_dense_done_1.hdf5 \
        --done_mode 1 --dense --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84
"""
import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy

from non_rigid.utils.script_utils import create_model, create_datamodule
import torch
# Load instantiate from hydra
import hydra
# Load an omegaconf config
from omegaconf import OmegaConf

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
from pytorch3d.transforms import Translate
from equi_diffpo.model.vision.pointnet2_classification import PointNet2_Cls
from train_prediction_classifier import ClassifierModule

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


TAX3D_CKPT_PATH = {
    'Square_D2': '/home/yingyuan/equidiff/third_party/non-rigid/scripts/logs/train_mimicgen_df_cross/2025-03-12/23-47-12/checkpoints/last.ckpt',
    'ThreePieceAssembly_D2': '/home/yingyuan/equidiff/third_party/non-rigid/scripts/logs/train_mimicgen_df_cross/2025-04-03/14-16-18/checkpoints/last.ckpt',
}

CLASSIFIER_CKPT_PATH = {
    'Square_D2': '/home/yingyuan/equidiff/logs/train_mimicgen_df_cross/2025-03-22/13-38-01/checkpoints/last.ckpt',
    'ThreePieceAssembly_D2': '/home/yingyuan/equidiff/logs/train_mimicgen_df_cross/2025-04-03/14-15-49/checkpoints/last.ckpt',
}


def extract_trajectory(
    env_meta,
    args, 
    initial_state, 
    states, 
    actions,
    tax3d_model=None,
    classifier_model=None,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a 
            success state. If 1, done is 1 at the end of each trajectory. 
            If 2, do both.
    """
    done_mode = args.done_mode
    if env_meta['env_name'].startswith('PickPlace'):
        camera_names=['birdview', 'agentview', 'robot0_eye_in_hand']
    else:
        camera_names=['birdview', 'agentview', 'sideview', 'robot0_eye_in_hand']
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        # camera_names=['frontview', 'birdview', 'agentview', 'sideview', 'agentview_full', 'robot0_robotview', 'robot0_eye_in_hand'], 
        camera_names=camera_names, 
        camera_height=args.camera_height, 
        camera_width=args.camera_width, 
        reward_shaping=args.shaped,
    )
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state)

    gripper_action = actions[..., -1]
    switch_indices = list(np.where(np.sign(gripper_action[:-1]) != np.sign(gripper_action[1:]))[0])
    switch_indices.append(len(gripper_action) - 1)
    switch_indices = np.array(switch_indices)

    # assert len(switch_indices) == 3

    traj = dict(
        obs=[], 
        next_obs=[], 
        rewards=[], 
        dones=[], 
        actions=np.array(actions), 
        states=np.array(states), 
        initial_state_dict=initial_state,
    )
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):

        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states" : states[t]})

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # get tax3d prediction
    tax3d_batch = {
            'pc_action': torch.tensor(np.array(traj["obs"]["point_cloud"])[:, 1024:, :3]).float().cuda(),
            'pc_anchor': torch.tensor(np.array(traj["obs"]["point_cloud"])[:, :1024, :3]).float().cuda(),
        }

    # using anchor_center
    center = tax3d_batch['pc_anchor'].mean(axis=-2).unsqueeze(1)
    action_center = tax3d_batch['pc_action'].mean(axis=-2).unsqueeze(1)

    tax3d_batch['pc_anchor'] = tax3d_batch['pc_anchor'] - center
    tax3d_batch['pc_action'] = tax3d_batch['pc_action'] - action_center
    action_seg = torch.zeros_like(tax3d_batch['pc_action'][:, 0]).int()
    tax3d_batch['seg'] = action_seg
    tax3d_batch['rel_pose'] = (action_center - center).squeeze(1)

    tax3d_batch['T_goal2world'] = Translate(center.squeeze(1)).get_matrix()
    tax3d_batch['T_action2world'] = Translate(action_center.squeeze(1)).get_matrix()

    num_samples = 1
    pred_dict = tax3d_model.predict(tax3d_batch, 
                                        num_samples=num_samples, 
                                        unflatten=False,
                                        progress=True,
                                        full_prediction=True)
    action_pred = pred_dict['point']['pred_world']
    action_pred = action_pred.reshape(-1, num_samples, 138, 3)  # [B*N, num_samples, 138, 3]

    # get classifier scores
    classifier_batch = {
        'scene': torch.tensor(np.array(traj["obs"]["point_cloud"])[:, :1024, :3]).reshape(-1, 1, 1024, 3).repeat(1, num_samples, 1, 1).reshape(-1, 1024, 3).cuda(),
        'pc': action_pred.reshape(-1, 138, 3)
    }
    classifier_score = classifier_model.predict(classifier_batch)['pred'].reshape(-1, num_samples, 1)
    _, indices = torch.max(classifier_score, dim=1)  # [B, 1]
    indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 138, 3)  # [B, N, 1, 138, 3]
    action_pred_wta = torch.gather(action_pred.reshape(-1, num_samples, 138, 3), dim=1, index=indices).squeeze(1)
    
    # add goal gripper pcd in each stage, based on finger switch
    goal_gripper_pcd = []
    all_goal_gripper_pcd = []
    goal_eef_pos, goal_eef_quat = [], []
    all_goal_eef_pos, all_goal_eef_quat = [], []

    for index in switch_indices:
        current_len = len(goal_gripper_pcd)

        # update goal point cloud
        tmp = traj['obs']['point_cloud'][index][1024:, :]
        all_goal_gripper_pcd.append(tmp)
        tmp = tmp.reshape(1, tmp.shape[0], tmp.shape[1])
        tmp = np.tile(tmp, (index - current_len + 1, 1, 1))
        goal_gripper_pcd.extend(tmp)

        # update low-level goal info
        tmp_pos = traj['obs']['robot0_eef_pos'][index]
        tmp_quat = traj['obs']['robot0_eef_quat'][index]
        all_goal_eef_pos.append(tmp_pos)
        all_goal_eef_quat.append(tmp_quat)
        tmp_pos = tmp_pos.reshape(1, -1)
        tmp_quat = tmp_quat.reshape(1, -1)
        tmp_pos = np.tile(tmp_pos, (index - current_len + 1, 1))
        tmp_quat = np.tile(tmp_quat, (index - current_len + 1, 1))
        goal_eef_pos.extend(tmp_pos)
        goal_eef_quat.extend(tmp_quat)

    all_goal_gripper_pcd = np.array(all_goal_gripper_pcd)
    traj['initial_state_dict']['goal_gripper_pcd'] = all_goal_gripper_pcd
    all_goal_eef_pos = np.array(all_goal_eef_pos)
    all_goal_eef_quat = np.array(all_goal_eef_quat)
    traj['initial_state_dict']['goal_eef_pos'] = all_goal_eef_pos
    traj['initial_state_dict']['goal_eef_quat'] = all_goal_eef_quat
    # override
    # goal_gripper_pcd = np.tile(all_goal_gripper_pcd[:2].reshape(1, -1, 6), (len(goal_gripper_pcd), 1, 1))

    traj["obs"]["goal_gripper_pcd"] = goal_gripper_pcd
    traj["obs"]["goal_eef_pos"] = goal_eef_pos
    traj["obs"]["goal_eef_quat"] = goal_eef_quat
    traj["obs"]["tax3d"] = action_pred_wta.cpu().numpy()


    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj

def worker(x):
    env_meta, args, initial_state, states, actions, tax3d_model, classifier_model = x
    traj = extract_trajectory(
        env_meta=env_meta,
        args=args,
        initial_state=initial_state, 
        states=states, 
        actions=actions,
        tax3d_model=tax3d_model,
        classifier_model=classifier_model,
    )
    return traj

def dataset_states_to_obs(args):
    num_workers = args.num_workers
    # create environment to use for data processing
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.input)

    # Load the config
    tax3d_cfg = OmegaConf.load("equi_diffpo/config/model/tax3d.yaml")
    tax3d_cfg, datamodule = create_datamodule(tax3d_cfg)
    network, tax3d_model = create_model(tax3d_cfg)
    ckpt_file = TAX3D_CKPT_PATH[env_meta['env_name']]
    ckpt = torch.load(ckpt_file)
    tax3d_model.load_state_dict(ckpt["state_dict"])
    tax3d_cfg = tax3d_cfg
    tax3d_model.cuda()
    tax3d_model.eval()

    classifier_network = PointNet2_Cls()
    classifier_model = ClassifierModule(network=classifier_network, cfg=tax3d_cfg)
    ckpt_file = CLASSIFIER_CKPT_PATH[env_meta['env_name']]
    ckpt = torch.load(ckpt_file)
    classifier_model.load_state_dict(ckpt["state_dict"])
    classifier_model.cuda()
    classifier_model.eval()

    if env_meta['env_name'].startswith('PickPlace'):
        camera_names=['birdview', 'agentview', 'robot0_eye_in_hand']
    else:
        camera_names=['birdview', 'agentview', 'sideview', 'robot0_eye_in_hand']
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        # camera_names=['frontview', 'birdview', 'agentview', 'sideview', 'agentview_full', 'robot0_robotview', 'robot0_eye_in_hand'], 
        camera_names=camera_names, 
        camera_height=args.camera_height, 
        camera_width=args.camera_width, 
        reward_shaping=args.shaped,
    )

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.input, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # output file in same directory as input file
    output_path = args.output
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print("input file: {}".format(args.input))
    print("output file: {}".format(output_path))

    total_samples = 0
    for i in range(0, len(demos), num_workers):
        end = min(i + num_workers, len(demos))
        initial_state_list = []
        states_list = []
        actions_list = []
        for j in range(i, end):
            ep = demos[j]
            # prepare initial state to reload from
            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            if is_robosuite_env:
                initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            actions = f["data/{}/actions".format(ep)][()]

            initial_state_list.append(initial_state)
            states_list.append(states)
            actions_list.append(actions)
            
        with multiprocessing.Pool(num_workers) as pool:
            trajs = pool.map(worker, [[env_meta, args, initial_state_list[j], states_list[j], actions_list[j], tax3d_model, classifier_model] for j in range(len(initial_state_list))]) 

        for j, ind in enumerate(range(i, end)):
            ep = demos[ind]
            traj = trajs[j]
            # maybe copy reward or done signal from source file
            if args.copy_rewards:
                traj["rewards"] = f["data/{}/rewards".format(ep)][()]
            if args.copy_dones:
                traj["dones"] = f["data/{}/dones".format(ep)][()]

            # store transitions

            # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
            #            consistent as well
            ep_data_grp = data_grp.create_group(ep)
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states/states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("states/goal_gripper_pcd", data=np.array(traj["initial_state_dict"]["goal_gripper_pcd"]))
            ep_data_grp.create_dataset("states/goal_eef_pos", data=np.array(traj["initial_state_dict"]["goal_eef_pos"]))
            ep_data_grp.create_dataset("states/goal_eef_quat", data=np.array(traj["initial_state_dict"]["goal_eef_quat"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            for k in traj["obs"]:
                if args.compress:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip")
                else:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                if not args.exclude_next_obs:
                    if args.compress:
                        ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]), compression="gzip")
                    else:
                        ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if is_robosuite_env:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]
            print("ep {}: wrote {} transitions to group {}".format(ind, ep_data_grp.attrs["num_samples"], ep))
        
        del trajs

    # copy over all filter keys that exist in the original hdf5
    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    print("Wrote {} trajectories to {}".format(len(demos), output_path))

    f.close()
    f_out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="name of output hdf5 dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped", 
        action='store_true',
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=[],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=84,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=84,
        help="(optional) width of image observations",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever 
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal 
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=2,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards", 
        action='store_true',
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones", 
        action='store_true',
        help="(optional) copy dones from source file instead of inferring them",
    )

    # flag to exclude next obs in dataset
    parser.add_argument(
        "--exclude-next-obs", 
        type=bool,
        default=True,
        help="(optional) exclude next obs in dataset",
    )

    # flag to compress observations with gzip option in hdf5
    parser.add_argument(
        "--compress", 
        type=bool,
        default=True,
        help="(optional) compress observations with gzip option in hdf5",
    )

    args = parser.parse_args()
    dataset_states_to_obs(args)
