import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import argparse
from dexart.env.task_setting import TRAIN_CONFIG, IMG_CONFIG, RANDOM_CONFIG
from stable_baselines3.common.torch_layers import PointNetImaginationExtractorGP
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO
from stable_baselines3.simple_callback import SimpleCallback
import torch

import gymnasium as gym
import sapien
from mani_skill.envs.sapien_env import BaseEnv
import dexhier.spraybottle
import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_3d_policy_kwargs(extractor_name):
    feature_extractor_class = PointNetImaginationExtractorGP
    feature_extractor_kwargs = {"pc_key": "instance_1-point_cloud", "gt_key": "instance_1-seg_gt",
                                "extractor_name": extractor_name,
                                "imagination_keys": [f'imagination_{key}' for key in IMG_CONFIG['robot'].keys()],
                                "state_key": "state"}

    policy_kwargs = {
        "features_extractor_class": feature_extractor_class,
        "features_extractor_kwargs": feature_extractor_kwargs,
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
        "activation_fn": nn.ReLU,
    }
    return policy_kwargs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training.
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ep', type=int, default=10)
    parser.add_argument('--bs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--freeze', dest='freeze', action='store_true', default=False)
    parser.add_argument('--task_name', type=str, default="laptop")
    parser.add_argument('--extractor_name', type=str, default="smallpn")
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--save_path', type=str, default=BASE_DIR)

    # environment.
    parser.add_argument('--env_id', type=str, default='SprayBottle-v1')
    parser.add_argument('--robot_uids', type=str, default='xarm6_leap')
    parser.add_argument('--render_mode', type=str, default='rgb_array')
    parser.add_argument('--obs_mode', type=str, default='none')
    parser.add_argument('--reward_mode', type=str, default=None)
    parser.add_argument('--control_mode', type=str, default=None)
    parser.add_argument('--sim_backend', type=str, default='auto')
    parser.add_argument('--shader', type=str, default='default')
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--record_dir', type=str, default=None)
    args = parser.parse_args()

    task_name = args.task_name
    extractor_name = args.extractor_name
    seed = args.seed if args.seed >= 0 else random.randint(0, 100000)
    pretrain_path = args.pretrain_path
    horizon = 200
    env_iter = args.iter * horizon * args.num_envs
    print(f"freeze: {args.freeze}")

    if args.render_mode == "none":
        args.render_mode = None
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )

    # env = SubprocVecEnv([create_env_fn] * args.workers, "spawn")  # train on a list of envs.
    env = gym.make(args.env_id, robot_uids=args.robot_uids, **env_kwargs)
    # env = ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=False)
    print(env.observation_space)
    print(env.action_space)
    print(env.single_action_space)
    exit()

    model = PPO("PointCloudPolicy", env, verbose=1,
                n_epochs=args.ep,
                n_steps=(args.num_envs // args.workers) * horizon,
                learning_rate=args.lr,
                batch_size=args.bs,
                seed=seed,
                policy_kwargs=get_3d_policy_kwargs(extractor_name=extractor_name),
                min_lr=args.lr,
                max_lr=args.lr,
                adaptive_kl=0.02,
                target_kl=0.2,
                )

    if pretrain_path is not None:
        state_dict: OrderedDict = torch.load(pretrain_path)
        model.policy.features_extractor.extractor.load_state_dict(state_dict, strict=False)
        print("load pretrained model: ", pretrain_path)

    rollout = int(model.num_timesteps / (horizon * args.num_envs))

    # after loading or init the model, then freeze it if needed
    if args.freeze:
        model.policy.features_extractor.extractor.eval()
        for param in model.policy.features_extractor.extractor.parameters():
            param.requires_grad = False
        print("freeze model!")

    model.learn(
        total_timesteps=int(env_iter),
        reset_num_timesteps=False,
        iter_start=rollout,
        callback=SimpleCallback(model_save_freq=args.save_freq, model_save_path=args.save_path, rollout=0),
    )