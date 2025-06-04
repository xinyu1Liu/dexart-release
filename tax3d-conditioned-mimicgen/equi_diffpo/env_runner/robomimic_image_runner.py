import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from equi_diffpo.gym_util.async_vector_env import AsyncVectorEnv
from equi_diffpo.gym_util.sync_vector_env import SyncVectorEnv
from equi_diffpo.gym_util.multistep_wrapper import MultiStepWrapper
from equi_diffpo.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from equi_diffpo.model.common.rotation_transformer import RotationTransformer

from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.env_runner.base_image_runner import BaseImageRunner
from equi_diffpo.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from non_rigid.utils.script_utils import create_model, create_datamodule
from non_rigid.metrics.flow_metrics import flow_rmse
from equi_diffpo.model.vision.pointnet2_classification import PointNet2_Cls
from train_prediction_classifier import ClassifierModule

from pytorch3d.transforms import Transform3d, Translate, quaternion_to_matrix, matrix_to_quaternion
import pickle

NUM_SCENE_PCD = 1024

TAX3D_CKPT_PATH = {
    'square_d2': './third_party/non-rigid/scripts/logs/train_mimicgen_df_cross/2025-03-12/23-47-12/checkpoints/last.ckpt',
    'three_piece_assembly_d2': './third_party/non-rigid/scripts/logs/train_mimicgen_df_cross/2025-04-03/14-16-18/checkpoints/last.ckpt',
}

CLASSIFIER_CKPT_PATH = {
    'square_d2': './logs/train_mimicgen_df_cross/2025-03-22/13-38-01/checkpoints/last.ckpt',
    'three_piece_assembly_d2': './logs/train_mimicgen_df_cross/2025-04-03/14-15-49/checkpoints/last.ckpt',
}

def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    return env


def align_point_clouds(P, Q):
    """
    Align two point clouds using SVD to compute the rotation matrix and translation vector.

    Parameters:
    P (torch.tensor): Reference point cloud of shape (B, N, 3).
    Q (torch.tensor): Transformed point cloud of shape (B, N, 3).

    Returns:
    R (torch.tensor): Rotation matrix of shape (B, 3, 3).
    T (torch.tensor): Translation vector of shape (B, 3,).
    """
    # Step 1: Compute centroids
    centroid_P = torch.mean(P, dim=1, keepdims=True)
    centroid_Q = torch.mean(Q, dim=1, keepdims=True)
    
    # Step 2: Center the point clouds
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # Step 3: Compute the covariance matrix
    H = P_centered.transpose(1, 2) @ Q_centered
    
    # Step 4: Perform SVD
    U, S, Vt = torch.linalg.svd(H)
    
    # Step 5: Compute the rotation matrix
    R = Vt.transpose(1, 2) @ U.transpose(1, 2)
    
    # Handle reflection case (ensure a proper rotation matrix with det(R) = 1)
    for i in range(len(R)):
        if torch.linalg.det(R[i]) < 0:
            Vt[i, -1, :] *= -1
            R[i] = Vt[i].T @ U[i].T
    
    # Step 6: Compute the translation vector
    T = centroid_Q - (R @ centroid_P.transpose(1, 2)).transpose(1, 2)
    
    return R, T


class RobomimicImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            task_name,
            shape_meta:dict,
            goal_mode='None',  # ['None', 'pointcloud_oracle', 'pointcloud_tax3d', 'lowdim_oracle', 'lowdim_tax3d']
            num_samples=5,
            predict_tax3d_every_step=True,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            test_start_idx=1000,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            tax3d_cfg=None,
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, 
                shape_meta=shape_meta
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    shape_meta=shape_meta,
                    enable_render=False
                )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                # init_state = {'states': f[f'data/demo_{train_idx}/states'][0],
                #               'model': f[f'data/demo_{train_idx}'].attrs['model_file']}

                if goal_mode.startswith('pointcloud'):
                    init_state = {'states': f[f'data/demo_{train_idx}/states/states'][0],
                                  'goal_gripper_pcd': torch.tensor(f[f'data/demo_{train_idx}/states/goal_gripper_pcd'][:]),
                                  'model': f[f'data/demo_{train_idx}'].attrs['model_file']}
                elif goal_mode.startswith('lowdim'):
                    init_state = {'states': f[f'data/demo_{train_idx}/states/states'][0],
                                'goal_gripper_pcd': torch.tensor(f[f'data/demo_{train_idx}/states/goal_gripper_pcd'][:]),
                                'goal_eef_pos': torch.tensor(f[f'data/demo_{train_idx}/states/goal_eef_pos'][:]),
                                'goal_eef_quat': torch.tensor(f[f'data/demo_{train_idx}/states/goal_eef_quat'][:]),
                                'model': f[f'data/demo_{train_idx}'].attrs['model_file']}
                elif goal_mode == 'None':
                    init_state = {'states': f[f'data/demo_{train_idx}/states/states'][0],
                                  'model': f[f'data/demo_{train_idx}'].attrs['model_file']}
                else:
                    raise NotImplementedError

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        # test
        # for i in range(n_test):
        #     seed = test_start_seed + i
        #     enable_render = i < n_test_vis

        #     def init_fn(env, seed=seed, 
        #         enable_render=enable_render):
        #         # setup rendering
        #         # video_wrapper
        #         assert isinstance(env.env, VideoRecordingWrapper)
        #         env.env.video_recoder.stop()
        #         env.env.file_path = None
        #         if enable_render:
        #             filename = pathlib.Path(output_dir).joinpath(
        #                 'media', wv.util.generate_id() + ".mp4")
        #             filename.parent.mkdir(parents=False, exist_ok=True)
        #             filename = str(filename)
        #             env.env.file_path = filename

        #         # switch to seed reset
        #         assert isinstance(env.env.env, RobomimicImageWrapper)
        #         env.env.env.init_state = None
        #         env.seed(seed)

        #     env_seeds.append(seed)
        #     env_prefixs.append('test/')
        #     env_init_fn_dills.append(dill.dumps(init_fn))

        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_test):
                test_idx = test_start_idx + i
                enable_render = i < n_test_vis
                # init_state = {'states': f[f'data/demo_{test_idx}/states'][0],
                #               'model': f[f'data/demo_{test_idx}'].attrs['model_file']}

                if goal_mode.startswith('pointcloud'):
                    init_state = {'states': f[f'data/demo_{test_idx}/states/states'][0],
                                  'goal_gripper_pcd': torch.tensor(f[f'data/demo_{test_idx}/states/goal_gripper_pcd'][:]),
                                  'model': f[f'data/demo_{test_idx}'].attrs['model_file']}
                elif goal_mode.startswith('lowdim'):
                    init_state = {'states': f[f'data/demo_{test_idx}/states/states'][0],
                                'goal_gripper_pcd': torch.tensor(f[f'data/demo_{test_idx}/states/goal_gripper_pcd'][:]),
                                'goal_eef_pos': torch.tensor(f[f'data/demo_{test_idx}/states/goal_eef_pos'][:]),
                                'goal_eef_quat': torch.tensor(f[f'data/demo_{test_idx}/states/goal_eef_quat'][:]),
                                'model': f[f'data/demo_{test_idx}'].attrs['model_file']}
                elif goal_mode == 'None':
                    init_state = {'states': f[f'data/demo_{test_idx}/states/states'][0],
                                  'model': f[f'data/demo_{test_idx}'].attrs['model_file']}
                else:
                    raise NotImplementedError

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(test_idx)
                env_prefixs.append('test/')
                env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.max_rewards = {}
        for prefix in self.env_prefixs:
            self.max_rewards[prefix] = 0
        self.goal_mode = goal_mode
        self.num_samples = num_samples
        self.predict_tax3d_every_step = predict_tax3d_every_step

        # initialize tax3d model
        if goal_mode.endswith('tax3d'):
            tax3d_cfg, datamodule = create_datamodule(tax3d_cfg)
            network, self.tax3d_model = create_model(tax3d_cfg)
            print("Attempting to use tax3d checkpoint... ")  # , tax3d_cfg.checkpoint.reference)
            # api = wandb.Api()
            # artifact_dir = tax3d_cfg.wandb.artifact_dir
            # artifact = api.artifact(tax3d_cfg.checkpoint.reference, type="model")
            # ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
            ckpt_file = TAX3D_CKPT_PATH[task_name]
            print(ckpt_file)
            ckpt = torch.load(ckpt_file)
            self.tax3d_model.load_state_dict(ckpt["state_dict"])
            self.tax3d_cfg = tax3d_cfg
            self.tax3d_model.cuda()
            self.tax3d_model.eval()

            # load classifier
            print("Attempting to use classifier checkpoint...")
            classifier_network = PointNet2_Cls()
            self.classifier_model = ClassifierModule(network=classifier_network, cfg=tax3d_cfg)
            ckpt_file = CLASSIFIER_CKPT_PATH[task_name]
            print(ckpt_file)
            ckpt = torch.load(ckpt_file)
            self.classifier_model.load_state_dict(ckpt["state_dict"])
            self.classifier_model.cuda()
            self.classifier_model.eval()

        if goal_mode.startswith('lowdim'):
            # loading gripper mesh
            with open('eef_pointcloud_wiz_pose_info.pkl', 'rb') as f:
                reference_eef_data = pickle.load(f)
            self.reference_eef_data = reference_eef_data

            self.robot_eef_pointcloud = torch.tensor(self.reference_eef_data['pointcloud']).cuda()
            reference_quat = self.reference_eef_data['quat']
            reference_quat = torch.tensor([reference_quat[3], reference_quat[0], reference_quat[1], reference_quat[2]]).cuda()  # xyzw -> wxyz
            self.reference_rot = quaternion_to_matrix(reference_quat).float()
            self.reference_pos = torch.tensor(self.reference_eef_data['pos']).cuda()

            tmp1 = (self.robot_eef_pointcloud.unsqueeze(0) - self.reference_pos).float()
            self.original_point_cloud = (self.reference_rot.T @ tmp1.transpose(1, 2)).transpose(1, 2)

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        if self.goal_mode.endswith('tax3d'):
            self.classifier_model.to(device)
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            num_steps = 0
            if self.goal_mode.endswith('tax3d'):
                self.prev_step_tax3d_prediction = None
                self.prev_step_tax3d_pos = None
                self.prev_step_tax3d_quat = None
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                
                # get tax3d-predicted goal gripper pcd
                if self.goal_mode.endswith('tax3d'): #  and num_steps % 1 == 0:
                    B, N, _, _ = obs_dict['point_cloud'].shape

                    if self.prev_step_tax3d_prediction is not None and not self.predict_tax3d_every_step:
                        # use classifier to decide which envs to run tax3d
                        classifier_batch = {
                            'scene': obs_dict['point_cloud'][..., :NUM_SCENE_PCD, :3].reshape(-1, NUM_SCENE_PCD, 3),  # (B*N, NUM_SCENE_PCD, 3)
                            'pc': self.prev_step_tax3d_prediction.reshape(-1, 138, 3)  # (B*N, 138, 3)
                        }
                        classifier_score = self.classifier_model.predict(classifier_batch)['pred'].reshape(B, N, 1)
                        min_vals, _ = classifier_score.min(dim=1)
                        recal_tax3d_indices = torch.where(min_vals[:, 0] < 0.999)[0]
                    else:
                        recal_tax3d_indices = torch.arange(B).to(device)
                    
                    print(recal_tax3d_indices)
                    if len(recal_tax3d_indices) > 0:
                        tax3d_batch = {
                            'pc_action': obs_dict['point_cloud'][recal_tax3d_indices, :, NUM_SCENE_PCD:, :3].reshape(-1, 138, 3),
                            'pc_anchor': obs_dict['point_cloud'][recal_tax3d_indices, :, :NUM_SCENE_PCD, :3].reshape(-1, NUM_SCENE_PCD, 3),
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

                        num_samples = self.num_samples
                        pred_dict = self.tax3d_model.predict(tax3d_batch, 
                                                            num_samples=num_samples, 
                                                            unflatten=False,
                                                            progress=True,
                                                            full_prediction=True)
                        action_pred = pred_dict['point']['pred_world'].to(device)
                        action_pred = action_pred.reshape(-1, num_samples, 138, 3)  # [B*N, num_samples, 138, 3]

                        # get classifier scores
                        classifier_batch = {
                            'scene': obs_dict['point_cloud'][recal_tax3d_indices, :, :NUM_SCENE_PCD, :3].reshape(-1, 1, NUM_SCENE_PCD, 3).repeat(1, num_samples, 1, 1).reshape(-1, NUM_SCENE_PCD, 3),
                            'pc': action_pred.reshape(-1, 138, 3)
                        }
                        classifier_score = self.classifier_model.predict(classifier_batch)['pred'].reshape(-1, N, num_samples, 1)
                        _, indices = torch.max(classifier_score, dim=2)  # [B, N, 1]
                        indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 138, 3)  # [B, N, 1, 138, 3]
                        action_pred_wta = torch.gather(action_pred.reshape(-1, N, num_samples, 138, 3), dim=2, index=indices).squeeze(2)
                        
                        # # getting wta sample
                        # seg = (action_seg == 0).to(device)
                        # ground_truth = obs_dict['goal_gripper_pcd'][..., :3].reshape(-1, 1, 138, 3)
                        # rmse = flow_rmse(action_pred, ground_truth, mask=False, seg=seg)
                        # winner = torch.argmin(rmse, dim=-1)
                        # action_pred_wta = action_pred[torch.arange(B * N), winner]
                        # action_pred_wta = action_pred_wta.reshape(B, N, 138, 3)

                        if self.prev_step_tax3d_prediction is not None:
                            self.prev_step_tax3d_prediction[recal_tax3d_indices, :, :, :3] = action_pred_wta
                        else:
                            self.prev_step_tax3d_prediction = action_pred_wta
                        obs_dict['goal_gripper_pcd'][..., :3] = self.prev_step_tax3d_prediction  # action_pred
                    else:
                        # use previous step prediction
                        obs_dict['goal_gripper_pcd'][..., :3] = self.prev_step_tax3d_prediction

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action
                num_steps += 1

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()
        
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        for i in range(len(self.env_fns)):
        # and comment out this line
        # for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value
            self.max_rewards[prefix] = max(self.max_rewards[prefix], value)
            log_data[prefix+'max_score'] = self.max_rewards[prefix]

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
