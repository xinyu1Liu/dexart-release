# Training file for deciding whether the prediction of the TAX3D model is good or not.
# Positive examples: Ground-truth from the dataset.
# Negative examples: Bad TAX3D predictions or perturbation of the ground-truth.
# 
# In the first version, we do not include TAX3D predictions as negative examples.

import numpy as np
import torch
import pytorch_lightning as pl
import lightning as L
import torch.utils.data as data
import os
import json
from torch import nn, optim
import torch.nn.functional as F
from pathlib import Path
import omegaconf

import hydra
import lightning as L
import wandb
from diffusers import get_cosine_schedule_with_warmup
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from pytorch3d.transforms import Translate
from non_rigid.utils.transform_utils import random_se3
from non_rigid.utils.augmentation_utils import plane_occlusion
from non_rigid.utils.script_utils import create_model, create_datamodule
from equi_diffpo.model.vision.pointnet2_segmentation import PointNet2_Segm
from equi_diffpo.model.vision.pointnet2_classification import PointNet2_Cls

import open3d as o3d


@hydra.main(config_path="third_party/non-rigid/configs", config_name="train", version_base="1.3")
def main(cfg):
    print(
        json.dumps(
            omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
            sort_keys=True,
            indent=4,
        )
    )

    TESTING = "PYTEST_CURRENT_TEST" in os.environ

    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("medium")

    # Global seed for reproducibility.
    L.seed_everything(cfg.seed)

    ######################################################################
    # Create the datamodule.
    # The datamodule is responsible for all the data loading, including
    # downloading the data, and splitting it into train/val/test.
    #
    # This could be swapped out for a different datamodule in-place,
    # or with an if statement, or by using hydra.instantiate.
    ######################################################################
    if cfg.mode == "eval":
        job_cfg = cfg.inference
        # check for full action
        if job_cfg.action_full:
            cfg.dataset.sample_size_action = -1
        stage = "predict"
    elif cfg.mode == "train":
        job_cfg = cfg.training
        stage = "fit"
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")
    
    # setting up datamodule
    datamodule = PredictionClassifierDataModule(
        batch_size=job_cfg.batch_size,
        val_batch_size=job_cfg.val_batch_size,
        num_workers=cfg.resources.num_workers,
        dataset_cfg=cfg.dataset,
    )
    datamodule.setup(stage)

    # updating job config sample sizes
    if cfg.dataset.scene:
        job_cfg.sample_size = cfg.dataset.sample_size_action + cfg.dataset.sample_size_anchor
    else:
        job_cfg.sample_size = cfg.dataset.sample_size_action
        job_cfg.sample_size_anchor = cfg.dataset.sample_size_anchor

    # training-specific job config setup
    if cfg.mode == "train":
        job_cfg.num_training_steps = len(datamodule.train_dataloader()) * job_cfg.epochs

    ######################################################################
    # Create the network(s) which will be trained by the Training Module.
    # The network should (ideally) be lightning-independent. This allows
    # us to use the network in other projects, or in other training
    # configurations. Also create the Training Module.
    #
    # This might get a bit more complicated if we have multiple networks,
    # but we can just customize the training module and the Hydra configs
    # to handle that case. No need to over-engineer it. You might
    # want to put this into a "create_network" function somewhere so train
    # and eval can be the same.
    #
    # If it's a custom network, a good idea is to put the custom network
    # in `python_ml_project_template.nets.my_net`.
    ######################################################################

    # Model architecture is dataset-dependent, so we have a helper
    # function to create the model (while separating out relevant vals).
    network = PointNet2_Cls()
    model = ClassifierModule(network=network, cfg=cfg)

    ######################################################################
    # Set up logging in WandB.
    # This is a bit complicated, because we want to log the codebase,
    # the model, and the checkpoints.
    ######################################################################

    # If no group is provided, then we should create a new one (so we can allocate)
    # evaluations to this group later.
    if cfg.wandb.group is None:
        id = wandb.util.generate_id()
        group = "experiment-" + id
    else:
        group = cfg.wandb.group

    logger = WandbLogger(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        log_model=True,  # Only log the last checkpoint to wandb, and only the LAST model checkpoint.
        save_dir=cfg.wandb.save_dir,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=False
        ),
        job_type=cfg.job_type,
        save_code=True,  # This just has the main script.
        group=group,
    )

    ######################################################################
    # Create the trainer.
    # The trainer is responsible for running the training loop, and
    # logging the results.
    #
    # There are a few callbacks (which we could customize):
    # - LogPredictionSamplesCallback: Logs some examples from the dataset,
    #       and the model's predictions.
    # - ModelCheckpoint #1: Saves the latest model.
    # - ModelCheckpoint #2: Saves the best model (according to validation
    #       loss), and logs it to wandb.
    ######################################################################

    trainer = L.Trainer(
        accelerator="gpu",
        devices=cfg.resources.gpus,
        # precision="16-mixed",
        precision="32-true",
        max_epochs=cfg.training.epochs,
        logger=logger if not TESTING else False,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epochs,
        # log_every_n_steps=2, # TODO: MOVE THIS TO TRAINING CFG
        log_every_n_steps=len(datamodule.train_dataloader()),
        gradient_clip_val=cfg.training.grad_clip_norm,
        callbacks=(
            [
                # Callback which logs whatever visuals (i.e. dataset examples, preds, etc.) we want.
                # LogPredictionSamplesCallback(logger),
                # This checkpoint callback saves the latest model during training, i.e. so we can resume if it crashes.
                # It saves everything, and you can load by referencing last.ckpt.
                # CustomModelPlotsCallback(logger),
                ModelCheckpoint(
                    dirpath=cfg.lightning.checkpoint_dir,
                    filename="{epoch}-{step}",
                    monitor="step",
                    mode="max",
                    save_weights_only=False,
                    save_last=True,
                ),
                # ModelCheckpoint(
                #     dirpath=cfg.lightning.checkpoint_dir,
                #     filename="{epoch}-{step}-{val_rmse_0:.3f}",
                #     monitor="val_rmse_0",
                #     mode="min",
                #     save_weights_only=False,
                #     save_last=False,
                #     # auto_insert_metric_name=False,
                # ),
                # ModelCheckpoint(
                #     dirpath=cfg.lightning.checkpoint_dir,
                #     filename="{epoch}-{step}-{val_rmse_wta_0:.3f}",
                #     monitor="val_rmse_wta_0",
                #     mode="min",
                #     save_weights_only=False,
                #     save_last=False,
                #     # auto_insert_metric_name=False,
                # ),
                # ModelCheckpoint(
                #     dirpath=cfg.lightning.checkpoint_dir,
                #     filename="{epoch}-{step}-{val_rmse_0:.3f}",
                #     monitor="val_rmse_0",
                #     mode="min",
                #     save_weights_only=False,
                #     save_last=False,
                # )
                # This checkpoint will get saved to WandB. The Callback mechanism in lightning is poorly designed, so we have to put it last.
                # ModelCheckpoint(
                #     dirpath=cfg.lightning.checkpoint_dir,
                #     filename="{epoch}-{step}-{val_loss:.2f}-weights-only",
                #     monitor="val_loss",
                #     mode="min",
                #     save_weights_only=True,
                # ),
            ]
            if not TESTING
            else []
        ),
        fast_dev_run=5 if TESTING else False,
        num_sanity_val_steps=0,
    )

    ######################################################################
    # Train the model.
    ######################################################################

    # this might be a little too "pythonic"
    if cfg.checkpoint.run_id:
        print(
            "Attempting to resume training from checkpoint: ", cfg.checkpoint.reference
        )

        api = wandb.Api()
        artifact_dir = cfg.wandb.artifact_dir
        artifact = api.artifact(cfg.checkpoint.reference, type="model")
        ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
        # ckpt = torch.load(ckpt_file)
        # # model.load_state_dict(
        # #     {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
        # # )
        # model.load_state_dict(ckpt["state_dict"])
    else:
        print("Starting training from scratch.")
        ckpt_file = None

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_file)

    ######################################################################
    # Log additional model checkpoints to wandb.
    ######################################################################
    monitors = ["val_rmse_wta_0", "val_rmse_0"]
    model_artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
    # iterate through each file in checkpoint dir
    for file in os.listdir(cfg.lightning.checkpoint_dir):
        if file.endswith(".ckpt"):
            # check if metric name is in monitors
            metric_name = file.split("-")[-1].split("=")[0]
            if metric_name in monitors:
                # add checkpoint to artifact
                model_artifact.add_file(os.path.join(cfg.lightning.checkpoint_dir, file))
    wandb.run.log_artifact(model_artifact, aliases=["monitor"])


class ClassifierModule(L.LightningModule):
    def __init__(self, network, cfg) -> None:
        super().__init__()
        self.network = network
        self.model_cfg = cfg.model
        self.mode = cfg.mode # train or eval
        
        # mode-specific processing
        if self.mode == "train":
            self.run_cfg = cfg.training
            # training-specific params
            self.lr = self.run_cfg.lr
            self.weight_decay = self.run_cfg.weight_decay
            self.num_training_steps = self.run_cfg.num_training_steps
            self.lr_warmup_steps = self.run_cfg.lr_warmup_steps
            self.additional_train_logging_period = self.run_cfg.additional_train_logging_period
        elif self.mode == "eval":
            self.run_cfg = cfg.inference
            # inference-specific params
            self.num_trials = self.run_cfg.num_trials
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # data params
        self.batch_size = self.run_cfg.batch_size
        self.val_batch_size = self.run_cfg.val_batch_size
        # TODO: it is debatable if the module needs to know about the sample size
        self.sample_size = self.run_cfg.sample_size
        self.sample_size_anchor = self.run_cfg.sample_size_anchor

    def configure_optimizers(self):
        assert self.mode == "train", "Can only configure optimizers in training mode."
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return [optimizer], [lr_scheduler]
    
    def forward(self, batch):
        """
        Forward pass to compute training loss.
        """
        pred = self.network(batch)
        loss = F.binary_cross_entropy(pred.squeeze(1), batch['label'].float())
        return None, loss

    @torch.no_grad()
    def predict(self, batch):
        """
        Forward pass to compute predictions.
        """
        pred = self.network(batch)
        return {"pred": pred}

    def training_step(self, batch):
        """
        Training step for the module. Logs training metrics and visualizations to wandb. Since this 
        is a regression module, we sample only 1 prediction - no need to log WTA metrics.
        """
        self.train()
        _, loss = self(batch)
        #########################################################
        # logging training metrics
        #########################################################
        self.log_dict(
            {"train/loss": loss},
            add_dataloader_idx=False,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation step for the module. Logs validation metrics and visualizations to wandb. Since this 
        is a regression module, we sample only 1 prediction - still logging WTA metrics for checkpoint 
        callback.
        """
        self.eval()
        with torch.no_grad():
            # winner-take-all predictions
            pred_dict = self.predict(batch)
            val_loss = F.binary_cross_entropy(pred_dict['pred'].squeeze(1), batch['label'].float())
        ####################################################
        # logging validation wta metrics
        ####################################################
        self.log_dict(
            {
                f"val_loss_{dataloader_idx}": val_loss.mean(),
            },
            add_dataloader_idx=False,
            prog_bar=True,
        )


class PredictionClassifierDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.split = split
        self.dataset_dir = self.root / self.split
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
        # TODO: print a message when loading dataset?
        self.dataset_cfg = dataset_cfg

        # determining dataset size - if not specified, use all demos in directory once
        size = self.num_demos - (self.num_demos % 16)  # self.dataset_cfg.train_size if "train" in self.split else self.dataset_cfg.val_size
        negative_size = size

        self.size = size + negative_size
        self.positive_size = size
        self.negative_size = negative_size

        print(f"Dataset split: {self.split}, size: {self.size}")

        # # initialize tax3d model
        # tax3d_cfg, datamodule = create_datamodule(tax3d_cfg)
        # network, self.tax3d_model = create_model(tax3d_cfg)
        # ckpt_file = '/home/yingyuan/equidiff/third_party/non-rigid/scripts/logs/train_mimicgen_df_cross/2025-03-12/23-47-12/checkpoints/last.ckpt'
        # ckpt = torch.load(ckpt_file)
        # self.tax3d_model.load_state_dict(ckpt["state_dict"])
        # self.tax3d_cfg = tax3d_cfg

        # self.tax3d_model.eval()
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index, return_indices=False, use_indices=None):
        """
        Args:
            return_indices: if True, return the indices used to downsample the point clouds.
            use_indices: if not None, use these indices to downsample the point clouds. If indices are provided,
                sample_size_action and sample_size_anchor are ignored.
        """
        if index < self.positive_size:  # positive example
            item = self.get_positive_example(index, return_indices=return_indices, use_indices=use_indices)
        else:  # negative example
            item = self.get_negative_example(index - self.positive_size, return_indices=return_indices, use_indices=use_indices)
        return item
    
    def get_positive_example(self, index, return_indices=False, use_indices=None):
        # get the index of the demo file
        file_index = index % self.num_demos

        # load data
        demo = np.load(self.dataset_dir / f"demo_{file_index}.npz", allow_pickle=True)
        action_pc = torch.as_tensor(demo["action_pc"]).float()
        anchor_pc = torch.as_tensor(demo["anchor_pc"]).float()
        flow = torch.as_tensor(demo["gripper_flow"]).float()

        # initializing item dict
        item = {
        }

        # compute goal action point cloud
        goal_action_pc = action_pc + flow

        item["scene"] = anchor_pc # Anchor points in goal position
        item["pc"] = goal_action_pc # Ground-truth action points
        item["flow"] = flow # Ground-truth flow (cross-frame) to action points
        item["label"] = 1

        return item
    
    def get_negative_example(self, index, return_indices=False, use_indices=None):
        # get the index of the demo file
        file_index = index % self.num_demos

        # load data
        demo = np.load(self.dataset_dir / f"demo_{file_index}.npz", allow_pickle=True)
        action_pc = torch.as_tensor(demo["action_pc"]).float()
        anchor_pc = torch.as_tensor(demo["anchor_pc"]).float()
        flow = torch.as_tensor(demo["gripper_flow"]).float()

        # initializing item dict
        item = {
        }

        # compute goal action point cloud
        goal_action_pc = action_pc + flow

        # randomly sample a transformation
        T = random_se3(rot_sample_method='random_upright')

        # apply transformation to the action point cloud
        goal_action_pc = T.transform_points(goal_action_pc)
        noise = np.random.normal(scale=0.003, size=goal_action_pc.shape)
        goal_action_pc += noise

        # apply transformation to the flow
        flow = goal_action_pc - action_pc

        item["scene"] = anchor_pc
        item["pc"] = goal_action_pc
        item["flow"] = flow
        item["label"] = 0

        return item
    

class PredictionClassifierDataModule(L.LightningDataModule):
    def __init__(self, batch_size, val_batch_size, num_workers, dataset_cfg):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.dataset_cfg = dataset_cfg
        self.stage = None

        # setting root directory based on dataset type
        data_dir = os.path.expanduser(self.dataset_cfg.data_dir)
        self.root = Path(data_dir) / self.dataset_cfg.task
    
    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = "fit"):
        self.stage = stage

        # initializing datasets
        self.train_dataset = PredictionClassifierDataset(self.root, self.dataset_cfg, "train_tax3d")
        self.val_dataset = PredictionClassifierDataset(self.root, self.dataset_cfg, "val_tax3d")

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True, # if self.stage == "train" else False,
            num_workers=self.num_workers,
            collate_fn=cloth_collate_fn,
        )
    
    def val_dataloader(self):
        val_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=cloth_collate_fn,
        )

        return val_dataloader  
    

def cloth_collate_fn(batch):
    # batch can contain a list of dictionaries
    # we need to convert those to a dictionary of lists
    # dict_keys = ["deform_transform", "rigid_transform", "deform_params", "rigid_params"]
    dict_keys = ["deform_data", "rigid_data", "label"]
    keys = batch[0].keys()
    out = {k: None for k in keys}
    for k in keys:
        if k in dict_keys:
        #if k == "deform_params":
            out[k] = torch.tensor([item[k] for item in batch])
        else:
            out[k] = torch.stack([item[k] for item in batch])
    return out


if __name__ == "__main__":
    main()

    # dataset_cfg = {
    #     "train_size": 1000,
    #     "val_size": 100
    # }
    # dataset = PredictionClassifierDataset(root=Path("/home/yingyuan/equidiff/data/robomimic/datasets/square_d2/"),
    #                                       dataset_cfg=dataset_cfg, split="train_tax3d")
    # for i in range(1800, 1805):
    #     item = dataset[i]
    #     print(item["scene"].shape, item["pc"].shape, item["flow"].shape, item["label"])

    #     # Convert to Open3D point cloud for visualization
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(item["scene"])
    #     pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0, 0, 1]]), (1024, 1)))
    #     goal = o3d.geometry.PointCloud()
    #     goal.points = o3d.utility.Vector3dVector(item["pc"])
    #     goal.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0, 1, 0]]), (138, 1)))
    #     o3d.visualization.draw_geometries([pcd, goal])