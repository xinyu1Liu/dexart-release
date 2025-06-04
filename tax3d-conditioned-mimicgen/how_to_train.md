# Guides to run this huge codebase

## Data Generation Process
Use another MimicGen codebase with another mimicgen env environment. If using the same robomimic as in this repo, you may encounter some errors when generating new dataset from human source data.
```
conda activate mimicgen
```
Here are the steps for generating data for a new task:
1. Find your task in `~/mimicgen/mimicgen/__init__.py`.

2. Run the following to get 1100 MimicGen demos:
```
yingyuan@kirby:~/mimicgen/mimicgen/scripts$ python download_datasets.py --dataset_type source --tasks [TASK]

yingyuan@kirby:~/mimicgen$ python mimicgen/scripts/prepare_src_dataset.py --dataset datasets/source/[TASK].hdf5 --env_interface MG_[TASK] --env_interface_type robosuite

yingyuan@kirby:~/mimicgen$ python mimicgen/scripts/generate_core_configs.py

yingyuan@kirby:~/mimicgen$ python mimicgen/scripts/generate_dataset.py --config /tmp/core_configs/demo_src_[TASK].json --auto-remove-exp
```

3. Copy the generated demo into `data/robomimic/datasets/[TASK]/[TASK].hdf5`.

4. Post-process the dataset as in EquiDiff:
```
python equi_diffpo/scripts/dataset_states_to_obs.py --input data/robomimic/datasets/[TASK]/[TASK].hdf5 --output data/robomimic/datasets/[TASK]/[TASK]_voxel.hdf5 --num_workers=24
```
You will need to comment out `.cuda()` in `env_robosuite.py` for the following command:
```
python equi_diffpo/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/[TASK]/[TASK]_voxel.hdf5 -o data/robomimic/datasets/[TASK]/[TASK]_voxel_abs.hdf5 -n 12
```
Now you can use `visualize_dataset.ipynb` to visualize demos in the dataset.

5. Run the following command with both `split='train'` and `split='val'` in the file to obtain TAX3D dataset:
```
python data_reformat_mimicgen.py --dataset data/robomimic/datasets/[TASK]/[TASK]_voxel_abs.hdf5
```

## Training low-level policy
Train with unconditioned modified DP3:
```
python train.py --config-name=dp3 task_name=square_d2 n_demo=1000 policy.pointnet_type=act3d task.env_runner.goal_mode=pointcloud_oracle policy.goal_mode=None enable_wandb=False training.seed=42
```
Train with oracle-conditioned modified DP3:
```
python train.py --config-name=dp3 task_name=square_d2 n_demo=1000 policy.pointnet_type=act3d task.env_runner.goal_mode=pointcloud_oracle policy.goal_mode=pointcloud_oracle enable_wandb=False training.seed=42
```

Note: adding `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_NUM_THREADS=1` before the script limits the number of CPU used by the runner, which accelerates evaluation.

## Training TAX3D
Run the following to train a TAX3D model:
```
yingyuan@kirby:~/equidiff/third_party/non-rigid/scripts$ ./train2.sh 0 cross_point_relative disabled dataset.task=[TASK] dataset.scene_transform_type=random_flat_upright model.scene_anchor=False model.center_type=anchor_center model.rel_pose=True model.predict_ref_frame=False resources.num_workers=16 dataset.train_size=[YOUR TRAIN SIZE]
```
Note:

1. change `disabled` to `online` to enable wandb.
2. you can customize number of training data by setting `dataset.train_size`. Important: `dataset.train_size` should be multiples of 16, or the code is going to throw a bug of shape mismatch.

## Training Classifier
Run the following to train a classifier:
```
yingyuan@kirby:~/equidiff$ ./train_classifier.sh 0 cross_point_relative disabled dataset.task=[TASK] dataset.scene_transform_type=random_flat_upright model.scene_anchor=False model.center_type=anchor_center model.rel_pose=True model.predict_ref_frame=False resources.num_workers=16 dataset.train_size=[YOUR TRAIN SIZE] training.check_val_every_n_epochs=1 training.weight_decay=0.0001
```
Note:

1. change `disabled` to `online` to enable wandb.
2. you can customize number of training data by setting `dataset.train_size`. Important: `dataset.train_size` should be multiples of 16, or the code is going to throw a bug of shape mismatch.

## Evaluation
Make sure that you are using the correct model directories for the following:

1. `output_dir` in `eval.py` is set to low-level policy checkpoint directory. Another way is adding `load_ckpt_path=[YOUR PATH]` to the script.

2. `ckpt_file` in `equi_diffpo/env_runner/robomimic_image_runner.py` is set to your TAX3D model and classifier checkpoint directories.

Then you can evaluate oracle policy with:
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_NUM_THREADS=1 python eval.py --config-name=dp3 task_name=[TASK] n_demo=1000 policy.pointnet_type=act3d task.env_runner.goal_mode=pointcloud_oracle policy.goal_mode=pointcloud_oracle
```
Evaluate with TAX3D+classifier (5 samples):
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_NUM_THREADS=1 python eval.py --config-name=dp3 task_name=[TASK] n_demo=1000 policy.pointnet_type=act3d task.env_runner.goal_mode=pointcloud_tax3d policy.goal_mode=pointcloud_tax3d task.env_runner.num_samples=5
```
Evaluate with TAX3D (1 sample):
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_NUM_THREADS=1 python eval.py --config-name=dp3 task_name=[TASK] n_demo=1000 policy.pointnet_type=act3d task.env_runner.goal_mode=pointcloud_tax3d policy.goal_mode=pointcloud_tax3d task.env_runner.num_samples=1
```
Evaluate with modified DP3:
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_NUM_THREADS=1 python eval.py --config-name=dp3 task_name=[TASK] n_demo=1000 policy.pointnet_type=act3d task.env_runner.goal_mode=pointcloud_oracle policy.goal_mode=None load_ckpt_path=[PATH]
```