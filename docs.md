# Commands

- Train low-level policy
```
python tax3d-conditioned-mimicgen/train.py --config-name=dp3 task_name=laptop n_demo=1000 policy.pointnet_type=act3d  policy.goal_mode=None training.seed=1
```

- Train high-level model
```
python train_high_level.py
```

- Eval high-level model standalone for later viz
```
python eval_high_level.py
```

- Evaluation example
```
# without high-level
python evaluate_policy.py eval.task_name=laptop eval.checkpoint_path=data/outputs/2025.06.22/00.34.21_train_dp3_stack_d1/dp3_epoch_61.pt eval.eval_per_instance=10 eval.model=dp3 policy.pointcloud_encoder_cfg.vision_encoder_type=pn_plus_plus eval.use_test_set=True

# With high-level
python evaluate_policy.py eval.task_name=laptop eval.checkpoint_path=data/outputs/2025.07.07/01.12.31_train_dp3_stack_d1/dp3_epoch_491.pt eval.eval_per_instance=1 eval.model=dp3 policy.goal_mode=high_level eval.high_level.ckpt_file=outputs/2025-07-06/00-44-41/dp3_epoch_11.pt eval.use_test_set=true
```



- Other
```
python train_dp3.py policy.type=dexart # Just the PointNet+MLP in DexArt as an Imitation Learning baseline
```
