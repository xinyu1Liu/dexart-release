# Commands

- Train low-level policy
```
python train_dp3.py policy.type=dexart # Just the PointNet+MLP in DexArt as an Imitation Learning baseline
python train_dp3.py policy.type=dp3 # Check tax3d-conditioned-mimicgen/equi_diffpo/config/dp3.yaml
```

- Evaluation example
```
python evaluate_policy.py eval.task_name=laptop eval.checkpoint_path=data/outputs/2025.06.22/00.34.21_train_dp3_stack_d1/dp3_epoch_61.pt eval.eval_per_instance=10 eval.model=dp3 policy.pointcloud_encoder_cfg.vision_encoder_type=pn_plus_plus eval.use_test_set=True
```

- Train high-level model
```
python train_high_level.py
```
