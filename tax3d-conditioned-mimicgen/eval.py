"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from equi_diffpo.workspace.base_workspace import BaseWorkspace

max_steps = {
    'stack_d1': 400,
    'stack_three_d1': 400,
    'square_d2': 400,
    'threading_d2': 400,
    'coffee_d2': 400,
    'three_piece_assembly_d2': 500,
    'hammer_cleanup_d1': 500,
    'mug_cleanup_d1': 500,
    'kitchen_d1': 800,
    'nut_assembly_d0': 500,
    'pick_place_d0': 1000,
    'coffee_preparation_d1': 800,
    'tool_hang': 700,
    'can': 400,
    'lift': 400,
    'square': 400,
}

def get_ws_x_center(task_name):
    if task_name.startswith('kitchen_') or task_name.startswith('hammer_cleanup_'):
        return -0.2
    else:
        return 0.

def get_ws_y_center(task_name):
    return 0.

OmegaConf.register_new_resolver("get_max_steps", lambda x: max_steps[x], replace=True)
OmegaConf.register_new_resolver("get_ws_x_center", get_ws_x_center, replace=True)
OmegaConf.register_new_resolver("get_ws_y_center", get_ws_y_center, replace=True)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'equi_diffpo','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    if cfg.name == 'train_dp3': # dp3
        # output_dir = '/home/yingyuan/equidiff/data/outputs/2025.03.14/14.10.30_train_dp3_square_d2'  # square d2, oracle_42
        output_dir = '/home/yingyuan/equidiff/data/outputs/2025.04.04/21.54.05_train_dp3_square_d2'  # square d2, oracle_0
    elif cfg.name == 'equi_diff':  # equidiff
        if cfg.policy.goal_mode == 'None':
            # output_dir = '/home/yingyuan/equidiff/data/outputs/2025.03.18/21.45.24_equi_diff_square_d2'  # 1000 demos
            output_dir = '/home/yingyuan/equidiff/data/outputs/2025.03.23/14.22.16_equi_diff_square_d2'  # 100 demos
        else:
            # output_dir = '/home/yingyuan/equidiff/data/outputs/2025.03.21/16.16.11_equi_diff_square_d2'  # 1000 demos
            output_dir = '/home/yingyuan/equidiff/data/outputs/2025.03.25/16.03.00_equi_diff_square_d2'  # 100 demos
    else:
        raise NotImplementedError
    if cfg.load_ckpt_path is not None:
        output_dir = cfg.load_ckpt_path
    workspace: BaseWorkspace = cls(cfg, output_dir)
    workspace.eval()

if __name__ == "__main__":
    main()