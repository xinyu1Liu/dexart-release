import sapien
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from copy import deepcopy
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils import common, sapien_utils
import torch

@register_agent()
class XArm6Leap(BaseAgent):
    uid = "xarm6_leap"
    urdf_path = "assets/robot/xarm6_description/xarm6_allegro_test.urdf"
    # urdf_path = "assets/robot/leap_hand/robot.urdf"  # only leap hand

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    1.56280772e-03,
                    -1.10912404e00,
                    -9.71343926e-02,
                    1.52969832e-04,
                    1.20606723e00,
                    1.66234924e-03,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]
            ),
            pose=sapien.Pose([0, 0, 0]),
        ),
        zeros=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j1=Keyframe(
            qpos=np.array([np.pi / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j2=Keyframe(
            qpos=np.array([0, np.pi / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j3=Keyframe(
            qpos=np.array([0, 0, np.pi / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j4=Keyframe(
            qpos=np.array([0, 0, 0, np.pi / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j5=Keyframe(
            qpos=np.array([0, 0, 0, 0, np.pi / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j6=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, np.pi / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
    )

    arm_joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]

    hand_joint_names = [
        "leap_0",
        "leap_1",
        "leap_2",
        "leap_3",
        "leap_4",
        "leap_5",
        "leap_6",
        "leap_7",
        "leap_8",
        "leap_9",
        "leap_10",
        "leap_11",
        "leap_12",
        "leap_13",
        "leap_14",
        "leap_15",
    ]

    arm_stiffness = [100, 100, 64, 64, 64, 40]  # 1000
    arm_damping = 3  # 1  # [50, 50, 50, 50, 50, 50]
    arm_friction = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    arm_force_limit = 100 # [50, 50, 32, 32, 32, 20]

    hand_stiffness = 3  # 3 # pgain
    hand_damping = 0.1  # 0.1 # dgain
    hand_friction = 0.01
    hand_force_limit = 0.5  # 0.5
    # armature 0.001

    ee_link_name = "palm_lower"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            friction=self.arm_friction,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
            self.arm_friction,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            self.arm_friction,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Hand
        # -------------------------------------------------------------------------- #
        hand_joint_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            None,
            None,
            self.hand_stiffness,
            self.hand_damping,
            self.hand_force_limit,
            self.hand_friction,
            normalize_action=False,
        )
        hand_joint_delta_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            -0.1,
            0.1,
            self.hand_stiffness,
            self.hand_damping,
            self.hand_force_limit,
            self.hand_friction,
            use_delta=True,
        )
        hand_joint_target_delta_pos = deepcopy(hand_joint_delta_pos)
        hand_joint_target_delta_pos.use_target = True


        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                hand=hand_joint_target_delta_pos,
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                hand=hand_joint_target_delta_pos,
            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                hand=hand_joint_target_delta_pos,
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                hand=hand_joint_target_delta_pos,
            ),
            pd_ee_pose=dict(
                arm=arm_pd_ee_pose,
                hand=hand_joint_target_delta_pos,
            ),
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos,
                hand=hand_joint_target_delta_pos,
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos,
                hand=hand_joint_target_delta_pos,
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose,
                hand=hand_joint_target_delta_pos,
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(
                arm=arm_pd_joint_vel,
                hand=hand_joint_target_delta_pos,
            ),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel,
                hand=hand_joint_target_delta_pos,
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel,
                hand=hand_joint_target_delta_pos,
            ),
        )

        return deepcopy_dict(controller_configs)
    
    # @property
    # def _sensor_configs(self):
    #     return [
    #         CameraConfig(
    #             uid="hand_camera",
    #             pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
    #             width=128,
    #             height=128,
    #             fov=np.pi / 2,
    #             near=0.01,
    #             far=100,
    #             mount=self.robot.links_map["world"],
    #         )
    #     ]

    def is_grasping(self, object:Actor, min_force=0.5, max_angle=85):
        return False
    
    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold
    
    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose