# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base_config import BaseConfig


class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        dof_vel_use_pos_diff = True
        fail_to_terminal_time_s = 0.5
        next_goal_threshold = 0.5
        reach_goal_delay = 0.1
        num_future_goal_obs = 2

    class terrain:
        horizontal_scale = 0.05  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        static_friction = 0.4
        dynamic_friction = 0.4
        restitution = 0.8
        # rough terrain only:
        measure_heights = True
        measured_points_x = [
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.0
        terrain_width = 8.0
        # trimesh only:
        slope_treshold = 0.1  # slopes above this threshold will be corrected to vertical surfaces

    class depth:
        use_camera = False

    class commands:
        curriculum = True
        basic_max_curriculum = 2.5
        advanced_max_curriculum = 1.5
        curriculum_threshold = 0.7
        num_commands = 3
        resampling_time = 5.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]  # min max [rad/s]
            height = [0.1, 0.25]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.0]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {
            "joint_a": 0.0,
            "joint_b": 0.0,
        }

    class control:
        control_type = "P"  # P: position, V: velocity, T: torques
        stiffness = {"joint_a": 10.0, "joint_b": 15.0}  # [N*m/rad]
        damping = {"joint_a": 1.0, "joint_b": 1.5}  # [N*m*s/rad]
        action_scale = 0.5
        decimation = 2

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None"  # name of the feet bodies, used to index body state and contact force tensors
        offset = 0
        l1 = 0
        l2 = 0
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = (
            True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        )
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001                     # 密度,决定质量,影响惯性矩,单位kg/m3
        angular_damping = 0.0               # 角阻尼,抑制角速度振荡(旋转阻力),无量纲
        linear_damping = 0.0                # 线阻尼,抑制线速度振荡(空气阻力),无量纲
        max_angular_velocity = 1000.0       # 最大角速度,限制数值爆炸,单位rad/s
        max_linear_velocity = 1000.0        # 最大线速度,限制数值爆炸,单位m/s
        armature = 0.0                      # 电机转动惯量,给关节增加惯性项,防止高频抖动,单位kg*m2
        thickness = 0.01                    # 厚度,碰撞检测用的形状外壳厚度,影响碰撞稳定性,单位m

    class domain_rand:
        randomize_friction = True  # 是否随机化摩擦系数
        friction_range = [0.2, 1.6]  # 摩擦系数的随机范围

        randomize_restitution = True  # 是否随机化弹性恢复系数
        restitution_range = [0.0, 1.0]  # 弹性恢复系数的随机范围

        randomize_base_mass = True  # 是否随机化机器人基座的质量
        added_mass_range = [-1.5, 1.5]  # 基座质量的随机附加范围

        randomize_inertia = True  # 是否随机化惯性矩
        randomize_inertia_range = [0.9, 1.1]  # 惯性矩的随机比例范围

        randomize_base_com = True  # 是否随机化基座的质心偏移
        rand_com_vec = [0.02, 0.03, 0.05]  # 基座质心的偏移量范围 (x, y, z)

        push_robots = True  # 是否随机推机器人
        push_interval_s = 7  # 推动的间隔时间 (秒)
        max_push_vel_xy = 1.5  # 推动力在XY平面的最大速度

        randomize_Kp = True  # 是否随机化比例控制器的增益Kp
        randomize_Kp_range = [0.8, 1.2]  # Kp随机比例范围
        randomize_Kd = True  # 是否随机化微分控制器的增益Kd
        randomize_Kd_range = [0.8, 1.2]  # Kd随机比例范围

        randomize_motor_torque = True  # 是否随机化电机的最大扭矩
        randomize_motor_torque_range = [0.8, 1.2]  # 电机扭矩的随机比例范围

        randomize_default_dof_pos = True  # 是否随机化关节的默认位置
        randomize_default_dof_pos_range = [-0.05, 0.05]  # 默认关节位置的随机偏移范围

        randomize_action_delay = True  # 是否随机化执行动作的延迟
        delay_ms_range = [0, 20]  # 动作延迟的随机范围 (毫秒)
        randomize_obs_delay = False # 是否随机化观测值反馈的延迟
        obs_delay_range = [0, 20]

        randomize_imu_offset = False
        randomize_imu_offset_range = [-1.2, 1.2]

    class rewards:
        class scales:
            keep_balance = 1.0
            collision = -50
        # 是否只保留正奖励（true 时负的总奖励修剪为 0，防止训练中提早终止的问题）
        only_positive_rewards = False
        # 奖励修剪
        clip_reward = 100
        clip_single_reward = 5
        # 关节位置的软限制，作为关节的百分比限制（超过此限制值的关节位置将受惩罚）
        soft_dof_pos_limit = 0.95
        # 关节速度的软限制（超过此速度限制将受到惩罚）
        soft_dof_vel_limit = 1.0
        # 扭矩的软限制（超过此扭矩限制将受到惩罚）
        soft_torque_limit = 0.8

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            contact_forces = 0.01
            torque = 0.05

        clip_observations = 100.0
        clip_actions = 100.0

    class noise:
        add_noise = True
        noise_level = 1.5  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.01

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [-5, -5, 3]  # [m]
        lookat = [0, 0, 0]  # [m]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 0.5
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-4  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.005
        max_grad_norm = 1.0

        extra_learning_rate = 1e-3

    class runner:
        policy_class_name = "ActorCriticSequence"  # could be ActorCritic, ActorCriticSequence
        algorithm_class_name = "PPO"
        num_steps_per_env = 48  # per iteration
        max_iterations = 5000  # number of policy updates

        # logging
        save_interval = 500  # check for potential saves every this many iterations
        experiment_name = "test"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
