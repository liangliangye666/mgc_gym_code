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

from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class Y1A_2WHEEL_VMCCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 8196  # 4096
        num_actions = 8
        num_observations = 39
        num_privileged_obs = num_observations + 7 * 11 + 3 + 3 + 3 + 8 * 4 + 1 * 3 + 6
        obs_history_length = 5  # number of observations stacked together
        obs_history_dec = 1
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        dof_vel_use_pos_diff = True
        fail_to_terminal_time_s = 1

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 0.5
        dynamic_friction = 0.5
        restitution = 0.5
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
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.2, 0.2, 0.1, 0.2, 0.1]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.53]  # [0.0, 0.0, 0.63]  # x,y,z [m]  #0.515
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        default_joint_angles = {  # target angles when action = 0.0
            "left_hip_joint": 35.0 / 180.0 * 3.14,
            "left_knee_joint": -5.0 / 180.0 * 3.14,
            "lb_wheel_joint": 0.0,  # 1 后
            "lf_wheel_joint": 0.0,  # 0
            "right_hip_joint": 35.0 / 180.0 * 3.14,
            "right_knee_joint": -5.0 / 180.0 * 3.14,
            "rb_wheel_joint": 0.0,  # 3 后
            "rf_wheel_joint": 0.0,  # 2
        }

        # pos = [0.0, 0.0, 0.5]  # [0.0, 0.0, 0.63]  # x,y,z [m]  #0.515
        # rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        # default_joint_angles = {  # target angles when action = 0.0
        #     "left_hip_joint": 0,
        #     "left_knee_joint": 0,
        #     "lb_wheel_joint": 0.0,  # 1
        #     "lf_wheel_joint": 0.0,  # 0
        #     "right_hip_joint": 0,  # 0.6
        #     "right_knee_joint": 0,  # 0.2
        #     "rb_wheel_joint": 0.0,  # 3
        #     "rf_wheel_joint": 0.0,  # 2
        # }

    class commands(LeggedRobotCfg.commands):
        curriculum = False  # True
        basic_max_curriculum = 2.5
        advanced_max_curriculum = 1.5
        curriculum_threshold = 0.7
        # 4  # default: lin_vel_x, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 5
        resampling_time = 10.0  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-3.0, 3.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]
            height = [0.30, 0.47, 0.40, 0.56]  # origin [0.38, 0.48]
            mode = [1, 1]  # 0为四轮模式，1为两轮
            heading = [-3.14, 3.14]

    class control(LeggedRobotCfg.control):
        # 位置动作的缩放系数
        action_scale_pos = 0.5
        # 速度动作的缩放系数
        action_scale_vel = 10.0

        # 虚拟腿长动作的缩放系数
        action_scale_l0 = 0.1

        # 虚拟腿长的偏移量
        l0_offset = 0.175
        # 前馈力
        feedforward_force = 40.0  # [N]

        # 角度控制的比例增益
        kp_theta = 50.0  # [N*m/rad]
        # 角度控制的微分增益
        kd_theta = 3.0  # [N*m*s/rad]
        # 虚拟腿长控制的比例增益
        kp_l0 = 900.0  # [N/m]
        # 虚拟腿长控制的微分增益
        kd_l0 = 20.0  # [N*s/m]

        # PD控制器参数:
        # 各关节的刚度系数
        stiffness = {"hip": 400.0, "knee": 400.0, "wheel": 0}  # [N*m/rad]
        # 各关节的阻尼系数
        damping = {"hip": 40.0, "knee": 40.0, "wheel": 10.0}  # [N*m*s/rad]

        # 抽取率：每个策略时间步长内的控制动作更新次数
        decimation = 1

    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/y1a/urdf/y1aurdf20240523.urdf"
        name = "y1a"
        foot_name = "wheel"
        offset_x = 0.107
        offset_z = -0.098
        l1 = 0.40
        l2 = 0.25
        penalize_contacts_on = ["right_hip", "right_knee", "base", "left_hip", "left_knee"]
        terminate_after_contacts_on = ["base", "b_wheel"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class domain_rand(LeggedRobotCfg.domain_rand):

        randomize_friction = True
        friction_range = [0.1, 2.0]

        randomize_restitution = True
        restitution_range = [0.0, 1.0]

        randomize_base_mass = True
        added_mass_range = [-20.0, 20.0]

        randomize_inertia = True
        randomize_inertia_range = [0.8, 1.2]

        randomize_base_com = True
        rand_com_vec = [0.1, 0.1, 0.1]

        push_robots = True
        push_interval_s = 7
        max_push_vel_xy = 0.5  # 最大推动线速度
        max_push_ang_vel = 0.4  # 最大推动角速度

        load_robots = False  # 0602加的

        randomize_Kp = True
        randomize_Kp_range = [0.75, 1.25]
        randomize_Kd = True
        randomize_Kd_range = [0.75, 1.25]

        randomize_motor_torque = True
        randomize_motor_torque_range = [0.9, 1.1]

        randomize_default_dof_pos = False
        randomize_default_dof_pos_range = [-0.05, 0.05]

        randomize_action_delay = True
        delay_ms_range = [0, 10]

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            l0 = 5.0
            l0_dot = 0.25

            lin_vel = 2.0
            ang_vel = 0.25  # 0.25

            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025

            quat = 1.0
            height_measurements = 12.0  # 40.0   #5.0

            torque = 0.05
            mode = 1

        class priv_obs_scales:
            external_wrench = 1.0

        clip_observations = 100.0
        clip_actions = 100.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 0.6

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            l0 = 0.02
            l0_dot = 0.1
            dof_pos = 0.05  # 关节位置噪声
            dof_vel = 0.5  # 关节速度噪声
            ang_vel = 0.1  # 角速度噪声
            quat = 0.03  # 四元数噪声
            height_measurements = 0.1  # 高度测量噪声

    class rewards:
        class scales:
            # 速度跟踪
            tracking_lin_vel = 2.5  # 1.0
            tracking_lin_vel_enhance = -1.0
            tracking_ang_vel = 1.5  # 1.0
            tracking_ang_vel_enhance = -1.0  # 1.0

            nominal_state = 0

            # 参考运动跟踪
            joint_pos = 1.6
            wheel_vel = 1.5

            # 惩罚z方向线速度和xy方向旋转速度
            lin_vel_z = -1.0
            ang_vel_xy = -0.05

            # 基座位置
            base_height = 5.0  # 1.0
            base_height_enhance = 3.0  # 1.0

            # 基座姿态
            orientation = 2  # humanoid

            # 能量
            dof_vel = -0.05  # origin -5e-5  0627 -0.01
            dof_acc = -0.0001  # origin -2.5e-7
            wheel_acc = -0.0005
            torques = -0.00001  # -0.00001
            power = -0.00001  # -0.00001

            action_smooth = -0.0002  # -0.01

            collision = -1.0

            dof_pos_limits = -0.4  # -1.0
            dof_vel_limits = -0.5
            torque_limits = -0.5

            base_acc = -0.2  # 惩罚基座高加速度

            # 前后轮速差
            wheel_vel_lb_diff = -0.5  #

            wheel_contact_force = -1e-3

            hip_ff = -0.002

            leg_end_x_diff = 1

            #
            back_wheel_contact = -0.01

            vel_mismatch_exp = 0.5  # 速度不匹配指数(z方向线速度;x,y方向角速度)
            low_speed = 0.2  # 低速
            track_vel_hard = 0  # 硬速度跟踪

        # 是否只保留正奖励（true 时负的总奖励修剪为 0，防止训练中提早终止的问题）
        only_positive_rewards = False

        enhance_factor = 0.001  # 增强奖励的系数

        # 单个奖励的修剪上限值
        clip_single_reward = 2

        # 跟踪奖励的高斯分布参数 σ，跟踪奖励计算公式：exp(-error^2/sigma)
        tracking_sigma = 0.25

        # 关节位置的软限制，作为关节的百分比限制（超过此限制值的关节位置将受惩罚）
        soft_dof_pos_limit = 0.97

        # 关节速度的软限制（超过此速度限制将受到惩罚）
        soft_dof_vel_limit = 1.0

        # 扭矩的软限制（超过此扭矩限制将受到惩罚）
        soft_torque_limit = 0.8

        # 接触力
        min_wheel_contact_force = 200.0

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 模拟时间步长 [秒]
        substeps = 1  # 每个时间步的子步数
        gravity = [0.0, 0.0, -9.81]  # 重力加速度 [m/s^2]
        up_axis = 1  # 选择“上轴”：0 表示 y 轴，1 表示 z 轴

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 20  # 物理引擎线程数
            solver_type = 1  # 求解器类型：0 表示 PGS（逐次松弛法），1 表示 TGS（无约束梯度下降法）
            num_position_iterations = 4  # 位置迭代次数
            num_velocity_iterations = 0  # 速度迭代次数
            contact_offset = 0.01  # 碰撞检测距离偏移量 [米]
            rest_offset = 0.0  # 静止状态偏移量 [米]
            bounce_threshold_velocity = 0.5  # 弹跳速度阈值 [米/秒]
            max_depenetration_velocity = 1.0  # 最大分离速度 [米/秒]
            max_gpu_contact_pairs = 2**23  # GPU 支持的最大碰撞对数，默认值 2**24，适用于 8000 环境及以上
            default_buffer_size_multiplier = 5  # 默认缓冲区大小的乘数
            contact_collection = 2  # 碰撞数据收集模式：0 表示不收集，1 表示收集最后一个子步的数据，2 表示收集所有子步的数据（默认值为 2）


class Y1A_2WHEEL_VMCCfgPPO(LeggedRobotCfgPPO):
    # PPO算法配置
    seed = 5  # 随机种子
    runner_class_name = "OnPolicyRunner"  # 运行器类名

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

        # only for ActorCriticSequence
        num_encoder_obs = Y1A_2WHEEL_VMCCfg.env.obs_history_length * Y1A_2WHEEL_VMCCfg.env.num_observations
        latent_dim = 3  # at least 3 to estimate base linear velocity
        encoder_hidden_dims = [256, 128, 64]
        # encoder_hidden_dims = [128, 64]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        kl_decay = (LeggedRobotCfgPPO.algorithm.desired_kl - 0.002) / LeggedRobotCfgPPO.runner.max_iterations

    class runner(LeggedRobotCfgPPO.runner):
        # logging
        experiment_name = "y1a_2wheel_vmc"
        max_iterations = 2000  # 初始1200
        load_run = -1
        checkpoint = -1
        resume = False
