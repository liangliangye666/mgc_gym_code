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

class L5A_2WHEEL_Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_actions = 8 # 每个环境中机器人的动作空间维度,通常对应于机器人的关节数量
        num_observations = 3 + 3 + 4 + 6 + num_actions*2  # 定义状态观测向量的维度为27,即每个环境的状态观测是一个包含27个特征值的向量
        num_privileged_obs = 3 + num_observations + 7 * 11 + 2 + 2 + 6 + 7# 特权观测,评论家网络输入
        obs_history_length = 10  # number of observations stacked together,状态观测历史堆叠的长度
        obs_history_dec = 1 # 状态历史堆叠的衰减参数,当前时刻权重最高,这里设置为1说明没有衰减

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh" # 地形网格类型:"plane-平坦地形","trimesh-三角形网格","heightfield-高度场"
        curriculum = True
        num_rows = 10  # number of terrain rows (levels),地形网格的行数(难度级别)
        num_cols = 10  # number of terrain cols (types),地形网格的列数(类型)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0] # 地形类型比例分布
        num_goals = 1

    class commands(LeggedRobotCfg.commands):
        curriculum = False  # True,是否使用课程学习
        basic_max_curriculum = 1.5 # 不同阶段的课程学习的难度值
        advanced_max_curriculum = 2 # 高级阶段的最大课程难度值
        curriculum_threshold = 0.7 # 课程学习的阈值
        resampling_time = 10.0  # time before command are changed[s],重新采样命令的时间间隔
        heading_command = True  # if true: compute ang vel command from heading error,true:根据朝向误差计算角速度命令,false:轮差速计算角速度
        num_commands = 6 # 命令的数量
        class ranges: # 定义了每个命令可以取值的范围
            lin_vel_x = [-0.6, 0.6]     # min max [m/s],线速度命令范围
            lin_vel_y = [-0.0, 0.0]
            ang_vel_yaw = [-0.5, 0.5]   # 角速度命令范围
            height = [0.643, 0.643]     # 高度范围上下浮动3cm
            # heading = [-3.14, 3.14]     # 朝向范围
            heading = [-1.0, 1.0]     # 朝向范围
            mode_normalization = [0, 1] # 模式归一化,暂时两个模式-步态占比0.8,两轮平衡占比0.2
        gait_train_proportion = 0.8     # 训练步态的环境占比
        gait_foot_height = 0.207
        gait_period = 0.5 

    class init_state(LeggedRobotCfg.init_state):

        pos = [0.0, 0.0, 0.643 + 0.05]  # [0.0, 0.0, 0.63]  # x,y,z [m]  #0.515,base初始位置
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat],base初始姿态
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0,各关节初始角度
            "left_hip_roll_joint": 0.0523599,
            "left_hip_pitch_joint": 0.261799,
            "left_knee_joint": -0.563811,
            "left_wheel_joint": 0.0,
            "right_hip_roll_joint": -0.0523599, 
            "right_hip_pitch_joint": 0.261799,   
            "right_knee_joint": -0.563811,  
            "right_wheel_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # 位置动作的缩放系数
        action_scale_pos = 0.25
        # 速度动作的缩放系数
        action_scale_vel = 0.5

        # PD控制器参数:
        # 各关节的刚度系数
        stiffness = {"hip_roll": 42, "hip_pitch": 42, "knee": 42, "wheel": 0}  # [N*m/rad]
        # 各关节的阻尼系数
        damping = {"hip_roll": 2.5, "hip_pitch": 2.5, "knee": 2.5, "wheel": 0.8}  # [N*m*s/rad]

        # 抽取率：每个策略时间步长内的控制动作更新次数
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/l5a/urdf/l5aurdf20260521.urdf" # 机器人urdf路径
        name = "l5a" # 机器人名称
        foot_name = "wheel" # 足部名称
        joint_indices = [0, 1, 2, 4, 5, 6] # 关节索引
        wheel_indices = [3, 7] # 轮子索引
        base_link_indices = [0]
        joint_link_indices = [1, 2, 3, 5, 6, 7]
        wheel_link_indices = [4, 8]
        wheel_radius = 0.127
        track_width = 0.14*2
        penalize_contacts_on = ["right_hip", "right_knee", "base", "left_hip", "left_knee"] # 惩罚区域
        terminate_after_contacts_on = ["base", "hip", "knee"] # 终止区域
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter,控制机器人自身各部分之间是否能够发生碰撞检测
        flip_visual_attachments = False # 设置是否反转视觉附件
        # stage_names = ["stand", "gait", "recover"]

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales: # 观测值缩放因子,将观测值缩放到统一维度,先给零观测值,把这些变量先读出来,然后缩放到正负1范围内
            lin_vel = 2.0
            lin_vel_y = 2.0
            ang_vel = 0.25  # 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025

            quat = 1.0
            height_measurements = 5.0  # 40.0   #5.0
            torque = 0.05

        clip_observations = 100.0 # 观测值剪裁
        clip_actions = 100.0 # 动作剪裁

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.5

        class noise_scales(LeggedRobotCfg.noise.noise_scales): # 噪声缩放因子
            dof_pos = 0.01  # 关节位置噪声
            dof_vel = 1.5  # 关节速度噪声
            lin_vel = 0.1  # 线速度噪声
            lin_vel_y = 0.1
            ang_vel = 0.2  # 角速度噪声
            gravity = 0.05
            height_measurements = 0.1  # 高度测量噪声

    class rewards(LeggedRobotCfg.rewards):
        class scales: # 奖励缩放因子
            # task related rewards
            # feet_air_time = 20
            feet_contact_number = -5
            # wheel_all_air = -10
            feet_clearance = -5
            foot_landing_vel = -5
            swing_foot_lift = 5

            # tracking related rewards
            tracking_goal = 5
            tracking_lin_vel_x = 1.5
            tracking_lin_vel_y = 1.0
            tracking_ang_vel = 3.0
            # tracking_ang_yaw = 1.0
            tracking_lin_vel_pb = 1.0
            tracking_ang_vel_pb = 1.0
            opposite_base_vel = -40
            opposite_wheel_vel = -5
            # stuck = -30

            # regulation related rewards
            # nominal_foot_position = 4.0
            leg_symmetry = 1.0
            same_foot_x_position = -2 # 0.5
            default_pos = -1
            # same_foot_z_position = -100
            lin_vel_z = -0.3
            ang_vel_xy = -0.6
            torques = -0.00016
            dof_acc = -2.5e-7
            # dof_vel = -1e-5
            action_rate = -0.01
            dof_pos_limits = -2.0
            action_smooth = -0.01
            orientation = -20.0
            feet_distance = -20
            base_height = -20
            # wheel_zero_velocity = 0.5
            wheel_spin = -20
            feet_contact_forces = -5
            collision = -50.0
            keep_balance = 0.15

        # 跟踪奖励的高斯分布参数 σ，跟踪奖励计算公式：exp(-error^2/sigma)
        tracking_sigma = 0.05
        ang_tracking_sigma = 0.1  # tracking reward = exp(-error^2/sigma)
        nominal_foot_position_tracking_sigma = 0.005
        nominal_foot_position_tracking_sigma_wrt_v = 0.5
        leg_symmetry_tracking_sigma = 0.001
        foot_x_position_sigma = 0.001
        height_tracking_sigma = 0.01
        base_height_target = 0.645
        feet_height_target = 0.10
        min_feet_distance = 0.27
        max_feet_distance = 0.30
        max_contact_force = 250.0  # forces above this value are penalized
        contact_force_scale = 100.0
        # kappa_gait_probs = 0.05
        # gait_force_sigma = 25.0
        # gait_vel_sigma = 0.25
        # gait_height_sigma = 0.005
        feet_clearance_sigma = 0.0025
        landing_height_threshold = 0.08
        landing_time_threshold = 0.12
        safe_landing_vel = 0.1
        contact_force_sigma = 40.0
    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 模拟时间步长 [秒]
        substeps = 1  # 每个时间步的子步数
        gravity = [0.0, 0.0, -9.81]  # 重力加速度 [m/s^2]
        up_axis = 1  # 选择“上轴”：0 表示 y 轴，1 表示 z 轴

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10  # 物理引擎线程数
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


class L5A_2WHEEL_CfgPPO(LeggedRobotCfgPPO):
    # PPO算法配置
    seed = 1  # 随机种子,确保实验可重复性,相同的种子会产生相同的随机行为
    runner_class_name = "OnPolicyRunner"  # 运行器类名,每次迭代都使用最新生成的数据进行训练

    class policy(LeggedRobotCfgPPO.policy): # 继承基类策略配置,覆盖特定参数
        actor_hidden_dims = [512, 256, 128] # 3层神经网络,策略网络,用于生成动作
        critic_hidden_dims = [512, 256, 128] # 3层神经网络,价值网络,评估状态价值
        activation = "elu" # 激活函数,ELU比ReLU更平滑,缓解梯度消失问题

        # only for ActorCriticSequence
        num_encoder_obs = L5A_2WHEEL_Cfg.env.obs_history_length * L5A_2WHEEL_Cfg.env.num_observations # 编码器输入维度=历史观测长度*单步观测维度
        latent_dim = 3  # at least 3 to estimate base linear velocity,基座速度空间
        encoder_hidden_dims = [256, 128] # 编码器网络层数和数量
        # encoder_hidden_dims = [128, 64]

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticSequence"  # could be ActorCritic, ActorCriticSequence,使用序列处理的actor-critic
        algorithm_class_name ="PPO" # 明确使用PPO算法
        num_steps_per_env = 48  # per iteration,每个环境每次迭代收集48步数据
        max_iterations = 100000  # number of policy updates,最大迭代次数

        # logging
        experiment_name = "l5a_2wheel"
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None  # updated from load_run and chkpt
