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

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.envs.base.legged_robot import LeggedRobot
from wheel_legged_gym.utils.terrain import Terrain
from wheel_legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
    get_euler_zyx_tensor,
    quat_from_euler_zyx,
)
from wheel_legged_gym.utils.helpers import class_to_dict
from .l5a_2wheel_config import L5A_2WHEEL_Cfg


class L5A_2WHEEL(LeggedRobot):
    def __init__(self, cfg: L5A_2WHEEL_Cfg, sim_params, physics_engine, sim_device, headless):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg # 获取训练环境配置参数
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum: # 启用地形课程学习
            self._update_terrain_curriculum(env_ids) # 调整地形复杂度与命令
            if self.cfg.commands.curriculum: # 根据超时环境调整命令生成难度
                time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten() # 找到所有超时的环境,超时环境不一定表示失败,而是可能表示机器人已经掌握了当前难度水平,需要更大挑战
                self.update_command_curriculum(time_out_env_ids) # 调整命令
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0): # 每个回合更新一次
            self.update_command_curriculum(env_ids)

        self._update_env_goals(env_ids) # 刷新环境目标
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0
        self._update_goals()
        self._update_env_step_heights(env_ids)

        # reset robot states
        self._reset_dofs(env_ids) # 重置关节状态
        self._reset_root_states(env_ids) # 重置本体位置/姿态
        self._resample_commands(env_ids) # 为每个环境生成新任务指令

        # reset buffers
        self.last_actions[env_ids] = 0.0 # 运动历史清零
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0     
        self.episode_length_buf[env_ids] = 0.0 # 回合计数器重置,统计当前回合已进行的步数,回合有终止或失败条件,然后重置时该变量刷新重置
        self.envs_steps_buf[env_ids] = 0 # 环境从创建以来累计的总步数,用于课程学习,固定步数周期做一次调整等地方
        self.reset_buf[env_ids] = 1
        self.obs_history[env_ids] = 0 
        obs_buf = self.compute_proprioception_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length) # 用当前观测值重置历史观测
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        self.dof_pos_int[env_ids] = 0
        self.last_contacts[env_ids] = 0.0
        self.contact_history[env_ids] = 0.0
        self.current_swing[env_ids] = 0.0
        self.swing_time[env_ids] = 0.0
        self.has_swing[env_ids] = False
        self.swing_mask[env_ids] = 0                            # 摆动腿
        self.stance_mask[env_ids] = 1.0                         # 两条腿都重置为支撑腿
        self.in_double_support[env_ids] = True                        # 双支撑
        self.ds_time[env_ids] = 0.0                                # 双支撑时间

        # fill extras
        self.extras["episode"] = {} # 创建一个空字典用于存储当前回合的统计信息
        for key in self.episode_sums.keys(): # 遍历所有奖励累加器
            self.extras["episode"]["rew_" + key] = ( # 计算标准化平均奖励
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0 # 计算env_ids环境下key奖励的平均奖励
        # log additional curriculum info
        if self.cfg.terrain.curriculum: # 地形课程信息记录
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum: # 命令课程信息记录
            self.extras["episode"]["a_flat_max_command_x"] = torch.mean( # 记录在平地上允许的最大线速度命令
                self.command_ranges["lin_vel_x"][self.flat_idx, 1].float()
            )
        # 记录各种非平地形的最大线速度命令(x方向)
        if self.cfg.terrain.curriculum and self.cfg.commands.curriculum: # 当两种课程同时启用时的地形分类命令难度
            self.extras["episode"]["a_smooth_slope_max_command_x"] = torch.mean( # 平滑斜坡地形
                self.command_ranges["lin_vel_x"][self.smooth_slope_idx, 1].float()
            )
            self.extras["episode"]["a_rough_slope_max_command_x"] = torch.mean( # 粗糙斜坡地形
                self.command_ranges["lin_vel_x"][self.rough_slope_idx, 1].float()
            )
            self.extras["episode"]["a_stair_up_max_command_x"] = torch.mean( # 上楼梯地形
                self.command_ranges["lin_vel_x"][self.stair_up_idx, 1].float()
            )
            self.extras["episode"]["a_stair_down_max_command_x"] = torch.mean( # 下楼梯地形
                self.command_ranges["lin_vel_x"][self.stair_down_idx, 1].float()
            )
            self.extras["episode"]["a_discrete_max_command_x"] = torch.mean( # 离散障碍地形
                self.command_ranges["lin_vel_x"][self.discrete_idx, 1].float()
            )
        # send timeout info to the algorithm,发送超时信息给强化学习算法
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # print("actions:", self.actions[0])
        # self.actions = torch.zeros( # 初始化动作张量
        #     self.num_envs,
        #     self.num_actions,
        #     dtype=torch.float,
        #     device=self.device,
        #     requires_grad=False,
        # )
        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            self.envs_steps_buf += 1
            self.action_fifo = torch.cat((self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1)
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
        )

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # Extract joint position and wheel velocity references
        self.joint_pos_ref = actions[:, self.joint_indices] * self.cfg.control.action_scale_pos # 获取关节位置参考
        self.wheel_vel_ref = actions[:, self.wheel_indices] * self.cfg.control.action_scale_vel # 获取轮子速度参考

        # Compute joint and wheel torques
        torque_joint = self.p_gains[:, self.joint_indices] * ( # 计算关节力矩
            self.joint_pos_ref + self.default_dof_pos[:, self.joint_indices] - self.dof_pos[:, self.joint_indices]
        ) + self.d_gains[:, self.joint_indices] * (-self.dof_vel[:, self.joint_indices])

        torque_wheel = self.d_gains[:, self.wheel_indices] * (self.wheel_vel_ref - self.dof_vel[:, self.wheel_indices]) # 轮子力矩

        # Combine torques while preserving the original order
        torques = torch.zeros_like(actions)  # Initialize with the same shape as actions
        torques[:, self.joint_indices] = torque_joint
        torques[:, self.wheel_indices] = torque_wheel

        return torch.clip(torques * self.torques_scale, -self.torque_limits, self.torque_limits) # 剪裁后输出力矩
    
    def compute_proprioception_observations(self):
        # note that observation noise need to modified accordingly !!!
        # gait_flag = self.gait_enable.unsqueeze(1)
        # sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1) * gait_flag
        # cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1) * gait_flag
        dof_pos = (self.dof_pos - self.default_dof_pos)[:,self.joint_indices]
        obs_buf = torch.cat(
            (
                # self.base_lin_vel * self.obs_scales.lin_vel, # 3, 机器人base线速度
                self.base_ang_vel * self.obs_scales.ang_vel, # 3 ,机器人base角速度(在base坐标系)
                self.projected_gravity, # 3 ,重力投影方向
                self.commands[:, :4] * self.commands_scale,  # 4 , 外界命令
                dof_pos * self.obs_scales.dof_pos,  # 6 ,机器人关节位置,左边髋膝关节
                self.dof_vel * self.obs_scales.dof_vel,  # 8 , 8个关节速度
                self.actions,  # 8 ,8个关节输出(上一时刻)
                # gait_flag,
                # sin_phase,
                # cos_phase,
                # self.leg_swing_first.unsqueeze(1),
            ),
            dim=-1,
        )
        return obs_buf
    
    def compute_privileged_observations(self):
        heights = (
            torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1,
                1.0,
            )
            * self.obs_scales.height_measurements
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        cur_goal = self.env_goals[env_ids, self.cur_goal_idx, :2]
        privileged_obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.obs_buf,
                heights,
                cur_goal[:, :2],
                target_vec_norm,
                self.contact_forces[:, self.feet_indices, :].flatten(1),   # 6
                self.contact_history.flatten(1),  # 6
            ),
            dim=-1,
        )
        return privileged_obs_buf

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # old_cmd = self.commands.clone()               # 拷贝旧命令
        cmd_vel_x = (
            self.command_ranges["lin_vel_x"][env_ids, 1]
            - self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_x"
        ][env_ids, 0]
        cmd_vel_x[torch.abs(cmd_vel_x) < 0.1] = 0.0
        self.commands[env_ids, 0] = cmd_vel_x
        high_env_mask = self.env_step_height[env_ids] > 0.1
        cmd_vel_x[high_env_mask] = torch.abs(cmd_vel_x[high_env_mask]) # 台阶高度大于10，那么只给向前的命令

        cmd_vel_y = (
            self.command_ranges["lin_vel_y"][env_ids, 1]
            - self.command_ranges["lin_vel_y"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_y"
        ][env_ids, 0]
        cmd_vel_y[torch.abs(cmd_vel_y) < 0.1] = 0.0
        self.commands[env_ids, 1] = cmd_vel_y

        self.commands[env_ids, 3] = (
            self.command_ranges["height"][env_ids, 1]
            - self.command_ranges["height"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "height"
        ][env_ids, 0]     

        if self.cfg.commands.heading_command: # 如果有朝向命令,那么在第五个命令处随机化一个朝向值
            self.commands[env_ids, 4] = (
                self.command_ranges["heading"][env_ids, 1]
                - self.command_ranges["heading"][env_ids, 0]
            ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
                "heading"
            ][env_ids, 0]   
        else:
            self.commands[env_ids, 2] = (
                self.command_ranges["ang_vel_yaw"][env_ids, 1]
                - self.command_ranges["ang_vel_yaw"][env_ids, 0]
            ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
                "ang_vel_yaw"
            ][env_ids, 0]  

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0]) # 创建一个与当前观测缓冲区第一个环境中的观测数目相同形状的全零张量
        self.add_noise = self.cfg.noise.add_noise # 添加噪声标志位,true
        noise_scales = self.cfg.noise.noise_scales # 各观测组成部分的噪声缩放系数
        noise_level = self.cfg.noise.noise_level # 全局噪声水平系数
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel # 最终角速度噪声系数=角速度噪声系数*全局噪声水平*角速度观测缩放系数
        noise_vec[3:6] = noise_scales.gravity * noise_level # 重力噪声系数
        noise_vec[6:8] = 0.0  # commands # 命令无噪声
        noise_vec[8:14] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos # 各自由度位置噪声系数
        noise_vec[14:20] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel # 各自由度速度噪声系数
        noise_vec[20:] = 0.0  # previous actions,无历史动作噪声
        return noise_vec
    
    # ----------------------------------------
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        super()._init_buffers()
        self.total_learning_iteration = 0 # 当前总迭代次数
        self.joint_indices = torch.tensor(list(self.cfg.asset.joint_indices), device=self.device) # 关节索引
        self.wheel_indices = torch.tensor(list(self.cfg.asset.wheel_indices), device=self.device) # 轮子索引
        self.base_link_indices = torch.tensor(list(self.cfg.asset.base_link_indices), device=self.device) # base_link索引
        self.joint_link_indices = torch.tensor(list(self.cfg.asset.joint_link_indices), device=self.device)  # joint_link索引
        self.wheel_link_indices = torch.tensor(list(self.cfg.asset.wheel_link_indices), device=self.device)   # wheel_link索引
        self.wheel_lin_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.wheel_ang_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.rwd_linVelTrackPrev = torch.zeros(self.num_envs, device=self.device)
        self.rwd_angVelTrackPrev = torch.zeros(self.num_envs, device=self.device)
        self.rwd_linVelTrackEnhancedPrev = torch.zeros(self.num_envs, device=self.device)
        self.rwd_angVelTrackEnhancedPrev = torch.zeros(self.num_envs, device=self.device)
        self.contact_history = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device, dtype=torch.float)
        self.current_swing = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.swing_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)                             # 摆动时间
        self.swing_mask = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)                          # 摆动腿
        self.stance_mask = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)                         # 支撑腿
        self.has_swing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.in_double_support = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)                        # 双支撑
        self.ds_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)                                # 双支撑时间

    # def _get_swing_stance_mask(self):
    #     # ===== 1. 接触检测 =====
    #     force_xy = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
    #     force_xy_local = torch.zeros_like(force_xy)
    #     for i in range(len(self.feet_indices)):
    #         force_xy_local[:, i, :] = quat_rotate_inverse(self.base_quat, self.contact_forces[:, self.feet_indices, :])
    #     force_x = force_xy_local[:, i, 0]
    #     wheel_world_vel = self.rigid_body_vel[:, self.wheel_link_indices, :3]


    #     vel_cmd = self.commands[:, 0]
    #     tirgger = (~vel_cmd.unsqueeze(1)) & (force_xy_local[:, :, 0] > 0)
    #     contact = force_xy > 20.0
    #     # print("force:", contact[0])
    #     stable = (self.contact_history.sum(dim=2) >= 2) & (tirgger)
    #     # 是否需要抬腿：这里建议用 stable，而不是瞬时 contact
    #     need_swing = stable.sum(dim=1) > 0
    #     # ===== 2. 选候选腿 =====
    #     candidate = torch.argmax(force_xy, dim=1)
    #     one_contact = (contact.sum(dim=1) == 1)
    #     if one_contact.any():
    #         candidate[one_contact] = torch.argmax(contact[one_contact].float(), dim=1)
    #     one_stable = (stable.sum(dim=1) == 1)
    #     mask = (contact.sum(dim=1) == 2) & one_stable
    #     if mask.any():
    #         candidate[mask] = torch.argmax(stable[mask].float(), dim=1)
    #     # ===== 3. swing状态机：去掉双支撑等待 =====
    #     swing_duration = 0.2
    #     start_swing = (~self.has_swing) & need_swing
    #     if start_swing.any():
    #         self.current_swing[start_swing] = candidate[start_swing]
    #         self.has_swing[start_swing] = True
    #         self.swing_time[start_swing] = 0.0
    #     in_swing = self.has_swing
    #     self.swing_time[in_swing] += self.dt
    #     swing_finished = in_swing & (self.swing_time >= swing_duration)
    #     if swing_finished.any():
    #         self.has_swing[swing_finished] = False
    #         self.swing_time[swing_finished] = 0.0
    #     # ===== 4. 输出 =====
    #     self.swing_mask.zero_()
    #     self.swing_mask[torch.arange(self.num_envs), self.current_swing] = self.has_swing.float()
    #     self.stance_mask = 1.0 - self.swing_mask
    #     self.stance_mask[~self.has_swing] = 1.0

    def _get_swing_stance_mask(self):
        # ===== 1. 接触检测 =====
        force_world = self.contact_forces[:, self.feet_indices, :]      # [N, 2, 3]
        force_base = torch.zeros_like(force_world)

        for i in range(len(self.feet_indices)):
            force_base[:, i, :] = quat_rotate_inverse(
                self.base_quat,
                force_world[:, i, :]
            )

        force_x = force_base[:, :, 0]                                   # [N, 2]
        # print("force:", force_x[0])
        vel_cmd = self.commands[:, 0].unsqueeze(1)
        vel_threshold = 0
        force_threshold = 10.0

        # 轮子向前运动，受到向后的水平力
        forward_blocked = (vel_cmd > vel_threshold) & (force_x > force_threshold)

        # 轮子向后运动，受到向前的水平力
        backward_blocked = (vel_cmd < -vel_threshold) & (force_x < -force_threshold)

        no_trigger = forward_blocked | backward_blocked                    # [N, 2]
        # print("no_trigger:", no_trigger[0])

        # ===== 2. 稳定触发 =====
        # 如果你只想瞬时触发，用这个：
        # stable = trigger

        # 如果你还想保留原来的接触历史抗抖，用这个：
        contact_history_stable = self.contact_history.sum(dim=2) >= 3
        # print("contact_history:", contact_history_stable[0])
        stable = (~no_trigger) & contact_history_stable
        # print("stable:", stable[0])

        need_swing = stable.sum(dim=1) > 0

        # ===== 3. 选候选腿 =====
        candidate = torch.argmax(torch.abs(force_x), dim=1)

        one_stable = stable.sum(dim=1) == 1
        if one_stable.any():
            candidate[one_stable] = torch.argmax(stable[one_stable].float(), dim=1)

        # ===== 4. swing 状态机 =====
        swing_duration = 0.05

        start_swing = (~self.has_swing) & need_swing

        if start_swing.any():
            self.current_swing[start_swing] = candidate[start_swing]
            self.has_swing[start_swing] = True
            self.swing_time[start_swing] = 0.0

        in_swing = self.has_swing
        self.swing_time[in_swing] += self.dt

        swing_finished = in_swing & (self.swing_time >= swing_duration)
        if swing_finished.any():
            self.has_swing[swing_finished] = False
            self.swing_time[swing_finished] = 0.0

        # ===== 5. 输出 mask =====
        self.swing_mask.zero_()
        self.swing_mask[torch.arange(self.num_envs), self.current_swing] = self.has_swing.float()
        # print("swing:", self.swing_mask[0])
        self.stance_mask = 1.0 - self.swing_mask
        self.stance_mask[~self.has_swing] = 1.0

    # def _get_swing_stance_mask(self):
    #     # ===== 1. 接触检测 =====
    #     force_xy = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
    #     # print("contact_force:", force_xy[0])
    #     contact = force_xy > 20.0
    #     stable = (self.contact_history.sum(dim=2) >= 2)     # 稳定接触（推荐 >=2 更鲁棒）
    #     need_swing = contact.sum(dim=1) > 0                 # 是否需要抬腿（平地不抬）
    #     # ===== 2. 选候选腿 =====
    #     candidate = torch.argmax(force_xy, dim=1)           # 默认：接触力大的,return:0-left_leg,1-right_leg
    #     one_contact = (contact.sum(dim=1) == 1)             # 如果单脚接触
    #     if one_contact.any():
    #         candidate[one_contact] = torch.argmax(contact[one_contact].float(), dim=1) # 抬单脚接触的那只脚
    #     one_stable = (stable.sum(dim=1) == 1)               # 如果双脚接触 & 单脚稳定接触,选稳定接触的脚
    #     mask = (contact.sum(dim=1) == 2) & one_stable
    #     if mask.any():
    #         candidate[mask] = torch.argmax(stable[mask].float(), dim=1)

    #     # ===== 3. swing状态机 =====
    #     swing_duration = 0.2                                # 摆动周期
    #     ds_duration = 0.1                                   # 双支撑时间
    #     in_ds = self.in_double_support
    #     self.ds_time[in_ds] += self.dt
    #     ds_finished = in_ds & (self.ds_time >= ds_duration) & need_swing     # 双支撑结束,开始摆动
    #     if ds_finished.any():
    #         self.current_swing[ds_finished] = candidate[ds_finished]        # 当前摆动腿索引
    #         self.has_swing[ds_finished] = True                              # 有摆动腿的环境
    #         self.in_double_support[ds_finished] = False                     # 无双腿支撑的环境
    #         self.swing_time[ds_finished] = 0.0                              # 摆动时间重置
    #         self.ds_time[ds_finished] = 0.0                                 # 双腿支撑时间重置
        
    #     in_swing = self.has_swing                                           # 有摆动腿的环境
    #     self.swing_time[in_swing] += self.dt                                # 摆动时间增加
    #     swing_finished = in_swing &  (self.swing_time >= swing_duration)     # 摆动结束,开始双支撑
    #     if swing_finished.any():
    #         self.has_swing[swing_finished] = False                          # 摆动结束
    #         self.in_double_support[swing_finished] = True                   # 进入双支撑状态
    #         self.swing_time[swing_finished] = 0.0                           # 摆动时间重置
    #         self.ds_time[swing_finished] = 0.0                              # 双支撑时间重置

    #     # ===== 4. 输出 =====
    #     self.swing_mask.zero_()
    #     self.swing_mask[torch.arange(self.num_envs), self.current_swing] = self.has_swing.float()
    #     self.stance_mask = 1.0 - self.swing_mask
    #     self.stance_mask[self.in_double_support] = 1.0

    def _update_contact_history(self):
        force_xy = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
        contact = (force_xy > 10.0).float()
        # 滑动窗口
        self.contact_history = torch.roll(self.contact_history, shifts=-1, dims=2)
        self.contact_history[:, :, -1] = contact

    def post_physics_step(self):
        super().post_physics_step()
        # # 更新控制状态机所在阶段
        self._update_goals()

    def _update_env_step_heights(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.env_step_height[env_ids] = self.terrain_step_height[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _update_env_goals(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        temp = self.terrain_goals[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[env_ids] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)

    def _update_goals(self):
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt                                                     # 达到目标时长判断
        self.cur_goal_idx[next_flag] += 1                                                                                               # 移动到下一个目标点
        # 限制在真实goal范围
        max_goal = self.cfg.terrain.num_goals - 1
        self.cur_goal_idx = torch.clamp(self.cur_goal_idx, max=max_goal)
        self.reach_goal_timer[next_flag] = 0                                                                              # 重置到达目标计时器
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reached_goal_ids = torch.norm(self.root_states[:, :2] - self.env_goals[env_ids, self.cur_goal_idx, :2], dim=1) < self.cfg.env.next_goal_threshold   # 目标到达判断
        self.reach_goal_timer[self.reached_goal_ids] += 1
        self._compute_current_goal()

    def _compute_current_goal(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.target_pos_rel = self.env_goals[env_ids, self.cur_goal_idx, :2] - self.root_states[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]                                                     # 下一目标位置
        self.target_pos_rel_3d = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_pos_rel_3d[:, :2] = self.target_pos_rel
        self.target_pos_rel_local = quat_rotate_inverse(self.base_quat, self.target_pos_rel_3d)[:, :2]
        # print("loacl_pos_rel:", self.target_pos_rel_local[0])
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)                                                               # 下一目标向量
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])                       # 下一目标方向

    # ------------ reward functions----------------
    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.abs(self.foot_positions[:, 0, 1] - self.foot_positions[:, 1, 1])
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1) + \
                 torch.clip(feet_distance - self.cfg.rewards.max_feet_distance, 0, 1)
        # print("feet_distance:", feet_distance[0])
        # print("rew:", -50*reward[0])
        return reward

    def _reward_nominal_foot_position(self):
        #1. calculate foot postion wrt base in base frame  
        nominal_base_height = -(self.cfg.rewards.base_height_target- self.cfg.asset.wheel_radius)
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        reward = 0
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
            height_error = nominal_base_height - foot_positions_base[:, i, 2]
            reward += torch.exp(-(height_error ** 2)/ self.cfg.rewards.nominal_foot_position_tracking_sigma)
        vel_cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        return reward / len(self.feet_indices)*torch.exp(-(vel_cmd_norm ** 2)/self.cfg.rewards.nominal_foot_position_tracking_sigma_wrt_v)
    
    def _reward_same_foot_z_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_z_position_err = foot_positions_base[:,0,2] - foot_positions_base[:,1,2]
        return foot_z_position_err ** 2

    def _reward_leg_symmetry(self):
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        leg_symmetry_err = (abs(foot_positions_base[:,0,1])-abs(foot_positions_base[:,1,1]))
        return torch.exp(-(leg_symmetry_err ** 2)/ self.cfg.rewards.leg_symmetry_tracking_sigma)

    def _reward_same_foot_x_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_x_position_err = foot_positions_base[:,0,0] - foot_positions_base[:,1,0]
        # reward = torch.exp(-(foot_x_position_err ** 2)/ self.cfg.rewards.foot_x_position_sigma)
        reward = torch.abs(foot_x_position_err)
        return reward

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return reward

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel[:, self.joint_indices]), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1)

    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1)
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        ans = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        # print("err:", lin_vel_error[0])
        # print("ans:", ans[0])
        return ans

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_pb(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_lin_vel_x() - self.rwd_linVelTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ans = torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)
        # print("cmd:", self.commands[0, 2])
        # print("act:", self.base_ang_vel[0, 2])
        # print("ans:", ans[0])
        return ans

    def _reward_tracking_ang_vel_pb(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_ang_vel() - self.rwd_angVelTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt
    
    def _reward_tracking_ang_yaw(self):
        # Tracking of angular velocity commands (yaw)
        ang_error = torch.clip(wrap_to_pi(self.commands[:, 2] - self.yaw), -1.0, 1.0)
        ans = torch.exp(-ang_error**2 / self.cfg.rewards.ang_tracking_sigma)
        # print("cmd:", self.commands[0, 2])
        # print("act:", self.yaw[0])
        # print("ans:", ans[0])
        return ans
    
    def _reward_opposite_base_vel(self):
        """
        惩罚 base 在指令反方向运动
        公式：max(0, -sgn(v_cmd) * v_x) * -40
        """
        v_cmd = self.commands[:, 0]   # x方向指令速度
        v_x = self.base_lin_vel[:, 0] # base x方向实际速度
        direction = torch.sign(v_cmd)
        opposite_vel = torch.relu(-direction * v_x)
        return opposite_vel
    
    def _reward_opposite_wheel_vel(self):
        """
        惩罚轮子在反方向旋转
        公式：sum(max(0, -sgn(v_cmd) * wheel_vel)) * -2
        """
        v_cmd = self.commands[:, 0]  # x方向速度指令
        wheel_vel = self.dof_vel[:, self.wheel_indices]  # (num_envs, 2)
        direction = torch.sign(v_cmd)
        # (num_envs, 2)
        opposite_wheel_vel = torch.relu(-direction.unsqueeze(-1) * wheel_vel)
        penalty = torch.sum(opposite_wheel_vel, dim=1)
        return penalty
    
    def _reward_stuck(self):
        v_cmd = self.commands[:, 0]
        v_base = self.base_lin_vel[:, 0]
        cmd_strength = torch.relu(torch.abs(v_cmd) - 0.3)
        stuck_level = torch.relu(0.1 - torch.abs(v_base))
        penalty = cmd_strength * stuck_level
        return penalty
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.abs(base_height - self.cfg.rewards.base_height_target)

    def _reward_tracking_goal(self):
        current_pos = self.root_states[:, :2]
        # print("cur_pos:", current_pos[0])
        env_ids = torch.arange(self.num_envs, device=self.device)
        target_pos = self.env_goals[env_ids, self.cur_goal_idx, :2]
        # print("target_pos:", target_pos[0])
        direction = target_pos - current_pos
        direction_norm = torch.norm(direction, dim=-1, keepdim=True) + 1e-6
        direction = direction / direction_norm
        vel = self.root_states[:, 7:9]
        cmd = self.commands[:, 0]
        sign = torch.sign(cmd)
        progress = self.total_learning_iteration / 10000
        decay = max(1.0-progress, 0.0)
        rew = sign * torch.sum(vel * direction, dim=-1) * decay
        # cmd_no_zero = (torch.abs(self.commands[:, 0]) > 1e-3).float()
        # print("rew:", sign[0] * torch.sum(vel * direction, dim=-1)[0])
        return rew

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        # print("last:", self.last_contacts[0])
        # print("now:", contact[0])
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        air_time_clamped = torch.clamp(self.feet_air_time, max=0.5)
        rew_airTime = torch.sum(air_time_clamped * first_contact, dim=1) # reward only on first contact with the ground
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_wheel_all_air(self):
        wheel_air = self.contact_forces[:, self.feet_indices, 2] < 1.
        wheel_all_air = torch.all(wheel_air, dim=1)
        return wheel_all_air
    
    # def _reward_feet_contact_number(self):
    #     fz = self.contact_forces[:, self.feet_indices, 2]
    #     contact_sigma = self.cfg.rewards.contact_force_sigma
    #     soft_contact = 1.0 - torch.exp(-torch.clamp(fz, min=0.0) / contact_sigma)
    #     target_height = torch.clamp(self.env_step_height.unsqueeze(1), min=0.02)
    #     clearance = self.foot_heights
    #     lift_ratio = torch.clamp(clearance / (target_height + 1e-6), 0.0, 1.0)
    #     # Swing foot contact is bad mainly before it has lifted enough.
    #     swing_contact_penalty = self.swing_mask * soft_contact
    #     return torch.sum(swing_contact_penalty, dim=1)

    def _reward_feet_contact_number(self): 
        fz = self.contact_forces[:, self.feet_indices, 2]                   # 接触力z,维度[N, 2]
        contact = (fz > 1.0).float()                                        # 是否接触
        # print("phase:", self.leg_phase[0])
        is_stance = self.stance_mask
        is_swing = self.swing_mask
        rew_contact = is_stance*contact - is_swing*contact # 摆动腿不接触,摆动腿不接触
        rew_gait = torch.mean(rew_contact, dim=1)                            # 步态奖励
        return rew_gait
    
    def _reward_feet_clearance(self):
        foot_height = self.foot_positions[:, :, 2]
        terrain_height = self._get_foot_heights()
        clearance = torch.clamp(
            foot_height - terrain_height - self.cfg.asset.wheel_radius,
            min=0.0,
        )
        swing_mask = self.swing_mask
        # print("swing:", swing_mask[0])
        # target_height = self.cfg.rewards.feet_height_target
        target_height = self.env_step_height.unsqueeze(1)
        sigma = self.cfg.rewards.feet_clearance_sigma
        # print("step_height:", self.env_step_height[0])
        # print("env:", self.terrain_levels[0], self.terrain_types[0])
        height_error = torch.clamp(target_height - clearance, min=0.0)
        penalty = 1.0 - torch.exp(-torch.abs(height_error) / sigma)
        h = penalty * swing_mask
        # print("pen:", penalty[0])
        # print("ans:", h[0])
        ans = torch.sum(penalty * swing_mask, dim=1)
        return ans
    
    def _reward_swing_foot_lift(self):
        swing_mask = self.swing_mask
        target_height = self.env_step_height.unsqueeze(1)
        target_height = torch.clamp(target_height, min=0.03)
        clearance = self.foot_heights
        normalized_lift = torch.clamp(clearance / (target_height + 1e-6), 0.0, 1.0)
        lift_reward = torch.sum(normalized_lift * swing_mask, dim=1)
        progress = self.total_learning_iteration / 20000
        decay = max(1.0-progress, 0.0)
        return lift_reward * decay
    
    def _reward_wheel_zero_velocity(self):
        swing_mask = self.swing_mask
        wheel_vel = self.dof_vel[:, self.wheel_indices]                             # 轮子速度
        air_wheel_wheel = torch.square(wheel_vel)                                   # 步态时轮子的速度
        wheel_vel = torch.sum(air_wheel_wheel * swing_mask, dim=1)                               # 两个轮子的速度和
        rew_wheel_vel = torch.exp(-wheel_vel) 
        return rew_wheel_vel
    
    def _reward_wheel_spin(self):
        """
        使用世界坐标系下的轮子速度计算打滑
        """
        # 获取世界坐标系下的轮子速度
        # 直接从 rigid_body_vel 获取，不转换到base系
        wheel_world_vel = self.rigid_body_vel[:, self.wheel_link_indices, :]  # 形状: (num_envs, 2, 3)
        # 轮子角速度
        wheel_vel = self.dof_vel[:, self.wheel_indices]  # 关节角速度
        # 轮子理论线速度（世界坐标系中的期望前进方向）
        wheel_lin_vel = torch.abs(wheel_vel * self.cfg.asset.wheel_radius)
        # 轮子实际水平速度（世界坐标系）
        # 注意：我们只关心水平面内的打滑
        wheel_actual_hor_vel = torch.abs(wheel_world_vel[:, :, 0])  # x-y平面速度模长
        # 论文公式
        slip_condition = 0.8 * wheel_lin_vel - wheel_actual_hor_vel - 0.1
        slip_penalty = torch.relu(slip_condition)
        # 对两个轮子求和
        total_slip = torch.sum(slip_penalty, dim=1)
        # print("rew_wheel_spin:", total_slip[0])
        return total_slip
    
    def _reward_feet_contact_forces(self):
        """
        惩罚足部垂直接触力超过阈值的情况
        公式：max(0, F_z - F_max) × -5.0
        """
        # 获取足部垂直接触力
        # self.contact_forces 形状: (num_envs, num_bodies, 3)
        feet_z_forces = self.contact_forces[:, self.feet_indices, 2]  # Z轴力，形状: (num_envs, 2)
        # print("contact:", self.contact_forces[0, self.feet_indices])
        # print("feet_z_forces:", feet_z_forces[0])
        # 设置最大允许力阈值（单位：牛顿）
        F_max = self.cfg.rewards.max_contact_force  # 例如50N，需要根据你的机器人调整
        F_scale = self.cfg.rewards.contact_force_scale
        # 计算超出阈值的部分
        excess_force = torch.relu(feet_z_forces - F_max)  # relu等同于max(0, x)
        normalized_excess = torch.clamp(excess_force / F_scale, 0.0, 2.0)
        # 对两个足部求和，然后乘以惩罚系数
        penalty = torch.sum(normalized_excess, dim=1)  # 对两个足部求和
        return penalty

    def _reward_default_pos(self):
        hip_error = torch.sum(torch.square(self.dof_pos[:, self.cfg.asset.joint_indices] - self.default_dof_pos[:, self.cfg.asset.joint_indices]), dim=1)
        rew_hip = hip_error
        return rew_hip
    
    def _reward_foot_landing_vel(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 1.0

        down_vel = torch.clamp(-self.foot_velocities[:, :, 2], min=0.0)
        # print("vel:", self.foot_velocities[0, :, 2])
        time_to_contact = self.foot_heights / (down_vel + 1e-6)

        about_to_land = (
            (self.foot_heights < self.cfg.rewards.landing_height_threshold)
            & (time_to_contact < self.cfg.rewards.landing_time_threshold)
            & (~contacts)
        )
        excess_down_vel = torch.relu(
            down_vel - self.cfg.rewards.safe_landing_vel
        )
        penalty = torch.square(excess_down_vel)
        ans = torch.sum(penalty * about_to_land.float(), dim=1)
        # print("ans:", ans[0])
        return ans
    
    def _reward_blocked_wheel_vel(self):
        force_xy = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
        blocked = force_xy > 20.0
        wheel_vel = torch.abs(self.dof_vel[:, self.wheel_indices])
        vel_scale = 1.0
        penalty = 1.0 - torch.exp(-torch.square(wheel_vel / vel_scale))
        return torch.sum(penalty * blocked.float(), dim=1)
    
    def _reward_triggered_leg_up_vel(self):
        foot_z_vel = self.foot_velocities[:, :, 2]
        up_vel = torch.relu(foot_z_vel)
        # 只奖励被触发的摆动腿向上运动
        reward = self.swing_mask * (1.0 - torch.exp(-up_vel / 0.15))
        return torch.sum(reward, dim=1) * self.has_swing.float()
    
    def _reward_wrong_leg_lift(self):
        clearance = self.foot_heights
        # 有摆动腿时，非摆动腿抬高就罚
        wrong_lift = self.stance_mask * torch.relu(clearance - 0.02)
        return torch.sum(wrong_lift, dim=1) * self.has_swing.float()
    
    def _reward_triggered_leg_action_dir(self):
        left_swing = self.swing_mask[:, 0]
        right_swing = self.swing_mask[:, 1]
        # 这里的动作索引按你的 joint 顺序：
        # left: hip_roll 0, hip_pitch 1, knee 2
        # right: hip_roll 4, hip_pitch 5, knee 6
        left_lift_action = torch.relu(self.actions[:, 1]) + torch.relu(-self.actions[:, 2])
        right_lift_action = torch.relu(self.actions[:, 5]) + torch.relu(-self.actions[:, 6])
        reward = left_swing * left_lift_action + right_swing * right_lift_action
        return reward * self.has_swing.float()