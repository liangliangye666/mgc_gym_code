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

        # self.rwd_linVelTrackEnhancedPrev[env_ids] = 0 # 误差积分重置
        # self.rwd_angVelTrackEnhancedPrev[env_ids] = 0
        # #每次reset环境时重置last_contact_forces
        # self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.last_contact_forces = self.contact_forces
        # # 清空所有 stage
        self.phase[env_ids] = 0.0
        self.stage_buf[env_ids, :] = 0.0
        self.stage_buf[env_ids, 0] = 1.0      # # 强制设为 stand
        self.stage_time_buf[env_ids] = 0.0
        self.last_stage[env_ids] = 1.0      # 默认为站立状态
        self.leg_swing_first[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device)

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
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
    
    def post_physics_step(self):
        super().post_physics_step()
        # # 更新控制状态机所在阶段
        self.update_stage()
        self.update_gait_phase()

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
        gait_flag = self.gait_enable.unsqueeze(1)
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1) * gait_flag
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1) * gait_flag
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
                gait_flag,
                sin_phase,
                cos_phase,
                self.leg_swing_first.unsqueeze(1),
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
        privileged_obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.obs_buf,
                heights,
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
        self.commands[env_ids, 0] = (
            self.command_ranges["lin_vel_x"][env_ids, 1]
            - self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_x"
        ][env_ids, 0]

        self.commands[env_ids, 1] = (
            self.command_ranges["lin_vel_y"][env_ids, 1]
            - self.command_ranges["lin_vel_y"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_y"
        ][env_ids, 0]

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

        self.commands[env_ids, 5] = (
            self.command_ranges["mode_normalization"][env_ids, 1]
            - self.command_ranges["mode_normalization"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "mode_normalization"
        ][env_ids, 0]    

        # need_gait = torch.abs(self.commands[:, 5]) < self.cfg.gait.gait_train_proportion
        # prev_need_gait = torch.abs(old_cmd[:, 5]) < self.cfg.gait.gait_train_proportion
        # enter_gait = (~prev_need_gait) & need_gait
        # reset_ids = torch.where(enter_gait)[0]
        # if len(reset_ids) > 0:
        #     self.stage_buf[reset_ids, :] = 0.0
        #     self.stage_buf[reset_ids, 0] = 1.0
        #     self.stage_time_buf[reset_ids] = 0.0
        #     self.phase[reset_ids] = 0.0
        #     self.last_stage[reset_ids] = 0

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
        noise_vec[6:10] = 0.0  # commands # 命令无噪声
        noise_vec[10:16] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos # 各自由度位置噪声系数
        noise_vec[16:24] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel # 各自由度速度噪声系数
        noise_vec[24:] = 0.0  # previous actions,无历史动作噪声
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
        self.rwd_linVelTrackPrev = torch.zeros(self.num_envs, device=self.device)
        self.rwd_angVelTrackPrev = torch.zeros(self.num_envs, device=self.device)
        self.rwd_linVelTrackEnhancedPrev = torch.zeros(self.num_envs, device=self.device)
        self.rwd_angVelTrackEnhancedPrev = torch.zeros(self.num_envs, device=self.device)
        # # 步态
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.last_stage = torch.zeros(self.num_envs, device=self.device)
        self.gait_enable = torch.zeros(self.num_envs, device=self.device)
        self.leg_phase = torch.zeros((self.num_envs, 2),device=self.device,dtype=torch.float32)
        self.num_stages = len(self.cfg.gait.stage_names)                                                       # 状态数量
        self.stage_buf = torch.zeros((self.num_envs, self.num_stages),dtype=torch.float32, device=self.device)  # 每个环境处于什么状态
        self.leg_swing_first = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stage_time_buf = torch.zeros(self.num_envs, device=self.device)                                    # 处于当前状态的时间
        # self.commands_stages = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)         # 状态命令
        # self.gait = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)       # 步态命令
        # self.lase_gait = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)  # 上次步态命令
        # self.prev_foot_contact = torch.ones((self.num_envs, len(self.feet_indices)), device=self.device)

    def update_stage(self):
        ##stage_buf: ["stand", "gait", "recover"] -> 两轮站立,步态,恢复两轮
        #当前为gait且步态命令false：进入stand
        need_gait = (torch.abs(self.commands[:, 5]) < self.cfg.gait.gait_train_proportion)  # 需要步态
        fz = self.contact_forces[:, self.feet_indices, 2]                           # 获取轮子z向力
        foot_contact = (fz > 5.0).float()                                           # 轮子接触情况
        both_on_ground = torch.prod(foot_contact, dim=1) > 0                        # 两个轮子都在地上
        ####################第一阶段:stand->gait######################
        from0_to1 = torch.logical_and(self.stage_buf[:, 0] == 1.0, torch.logical_and(need_gait, self.stage_time_buf > 0.3)).float()
        # print("gait:",(self.gait == True).sum().item)
        ####################第二阶段:gait->recover####################
        from1_to2 = ((self.stage_buf[:, 1] == 1.0) & (~need_gait) & (self.stage_time_buf > 0.6)).float()
        ####################第三阶段:recover->stand###################
        from2_to0 = torch.logical_and(self.stage_buf[:, 2] == 1.0, torch.logical_and(
            both_on_ground,
            self.stage_time_buf > 0.15                  # 收脚完成
        )).float()
        # 更新stage_buf
        self.stage_buf[:, 0] = (self.stage_buf[:, 0] * (1.0 - from0_to1) + from2_to0)
        self.stage_buf[:, 1] = (self.stage_buf[:, 1] * (1.0 - from1_to2) + from0_to1)
        self.stage_buf[:, 2] = (self.stage_buf[:, 2] * (1.0 - from2_to0) + from1_to2)
        # print("stage_buf:", self.stage_buf)

        current_stage = torch.argmax(self.stage_buf, dim=1)
        # print(current_stage)
        same_stage = current_stage == self.last_stage
        self.stage_time_buf = torch.where(
            same_stage,
            self.stage_time_buf + self.dt,
            torch.zeros_like(self.stage_time_buf))
        self.last_stage = current_stage.clone()

    def update_gait_phase(self):
        period = self.cfg.gait.gait_period
        offset = 0.5
        self.gait_enable = self.stage_buf[:, 1] == 1.0  # gait
        phase = torch.zeros_like(self.stage_time_buf)
        # 只在 gait 阶段推进 phase
        phase[self.gait_enable] = (self.stage_time_buf[self.gait_enable] % period) / period
        self.phase = phase
        self.phase_left = phase.clone()
        self.phase_right = torch.where(self.gait_enable, (phase + offset) % 1.0, torch.zeros_like(phase))
        left_first_mask = (self.leg_swing_first==1) & self.gait_enable
        # 交换左右
        tmp = self.phase_left[left_first_mask].clone()
        self.phase_left[left_first_mask] = self.phase_right[left_first_mask]
        self.phase_right[left_first_mask] = tmp
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

    # ------------ reward functions----------------
    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.norm(
            self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1
        )
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1) + \
                 torch.clip(feet_distance - self.cfg.rewards.max_feet_distance, 0, 1)
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

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_pb(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_lin_vel() - self.rwd_linVelTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)

    def _reward_tracking_ang_vel_pb(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_ang_vel() - self.rwd_angVelTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.abs(base_height - self.cfg.rewards.base_height_target)

# ################## 步态控制 ##################
    def _reward_contact_number(self): 
        gait_enable  = self.stage_buf[:, 1]                                 # 只在 gait
        fz = self.contact_forces[:, self.feet_indices, 2]                   # 接触力z,维度[N, 2]
        contact = (fz > 1.0).float()                                        # 是否接触
        # print("phase:", self.leg_phase[0])
        is_stance = (self.leg_phase < 0.7).float()                         # 支撑脚和摆动脚,维度[N, 2]
        rew_contact = is_stance*contact - 1.3*(1-is_stance) * contact # 摆动腿不接触,摆动腿不接触
        rew_gait = torch.mean(rew_contact, dim=1)                            # 步态奖励
        return rew_gait * gait_enable
    
    def _reward_feet_clearance(self):
        """
        改进版：使用连续函数替代二值判断，提供更平滑的梯度。
        """
        clearance = self.foot_heights
        # print("clearance:", clearance[0])
        phase = self.leg_phase
        d = 0.7
        swing_mask = (phase >= d).float()
        # clearance = foot_height - self.cfg.asset.wheel_radius
        # print("clearance:", clearance[0])
        foot_height_ref = self.cfg.rewards.feet_height_target
        swing_phase = torch.relu((phase - d) / (1 - d))
        swing_phase = torch.clamp(swing_phase, 0.0, 1.0)
        swing_height_des = foot_height_ref * torch.sin(np.pi * swing_phase) ** 2

        # reward = 1 - torch.exp( -torch.square(foot_height - foot_height_ref) / 0.005)
        reward = torch.square(clearance - swing_height_des) / 0.001
        # 3. 对所有腿的得分求和（鼓励所有摆动腿都达标）
        total_reward = torch.sum(reward * swing_mask, dim=1)
        # print("total_reward:", total_reward[0])
        return total_reward
    
   # def _reward_feet_swing_height(self):
    #     gait_enable  = self.stage_buf[:, 1]                             # 只在 gait
    #     z = self.wheel_pos[:, :, 2]                                     # 抬脚高度
    #     phase = self.leg_phase
    #     d = 0.55
    #     swing = (phase >= d).float()                                    # 摆动脚
    #     z_min = self.cfg.asset.wheel_radius                             # 轮子半径
    #     z_target = self.cfg.commands.gait_foot_height                   # 步态抬脚高度
    #     delta_z = z_target - z_min
    #     # 构造归一化摆动腿swing_phase变量
    #     swing_phase = torch.relu((phase - d) / (1 - d))
    #     swing_phase = torch.clamp(swing_phase, 0.0, 1.0)
    #     # 光滑正弦轨迹
    #     z_ref = z_min + delta_z * torch.sin(np.pi * swing_phase) ** 2
    #     rew_foot_up = torch.clamp(z - z_min, min=0.0, max=0.1)          # 抬脚先给奖励
    #     height_err = torch.abs(z - z_ref)                               # 高度偏差惩罚（平滑）
    #     height_penalty = torch.clamp(height_err, max=delta_z)
    #     rew = torch.sum((10 * rew_foot_up - 2 * height_penalty) * swing, dim=1)   # 只对摆动脚起作用
    #     return rew * gait_enable

    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.foot_body_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, 1:3])
        return torch.sum(penalize, dim=(1,2))

    def _reward_wheel_zero_vel(self):
        gait_enable  = self.stage_buf[:, 1]                                             # 只在 gait
        wheel_vel = torch.sum(torch.square(self.dof_vel[:, self.wheel_indices]), dim=1)  # 轮子速度
        rew_wheel_vel = torch.exp(-0.01 * wheel_vel)
        return rew_wheel_vel * gait_enable                                              # 惩罚步态时轮子速度
