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
from .l5b_2wheel_config import L5B_2WHEEL_Cfg


class L5B_2WHEEL(LeggedRobot):
    def __init__(self, cfg: L5B_2WHEEL_Cfg, sim_params, physics_engine, sim_device, headless):
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

    def step(self, actions):
        """
        应用动作，进行模拟，并调用 self.post_physics_step()。

        该方法首先对输入的动作进行裁剪，以确保其在允许的范围内。然后，它渲染当前帧，调用 pre_physics_step() 方法进行模拟前的准备工作，
        并在模拟过程中多次调用 leg_post_physics_step() 方法。在模拟结束后，它调用 post_physics_step() 方法进行模拟后的处理工作，
        并返回裁剪后的观测、奖励、完成标志和其他信息。

        Args:
            actions (torch.Tensor): 形状为 (num_envs, num_actions_per_env) 的动作张量
        """
        # 裁剪动作以确保其在允许的范围内
        actions = self._apply_swing_feedforward(actions)
        # if self.has_swing[0]:
        #     print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # print("actions:", actions[0])
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # 渲染当前帧
        self.render()

        # 在模拟前进行准备工作,实际准备的是奖励函数,包含了即时奖励和奖励积分
        self.pre_physics_step()

        # 进行多次模拟步骤
        for _ in range(self.cfg.control.decimation):
            # 在模拟后进行处理工作
            # self.leg_post_physics_step()
            # 更新环境步数缓冲区
            self.envs_steps_buf += 1
            # 更新动作FIFO缓冲区,fifo维度(num_envs,max_delays,num_actions),unsqueeze(1)->在维度1增加一个大小为1的维度,也就是(num_envs,1,num_actions),注意这里是把新动作插到了前面
            self.action_fifo = torch.cat((self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1) # action_fifo[:,:-1,:]->现有动作队列,[所有环境,从开始到倒数第二个元素,所有动作维度]
            # 计算扭矩                                                                                     torch.cat(tensors,dim=1)->沿指定维度连接张量序列,前提是其他维度要一致
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :] # torch.arange()-生成一维整数序列(0,1,2,3...,num_envs-1)
            ).view(self.torques.shape)
            # 设置执行器的扭矩
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques)) # 将力矩设置到gym中
            # 如果启用了机器人推动功能，则随机推动机器人
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            # 进行模拟
            self.gym.simulate(self.sim) # 仿真一步
            # 如果设备是CPU，则获取模拟结果
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            # 更新自由度状态张量
            self.gym.refresh_dof_state_tensor(self.sim) # 更新自由度状态张量
            self.gym.refresh_dof_force_tensor(self.sim) # 刷新关节力矩张量
            # 计算自由度速度
            self.compute_dof_vel()

        # 在模拟后进行处理工作
        self.post_physics_step()

        # 返回裁剪后的观测、裁剪后的状态（如果有）、奖励、完成标志和其他信息
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

    def compute_dof_vel(self):
        # 计算关节位置差分，并处理角度边界（例如从359度到1度）
        # 使用remainder（取余）来确保角度差分在[-π, π]范围内
        diff = torch.remainder(self.dof_pos - self.last_dof_pos + self.pi, 2 * self.pi) - self.pi

        # 计算关节角速度（位置差分除以时间步长）
        self.dof_pos_dot = diff / self.sim_params.dt

        # 如果配置文件指定使用位置差分作为速度
        # 这通常用于处理速度测量不准确的情况
        if self.cfg.env.dof_vel_use_pos_diff:
            self.dof_vel = self.dof_pos_dot

        # 更新上一时刻的关节位置
        self.last_dof_pos[:] = self.dof_pos[:]

    def _update_goals(self):
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt                                                     # 达到目标时长判断
        self.cur_goal_idx[next_flag] += 1                                                                                               # 移动到下一个目标点
        # 限制在真实goal范围
        max_goal = self.cfg.terrain.num_goals - 1
        self.cur_goal_idx = torch.clamp(self.cur_goal_idx, max=max_goal)
        self.reach_goal_timer[next_flag] = 0                                                                                            # 重置到达目标计时器
        self.reached_goal_ids = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1) < self.cfg.env.next_goal_threshold   # 目标到达判断
        self.reach_goal_timer[self.reached_goal_ids] += 1
        self._compute_current_goal()
        # self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]                                                           # 目标位置
        # self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]                                                     # 下一目标位置
        # norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)                                                                    # 当前目标向量
        # target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        # self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])                                                     # 当前目标方向
        # norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)                                                               # 下一目标向量
        # target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        # self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])                                                # 下一目标方向

    def _compute_current_goal(self):
        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]                                                     # 下一目标位置
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)                                                               # 下一目标向量
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])                                                # 下一目标方向

    def post_physics_step(self):
        """
        检查终止条件，计算观测值和奖励。
        调用 self._post_physics_step_callback() 进行通用计算。
        如果需要，调用 self._draw_debug_vis()。

        该方法在物理模拟步骤之后执行以下操作：
        1. 刷新 actor 根状态张量、网络接触力张量和刚体状态张量。
        2. 更新 episode 长度缓冲区和通用步骤计数器。
        3. 准备各种物理量，如基座的四元数、线速度、角速度、加速度等。
        4. 调用 self._post_physics_step_callback() 进行通用计算。
        5. 检查终止条件，计算奖励，重置环境，计算观测值。
        6. 更新上一次的动作、基座位置、自由度速度和根速度。
        7. 如果启用了调试可视化，则绘制调试可视化。
        """
        # 刷新 actor 根状态张量
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # 刷新网络接触力张量
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # 刷新刚体状态张量
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 更新 episode 长度缓冲区
        self.episode_length_buf += 1
        # 更新通用步骤计数器,该变量从不重置,持续累加直到训练结束
        self.common_step_counter += 1

        # 准备基座的四元数
        self.base_quat[:] = self.root_states[:, 3:7]

        # 计算基座的线速度
        # self.base_lin_vel = (self.base_position - self.last_base_position) / self.dt
        # # 将基座线速度从世界坐标系转换到基座坐标系
        # self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.base_lin_vel)

        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        # 将基座角速度从世界坐标系转换到基座坐标系
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # 计算加速度
        self.dof_acc = (self.dof_vel - self.last_dof_vel) / self.dt

        # 将重力向量从世界坐标系转换到基座坐标系
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 获取基座的欧拉角
        self.base_euler_zyx = get_euler_zyx_tensor(self.base_quat)

        # without yaw
        base_euler_zyx_local = self.base_euler_zyx.clone()
        base_euler_zyx_local[:, 2] = 0

        self.base_quat_local = quat_from_euler_zyx(base_euler_zyx_local)
        self.base_lin_vel_head = quat_rotate_inverse(self.base_quat_local, self.root_states[:, 7:10]) # head坐标系中的线速度
        self.base_ang_vel_head = quat_rotate_inverse(self.base_quat_local, self.root_states[:, 10:13]) # head坐标系中的角速度
        # 使用轮子速度作为机器人线速度,使用轮子差速作为机器人角速度
        self.wheel_lin_vel = torch.sum(self.dof_vel[:, self.wheel_indices[[0, 1]]], dim=1) * self.cfg.asset.wheel_radius * 0.5
        self.wheel_ang_vel = (
            (self.dof_vel[:, self.wheel_indices[1]] - self.dof_vel[:, self.wheel_indices[0]]) * self.cfg.asset.wheel_radius / self.cfg.asset.track_width
        )

        # 计算基座坐标系下两个轮的位置和姿态
        self.wheel_pos = self.rigid_body_pos[:, self.wheel_link_indices, :]
        self.wheel_vel_left = quat_rotate_inverse(self.base_quat, self.rigid_body_vel[:, self.wheel_link_indices[0], :]).unsqueeze(1)
        self.wheel_vel_right = quat_rotate_inverse(self.base_quat, self.rigid_body_vel[:, self.wheel_link_indices[1], :]).unsqueeze(1)
        self.wheel_body_vel = torch.cat((self.wheel_vel_left, self.wheel_vel_right), dim=1)
        self.wheel_pos_left_local = quat_rotate_inverse(self.base_quat, (self.rigid_body_pos[:, self.wheel_link_indices[0], :]-self.rigid_body_pos[:, self.base_link_indices[0], :]))
        self.wheel_pos_right_local = quat_rotate_inverse(self.base_quat, (self.rigid_body_pos[:, self.wheel_link_indices[1], :]-self.rigid_body_pos[:, self.base_link_indices[0], :]))
        self.wheel_euler_left = get_euler_zyx_tensor(self.rigid_body_quat[:, self.wheel_indices[0], :])
        self.wheel_euler_left_local = quat_rotate_inverse(self.base_quat, self.wheel_euler_left)
        self.wheel_euler_right = get_euler_zyx_tensor(self.rigid_body_quat[:, self.wheel_indices[1], :])
        self.wheel_euler_right_local = quat_rotate_inverse(self.base_quat, self.wheel_euler_right)
        self.forward_kinematics() # 计算运动学信息

        # 调用回调函数进行通用计算,重新采样命令和地形高度信息
        self._update_goals()
        self._update_contact_history()
        self._get_swing_stance_mask()
        self._post_physics_step_callback()

        # 检查终止条件
        self.check_termination()
        # 计算奖励
        self.compute_reward()
        # 更新控制状态机所在阶段
        # self.update_stage()
        # self.update_gait_phase()
        # 获取需要重置的环境 ID
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # 重置指定的环境
        self.reset_idx(env_ids)

        self.cur_goals = self._gather_cur_goals()                                                   # 当前目标
        self.next_goals = self._gather_cur_goals(future=1)                                          # 下一目标
        
        # 计算观测值
        self.compute_observations()  # 在某些情况下，可能需要进行模拟步骤以刷新某些观测值（例如身体位置）

        # 更新上一次的动作
        self.last_actions[:, :, 1] = self.last_actions[:, :, 0]
        self.last_actions[:, :, 0] = self.actions[:]
        # 更新上一次的基座位置
        self.last_base_position[:] = self.base_position[:]
        # 更新上一次的joint速度
        self.last_dof_vel[:] = self.dof_vel[:]
        # 更新上一次的base速度_inWorld
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # 如果启用了调试可视化，则绘制调试可视化
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
            self._draw_goals()

    def check_termination(self):
        # 检查环境是否需要重置
        reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 10.0,
            dim=1,
        )
        # 如果接触力超过10N，则认为是失败

        fail_buf = torch.logical_or(
            torch.abs(self.base_euler_zyx[:, 1]) > 0.2, torch.abs(self.base_euler_zyx[:, 0]) > 0.2
        )  # base_euler_zyx (roll pitch yaw)
        # 检查横滚或俯仰角是否超过阈值

        self.fail_buf *= fail_buf  # 如果已失败，则保持失败状态
        self.fail_buf += fail_buf  # 如果是第一次失败，则标记为失败
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # 检查是否达到最大步数
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.edge_reset_buf = self.base_position[:, 0] > self.terrain_x_max - 1
            # 如果机器人在x轴方向超出最大x坐标，则重置
            self.edge_reset_buf |= self.base_position[:, 0] < self.terrain_x_min + 1
            # 如果机器人在x轴方向超出最小x坐标，则重置
            self.edge_reset_buf |= self.base_position[:, 1] > self.terrain_y_max - 1
            # 如果机器人在y轴方向超出最大y坐标，则重置
            self.edge_reset_buf |= self.base_position[:, 1] < self.terrain_y_min + 1
            # 如果机器人在y轴方向超出最小y坐标，则重置
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt) | self.time_out_buf | self.edge_reset_buf | reset_buf
        )
        # 将达到失败次数、超时或超出边界的情况都标记为重置
        # fail_buf：失败标记
        # time_out_buf：超时标记
        # edge_reset_buf：超出边界标记

    def forward_kinematics(self):
        self.wheel_pos_left_local_x = self.wheel_pos_left_local[:, 0]
        self.wheel_pos_left_local_y = self.wheel_pos_left_local[:, 1]
        self.wheel_pos_left_local_z = self.wheel_pos_left_local[:, 2]
        self.wheel_pos_right_local_x = self.wheel_pos_right_local[:, 0]
        self.wheel_pos_right_local_y = self.wheel_pos_right_local[:, 1]
        self.wheel_pos_right_local_z = self.wheel_pos_right_local[:, 2]
        self.wheel_euler_left_local_x = self.wheel_euler_left_local[:, 0]
        self.wheel_euler_left_local_z = self.wheel_euler_left_local[:, 2]
        self.wheel_euler_right_local_x = self.wheel_euler_right_local[:, 0]
        self.wheel_euler_right_local_z = self.wheel_euler_right_local[:, 2]
        # print("wheel_x: ", self.wheel_pos_left_local_x, self.wheel_pos_right_local_x)
        # print("wheel_y: ", self.wheel_pos_left_local_y, self.wheel_pos_right_local_y)
        # print("wheel_ind", self.wheel_indices)
        # print("end_x", self.end_x)

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
        self._compute_current_goal()    # 计算当前目标方向等
        # reset buffers
        self.last_actions[env_ids] = 0.0 # 运动历史清零
        self.last_dof_vel[env_ids] = 0.0
        self.last_contacts[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.cur_goal_idx[env_ids] = 0
        self.trigger_cooldown[env_ids] = self.cfg.rewards.cooldown_time
        self.stuck_counter[env_ids] = 0.0
        self.was_hitting[env_ids] = False
        self.contact_history[env_ids] = 0.0
        self.current_swing[env_ids] = 0.0
        self.swing_time[env_ids] = 0.0
        self.has_swing[env_ids] = False
        self.swing_mask[env_ids] = 0                            # 摆动腿
        self.stance_mask[env_ids] = 1.0                         # 两条腿都重置为支撑腿
        self.in_double_support[env_ids] = True                        # 双支撑
        self.ds_time[env_ids] = 0.0                                # 双支撑时间
        self.episode_length_buf[env_ids] = 0 # 回合计数器重置,统计当前回合已进行的步数,回合有终止或失败条件,然后重置时该变量刷新重置
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0 # 环境从创建以来累计的总步数,用于课程学习,固定步数周期做一次调整等地方
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.obs_history[env_ids] = 0 
        obs_buf = self.compute_proprioception_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length) # 用当前观测值重置历史观测
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
            self.extras["time_outs"] = self.time_out_buf

        self.base_quat[env_ids] = self.root_states[env_ids, 3:7] # 本体姿态四元数更新
        self.base_euler_zyx = get_euler_zyx_tensor(self.base_quat) # 欧拉角姿态计算
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids]) # 重力在基座坐标系的投影

        # without yaw
        base_euler_zyx_local = self.base_euler_zyx.clone() # 局部姿态处理,忽略yaw角度
        base_euler_zyx_local[:, 2] = 0
        self.base_quat_local = quat_from_euler_zyx(base_euler_zyx_local)

        self.rwd_linVelTrackEnhancedPrev[env_ids] = 0 # 误差积分重置
        self.rwd_angVelTrackEnhancedPrev[env_ids] = 0
        #每次reset环境时重置last_contact_forces
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.last_contact_forces = self.contact_forces
        if self.cfg.commands.gait_command:
            # 清空所有 stage
            self.phase[env_ids] = 0.0
            self.stage_buf[env_ids, :] = 0.0
            # 强制设为 stand
            self.stage_buf[env_ids, 0] = 1.0
            # 可选：清空阶段时间，防止残留
            self.stage_time_buf[env_ids] = 0.0
            self.last_stage[env_ids] = 0.0

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        # 初始化奖励缓冲区为零
        self.rew_buf[:] = 0.0
        # 遍历所有奖励函数
        for i in range(len(self.reward_functions)):
            # 获取奖励函数的名称
            name = self.reward_names[i]
            # 调用奖励函数并乘以相应的权重,reward_functions返回的是(num_envs,)维度
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # print("rew_be_name", name, rew)
            # 裁剪奖励值，防止奖励值过大或过小, 单个奖励的最大值为1
            rew = torch.clip( # 剪裁到正负0.01
                rew,
                -self.cfg.rewards.clip_single_reward * self.dt,
                self.cfg.rewards.clip_single_reward * self.dt,
            )
            # print("rew_af_name", name, rew)
            # 将裁剪后的奖励值加到奖励缓冲区中,rew_buf是每次迭代的总奖励
            self.rew_buf += rew
            # 将裁剪后的奖励值加到该奖励项的总累计奖励中
            self.episode_sums[name] += rew
        # 如果配置中设置了只使用正奖励，则将奖励缓冲区中的值裁剪为非负数
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        # 如果配置中设置了终止奖励，则将终止奖励加到奖励缓冲区中
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins: # 如果是复杂地形,那么使用自定义原点
            self.root_states[env_ids] = self.base_init_state # 位置/姿态/角速度/线速度
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -0.1, 0.1, (len(env_ids), 2), device=self.device
            )  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _push_robots(self):
        """随机推动机器人"""
        # 获取需要被推动的环境ID
        # 根据推动间隔时间和仿真时间步长，确定哪些环境需要被推动
        env_ids = (
            (self.envs_steps_buf % int(self.cfg.domain_rand.push_interval_s / self.sim_params.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(env_ids) == 0:
            return

        # 计算最大推力
        # 根据机器人质量和最大速度变化计算最大推力
        # F = ma = m * (Δv/Δt)
        max_push_force = self.base_mass.mean().item() * self.cfg.domain_rand.max_push_vel_xy / self.sim_params.dt
        max_push_torque = self.base_mass.mean().item() * self.cfg.domain_rand.max_push_ang_vel / self.sim_params.dt

        # 重置外力张量
        self.push_forces[:] = 0
        self.push_torques[:] = 0

        # 生成随机推力
        # 在[-max_push_force, max_push_force]范围内随机生成三维推力
        push_forces = torch_rand_float(-max_push_force, max_push_force, (self.num_envs, 3), device=self.device)
        push_torques = torch_rand_float(-max_push_torque, max_push_torque, (self.num_envs, 3), device=self.device)

        # 将推力从世界坐标系转换到机器人局部坐标系
        self.push_forces[env_ids, 0, :] = quat_rotate(self.base_quat[env_ids], push_forces[env_ids]) # 推力作用在env_ids环境的0号刚体(base)上
        self.push_torques[env_ids, 0, :] = quat_rotate(self.base_quat[env_ids], push_torques[env_ids])

        # 减小竖直方向(z轴)的推力
        self.push_forces[env_ids, 0, 2] *= 0.5

        # 将外力施加到仿真环境中
        # 使用Isaac Gym的API应用外力和扭矩
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.push_forces),
            gymtorch.unwrap_tensor(self.push_torques),
            gymapi.ENV_SPACE,
        )

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof,
                2,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()

                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # # 打印初始质量信息（调试用）
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")

        # 随机化基座质量
        if self.cfg.domain_rand.randomize_base_mass:
            if env_id == 0:
                # 获取配置中设定的额外质量范围
                min_add_mass, max_add_mass = self.cfg.domain_rand.added_mass_range
                # 为每个环境生成随机的额外质量
                self.base_add_mass = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (max_add_mass - min_add_mass)
                    + min_add_mass
                )
                self.raw_base_mass = props[0].mass
                # 计算基座的总质量（原始质量+额外质量）
                self.base_mass = props[0].mass + self.base_add_mass
            # 为当前环境添加额外质量
            props[0].mass += self.base_add_mass[env_id]
        else:
            # 如果不进行随机化，则使用原始质量
            self.base_mass[:] = props[0].mass

        # 随机化基座重心位置
        if self.cfg.domain_rand.randomize_base_com:
            if env_id == 0:
                # 获取配置中设定的重心偏移范围
                com_x, com_y, com_z = self.cfg.domain_rand.rand_com_vec
                # 在x方向随机生成重心偏移
                self.base_com[:, 0] = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (com_x * 2)
                    - com_x
                )
                # 在y方向随机生成重心偏移
                self.base_com[:, 1] = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (com_y * 2)
                    - com_y
                )
                # 在z方向随机生成重心偏移
                self.base_com[:, 2] = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (com_z * 2)
                    - com_z
                )
            # 为当前环境设置重心偏移
            props[0].com.x += self.base_com[env_id, 0]
            props[0].com.y += self.base_com[env_id, 1]
            props[0].com.z += self.base_com[env_id, 2]

        # 随机化转动惯量
        if self.cfg.domain_rand.randomize_inertia:
            for i in range(len(props)):
                # 获取配置中设定的转动惯量缩放范围
                low_bound, high_bound = self.cfg.domain_rand.randomize_inertia_range
                # 随机生成转动惯量缩放系数
                inertia_scale = np.random.uniform(low_bound, high_bound)
                # 应用缩放系数到质量和转动惯量
                props[i].mass *= inertia_scale
                props[i].inertia.x.x *= inertia_scale
                props[i].inertia.y.y *= inertia_scale
                props[i].inertia.z.z *= inertia_scale

        return props

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 检查是否需要重新采样命令
        # 如果当前环境的episode长度是重新采样时间的整数倍，则需要重新采样命令
        env_ids = (
            (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        # 对需要重新采样命令的环境重新采样命令
        self._resample_commands(env_ids)
        # 如果启用了heading命令，则根据目标和当前方向计算角速度命令
        if self.cfg.commands.heading_command:
            # 计算当前方向的前向向量
            forward = quat_apply(self.base_quat, self.forward_vec)
            # 计算当前方向的偏航角
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            # 根据目标偏航角和当前偏航角计算角速度命令
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 4] - heading), -1.0, 1.0)

        # 如果启用了测量地形高度，则计算并存储测量的地形高度
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        # 计算并存储机器人的平均高度,去掉地形高度的影响,squeeze删除维度,unsqueeze增加维度
        self.base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        old_cmd = self.commands.clone()               # 拷贝旧命令
        self.commands[env_ids, 0] = torch_rand_float( # 线速度重采样
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.commands[env_ids, 1] = torch_rand_float( # 线速度重采样
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.commands[env_ids, 3] = torch_rand_float( # 线速度重采样
            self.command_ranges["height"][0],
            self.command_ranges["height"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)       

        if self.cfg.commands.heading_command: # 如果有朝向命令,那么在第五个命令处随机化一个朝向值
            self.commands[env_ids, 4] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
            # self.commands[env_ids, 4] = self.target_yaw[env_ids]
        else:
            self.commands[env_ids, 2] = torch_rand_float( # 否则直接在第二个命令处随机化一个角速度值
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        self.commands[env_ids,5] = torch_rand_float(        # 训练模式采样
            self.command_ranges["mode_normalization"][0],
            self.command_ranges["mode_normalization"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)   
        if self.cfg.commands.gait_command:
            need_gait = torch.abs(self.commands[:, 5]) < self.cfg.commands.gait_train_proportion
            prev_need_gait = torch.abs(old_cmd[:, 5]) < self.cfg.commands.gait_train_proportion
            enter_gait = (~prev_need_gait) & need_gait
            reset_ids = torch.where(enter_gait)[0]
            if len(reset_ids) > 0:
                self.stage_buf[reset_ids, :] = 0.0
                self.stage_buf[reset_ids, 0] = 1.0
                self.stage_time_buf[reset_ids] = 0.0
                self.phase[reset_ids] = 0.0
                self.last_stage[reset_ids] = 0

    def compute_proprioception_observations(self):
        # note that observation noise need to modified accordingly !!!
        gait_flag = self.gait_enable.unsqueeze(1)
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1) * gait_flag
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1) * gait_flag
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        obs_buf = torch.cat(
            (
                # self.base_lin_vel * self.obs_scales.lin_vel, # 3, 机器人base线速度
                self.base_ang_vel * self.obs_scales.ang_vel, # 3 ,机器人base角速度(在base坐标系)
                self.base_quat_local * self.obs_scales.quat, # 4 ,机器人姿态四元数
                self.commands[:, :4] * self.commands_scale,  # 4 , 外界命令
                self.dof_pos[:, :3] * self.obs_scales.dof_pos,  # 3 ,机器人关节位置,左边髋膝关节
                self.dof_pos[:, 4:7] * self.obs_scales.dof_pos,  # 3 ,机器人关节位置,右边髋膝关节
                self.dof_vel * self.obs_scales.dof_vel,  # 8 , 8个关节速度
                self.actions,  # 8 ,8个关节输出(上一时刻)
                # gait_flag,
                # sin_phase,
                # cos_phase,
                target_vec_norm,
            ),
            dim=-1,
        )
        return obs_buf

    def compute_observations(self):
        """Computes observations"""
        self.obs_buf = self.compute_proprioception_observations()

        if self.cfg.env.num_privileged_obs is not None: # 使用特权观测
            heights = ( # 机器人相对于地形的高度观测值,是机器人感知地形变化的关键信息
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            external_forces_and_torques = torch.cat((self.push_forces[:, 0, :], self.push_torques[:, 0, :]), dim=-1) # 观测0号刚体(base)受到的力和力矩

            self.privileged_obs_buf = torch.cat(
                (
                    self.base_lin_vel * self.obs_scales.lin_vel,  # 3,基座线速度
                    self.base_euler_zyx * self.obs_scales.quat,  # 3,基座欧拉角
                    self.obs_buf, # action网络观测值
                    self.projected_gravity,  # 3,重力投影
                    self.last_actions[:, :, 0],  # 8,上上动作
                    self.last_actions[:, :, 1],  # 8,上一动作
                    self.dof_acc * self.obs_scales.dof_acc,  # 8,关节加速度,速度差分来的
                    # (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 8
                    heights,  # 7*11,地形信息
                    self.dof_torques * self.obs_scales.torque,  # 8,力矩信息
                    (self.base_mass - self.raw_base_mass).view(self.num_envs, 1),  # 1,base质量
                    self.base_com,  # 3,base质心
                    # self.default_dof_pos - self.raw_default_dof_pos,  # 8
                    self.friction_coef.view(self.num_envs, 1),  # 1,摩擦系数
                    self.restitution_coef.view(self.num_envs, 1),  # 1,弹性系数
                    external_forces_and_torques * self.priv_obs_scales.external_wrench,  # 6,基座外力(矩)
                    # self.stage_buf.float(),
                    self.cur_goals[:, :2],                          #   2
                    self.contact_forces[:, self.feet_indices, :].flatten(1),   #   6
                    self.current_swing.unsqueeze(1),                             #   1
                    self.swing_time.unsqueeze(1),                                #   1,  摆动时间
                    self.swing_mask,                                #   2,  摆动腿
                    self.stance_mask,                               #   2,  支撑腿
                    self.has_swing.unsqueeze(1),                                 #   1
                ),
                dim=-1,
            )

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec # rand_link生成[0,1)之间的随机数

        self.obs_history = torch.cat((self.obs_history[:, self.num_obs :], self.obs_buf), dim=-1)

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

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        # noise_vec[0:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel  # lin_vel
        # noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel  # ang_vel
        # noise_vec[6:10] = noise_scales.quat * noise_level * self.obs_scales.quat  # quat
        # noise_vec[10:14] = 0.0  # commands
        # noise_vec[14:16] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos 
        # noise_vec[16:18] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos 
        # noise_vec[18:24] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # dof_vel_all
        # noise_vec[24:30] = 0.0  # previous actions
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel  # ang_vel
        noise_vec[3:7] = noise_scales.quat * noise_level * self.obs_scales.quat  # quat
        noise_vec[7:11] = 0.0  # commands
        noise_vec[11:14] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos 
        noise_vec[14:17] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos 
        noise_vec[17:25] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # dof_vel_all
        noise_vec[25:33] = 0.0  # previous actions
        # noise_vec[33:36] = 0.0
        # noise_vec[33:35] = 0.0
        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)       # 演员(机器人)的根状态(位置/旋转/速度),存储了13个值:[x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz]
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)              # 所有关节(自由度)的状态(位置和速度),对于每个关节,存储了2个值:[position,velocity]
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)              # 所有关节的力矩信息,[torque]
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)    # 所有刚体的净接触力,存储每个刚体的三维接触力:[Fx,Fy,Fz]
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)      # 所有刚体在世界坐标系中的位置,维度:[num_envs * num_bodies, 13]

        self.gym.refresh_dof_state_tensor(self.sim) # 刷新自由度状态张量,确保数据最新,以上四个是指向对应物理量的指针,刷新数据后保证指针指向的是最新的数据
        self.gym.refresh_dof_force_tensor(self.sim) # 刷新关节力矩张量
        self.gym.refresh_actor_root_state_tensor(self.sim) # 刷新演员根状态张量
        self.gym.refresh_net_contact_force_tensor(self.sim) # 刷新刚体净接触力张量
        self.gym.refresh_rigid_body_state_tensor(self.sim) # 刷新刚体状态张量

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)       # 机器人根状态张量转换为PyTorch张量,actor_root_state是指向GPU数据的原始指针,wrap_tensor是将指针指向数据包装成PyTorch张量
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)         # 机器人自由度状态张量转换为PyTorch张量
        self.dof_torques = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)     # 关节力矩反馈信息,维度[num_envs,num_dof]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0] # 关节位置,维度(num_envs,num_dof),view是引用,因此关节位置和速度会自动刷新,将张量按照期望维度排列,[...,0]是提取该维度信息
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1] # 关节速度,维度(num_envs,num_dof),提取后维度减1
        self.dof_acc = torch.zeros_like(self.dof_vel) # 初始化自由度加速度为0
        self.base_quat = self.root_states[:, 3:7] # 获取base姿态
        self.base_euler_zyx = get_euler_zyx_tensor(self.base_quat)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor) # 获取刚体状态
        self.rigid_body_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[..., 0:3] # 获取刚体位置
        self.rigid_body_quat = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[..., 3:7] # 获取刚体姿态
        self.rigid_body_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[..., 7:10] # 获取刚体速度

        # without yaw
        base_euler_zyx_local = self.base_euler_zyx.clone()
        base_euler_zyx_local[:, 2] = 0
        self.base_quat_local = quat_from_euler_zyx(base_euler_zyx_local)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view( # 获取机器人各刚体接触力,维度(num_envs,num_bodies),每个index存放[Fx,Fy,Fz]信息
            self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0 # 初始化一个公共步数计数器
        self.extras = {} # 初始化一个空字典用于存放额外信息
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg) # 存储噪声缩放向量
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat( # 创建重力向量
            (self.num_envs, 1)
        )
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1)) # 创建前向向量
        self.torques = torch.zeros( # 初始化力矩张量
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.torques_scale = torch.ones( # 初始化力矩缩放因子
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros( # 初始化p增益系数
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.d_gains = torch.zeros( # 初始化d增益系数
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.actions = torch.zeros( # 初始化动作张量
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.actions_ff = torch.zeros( # 初始化动作张量
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.actions_ff = torch.zeros( # 初始化动作张量
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros( # 存放前两个周期的actions,用于平滑连续命令之间的差异
            self.num_envs,
            self.num_actions,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.base_position = self.root_states[:, :3] # 存储所有环境的位置
        # print(self.base_position)
        self.last_base_position = self.base_position.clone() # 上一个时刻所有环境的位置
        self.last_dof_pos = torch.zeros_like(self.dof_pos) # 上一个时刻所有关节的位置
        self.last_dof_vel = torch.zeros_like(self.dof_vel) # 上一个时刻所有关节的速度
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13]) # 上一个时刻所有环境的速度
        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)   # 到达目标点计时器
        self.target_yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)   # 目标点的yaw角度
        self.last_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)   # 上一时刻距离目标点距离
        self.feet_air_time = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device, dtype=torch.float)  # 脚腾空时间  
        self.trigger_cooldown = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device)                  # 卡住冷却时间 
        self.stuck_counter = torch.zeros(self.num_envs, device=self.device)   
        self.was_hitting = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device, dtype=torch.bool)
        self.contact_history = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device, dtype=torch.float)
        self.current_swing = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.swing_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)                             # 摆动时间
        self.swing_mask = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)                          # 摆动腿
        self.stance_mask = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)                         # 支撑腿
        self.has_swing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.in_double_support = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)                        # 双支撑
        self.ds_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)                                # 双支撑时间
        self.commands = torch.zeros( # 初始化命令张量,用于存储每个环境的控制指令,额外一列可能用于存储命令的持续时间或其他附加信息
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.commands_scale = torch.tensor( # 初始化命令缩放因子,用于标准化命令值
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel_y,
                self.obs_scales.ang_vel,
                self.obs_scales.height_measurements,
            ],
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros( # 初始化上一个接触状态信息
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.last_contact_forces = torch.zeros(
            self.num_envs,
            self.num_bodies,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10]) # 基座线速度
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13]) # 基座角速度
        self.base_lin_vel_head = quat_rotate_inverse(self.base_quat_local, self.root_states[:, 7:10]) # head坐标系中的线速度
        self.base_ang_vel_head = quat_rotate_inverse(self.base_quat_local, self.root_states[:, 10:13]) # head坐标系中的角速度
        self.push_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False) # 推动力
        self.push_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False) # 推动力矩
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec) # 重力投影
        self.action_delay_idx = torch.zeros( # 动作延时
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        delay_max = np.int64(np.round(self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt) + 1) # 最大延时
        self.action_fifo = torch.zeros( # 演员fifo,维度(num_envs,delay_max,num_actions)
            (self.num_envs, delay_max, self.cfg.env.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        if self.cfg.terrain.measure_heights: # 测量高度的位置
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1) # base高度
        self.base_height_local = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.joint_indices = torch.tensor(list(self.cfg.asset.joint_indices), device=self.device) # 关节索引
        self.wheel_indices = torch.tensor(list(self.cfg.asset.wheel_indices), device=self.device) # 轮子索引
        self.base_link_indices = torch.tensor(list(self.cfg.asset.base_link_indices), device=self.device) # base_link索引
        self.joint_link_indices = torch.tensor(list(self.cfg.asset.joint_link_indices), device=self.device)  # joint_link索引
        self.wheel_link_indices = torch.tensor(list(self.cfg.asset.wheel_link_indices), device=self.device)   # wheel_link索引

        self.wheel_lin_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.wheel_ang_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # joint positions offsets and PD gains
        self.raw_default_dof_pos = torch.zeros(  # 初始化默认关节位置,单个环境
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.default_dof_pos = torch.zeros( # 存储所有环境的默认关节位置
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_dofs):  # 遍历所有关节,设置默认位置和控制器增益
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.raw_default_dof_pos[i] = angle
            self.default_dof_pos[:, i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[:, i] = 0.0
                self.d_gains[:, i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")

        if self.cfg.domain_rand.randomize_Kp: # 随机化Kp
            (
                p_gains_scale_min,
                p_gains_scale_max,
            ) = self.cfg.domain_rand.randomize_Kp_range
            self.p_gains *= torch_rand_float(
                p_gains_scale_min,
                p_gains_scale_max,
                self.p_gains.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_Kd: # 随机化Kd
            (
                d_gains_scale_min,
                d_gains_scale_max,
            ) = self.cfg.domain_rand.randomize_Kd_range
            self.d_gains *= torch_rand_float(
                d_gains_scale_min,
                d_gains_scale_max,
                self.d_gains.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_motor_torque: # 随机化电机力矩
            (
                torque_scale_min,
                torque_scale_max,
            ) = self.cfg.domain_rand.randomize_motor_torque_range
            self.torques_scale *= torch_rand_float(
                torque_scale_min,
                torque_scale_max,
                self.torques_scale.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_default_dof_pos: # 随机化默认位置
            self.default_dof_pos += torch_rand_float(
                self.cfg.domain_rand.randomize_default_dof_pos_range[0],
                self.cfg.domain_rand.randomize_default_dof_pos_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_action_delay: # 随机化动作延时
            action_delay_idx = torch.round(
                torch_rand_float(
                    self.cfg.domain_rand.delay_ms_range[0] / 1000 / self.sim_params.dt,
                    self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt,
                    (self.num_envs, 1),
                    device=self.device,
                )
            ).squeeze(-1)
            self.action_delay_idx = action_delay_idx.long()

        self.rwd_linVelTrackPrev = torch.zeros(self.num_envs, device=self.device)
        self.rwd_angVelTrackPrev = torch.zeros(self.num_envs, device=self.device)
        self.rwd_linVelTrackEnhancedPrev = torch.zeros(self.num_envs, device=self.device)
        self.rwd_angVelTrackEnhancedPrev = torch.zeros(self.num_envs, device=self.device)
        # 步态
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.last_stage = torch.zeros(self.num_envs, device=self.device)
        self.gait_enable = torch.zeros(self.num_envs, device=self.device)
        self.leg_phase = torch.zeros((self.num_envs, 2),device=self.device,dtype=torch.float32)
        self.num_stages = len(self.cfg.asset.stage_names)                                                       # 状态数量
        self.stage_buf = torch.zeros((self.num_envs, self.num_stages),dtype=torch.float32, device=self.device)  # 每个环境处于什么状态
        self.stage_time_buf = torch.zeros(self.num_envs, device=self.device)                                    # 处于当前状态的时间
        self.commands_stages = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)         # 状态命令
        self.gait = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)       # 步态命令
        self.lase_gait = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)  # 上次步态命令
        self.prev_foot_contact = torch.ones((self.num_envs, len(self.feet_indices)), device=self.device)


    def _prepare_reward_function(self):
        """准备一个奖励函数列表，这些函数将被调用以计算总奖励。
        查找self._reward_<REWARD_NAME>，其中<REWARD_NAME>是cfg中所有非零奖励权重的名称。
        """
        # 移除零权重的奖励项，并将非零权重乘以时间步长dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)  # 移除权重为0的奖励项
            else:
                self.reward_scales[key] *= self.dt  # 将奖励权重乘以时间步长

        # 准备奖励函数列表和名称列表
        self.reward_functions = []  # 存储奖励函数
        self.reward_names = []  # 存储奖励名称
        for name, scale in self.reward_scales.items():
            if name == "termination":  # 跳过终止奖励
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))  # 将奖励函数添加到列表中

        # 初始化每个环境的奖励累计值
        self.episode_sums = {
            name: torch.zeros(
                self.num_envs,  # 环境数量
                dtype=torch.float,  # 数据类型为浮点数
                device=self.device,  # 使用指定设备
                requires_grad=False,  # 不需要梯度
            )
            for name in self.reward_scales.keys()  # 为每个环境下每个奖励项创建累计值
        }

    def _get_heights(self, env_ids=None):
        """采样每个机器人周围指定点的地形高度。
        这些点根据机器人基座的位置进行偏移，并根据基座的偏航角进行旋转。
            Args:
            env_ids (List[int], optional): 需要返回高度的环境子集。默认为None。
            Raises:
            NameError: 当地形网格类型为'none'时抛出异常。
            Returns:
            torch.Tensor: 包含采样高度的张量。
        """
        # 如果地形是平面，返回全零张量
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs,
                self.num_height_points, # 11*7
                device=self.device,
                requires_grad=False,
            )
        # 如果没有地形，抛出异常
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        # 根据是否指定环境ID来处理采样点
        if env_ids:
            # 对指定环境的采样点进行旋转和平移
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points), # 给每个环境复制77份四元数姿态
                self.height_points[env_ids], # 高度点,此时只有xy有值,z是0,维度(num_envs,num_height_points,3)
            ) + (self.root_states[env_ids, :3]).unsqueeze(1) # 维度变为(num_envs,1,3),维度相同即可相加扩展广播,会将root_states的对应维度扩展(复制)num_height_points份
        else:
            # 对所有环境的采样点进行旋转和平移
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_states[:, :3]
            ).unsqueeze(1)

        # 添加地形边界大小,假设世界坐标系是0,但是在地形坐标系中,边界起点才是0,原来世界坐标系的原点就是border_size了
        points += self.terrain.cfg.border_size
        # 将点坐标转换为地形网格的索引
        points = (points / self.terrain.cfg.horizontal_scale).long()
        # 提取x和y坐标
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        # 裁剪坐标以确保在有效范围内
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        # 获取相邻三个点的高度
        heights1 = self.height_samples[px, py]  # 当前点的高度
        heights2 = self.height_samples[px + 1, py]  # x方向下一个点的高度
        heights3 = self.height_samples[px, py + 1]  # y方向下一个点的高度
        # 取三个高度中的最小值
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        # 返回最终的高度值，并应用垂直缩放
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        # 获取机器人资产文件的路径
        asset_path = self.cfg.asset.file.format(WHEEL_LEGGED_GYM_ROOT_DIR=WHEEL_LEGGED_GYM_ROOT_DIR)
        # 获取资产文件所在的目录
        asset_root = os.path.dirname(asset_path)
        # 获取资产文件的文件名
        asset_file = os.path.basename(asset_path)

        # 创建资产选项对象
        asset_options = gymapi.AssetOptions()
        # 设置默认的自由度驱动模式
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        # 设置是否折叠固定关节
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        # 设置是否用胶囊体替换圆柱体
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        # 设置是否翻转视觉附件
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        # 设置是否固定基座链接
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        # 设置资产的密度
        asset_options.density = self.cfg.asset.density
        # 设置资产的角阻尼
        asset_options.angular_damping = self.cfg.asset.angular_damping
        # 设置资产的线阻尼
        asset_options.linear_damping = self.cfg.asset.linear_damping
        # 设置资产的最大角速度
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        # 设置资产的最大线速度
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        # 设置资产的电枢
        asset_options.armature = self.cfg.asset.armature
        # 设置资产的厚度
        asset_options.thickness = self.cfg.asset.thickness
        # 设置是否禁用重力
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        # 加载机器人资产
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        # 获取资产的自由度数量
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        # 获取资产的刚体数量
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        # 获取资产的自由度属性
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        # 获取资产的刚体形状属性
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # 保存资产的刚体名称
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        # print("body_names:", body_names)
        # 保存资产的自由度名称
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        # print("dof_names:", self.dof_names)
        # 获取刚体数量
        self.num_bodies = len(body_names)
        # 获取自由度数量
        self.num_dofs = len(self.dof_names)
        # 获取脚部名称
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        # 获取需要惩罚接触的名称
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        # 获取需要终止接触的名称
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # 初始化基座状态
        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        # 设置初始姿态
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # 获取环境原点
        self._get_env_origins()
        # 设置环境的下限和上限
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        # 初始化演员句柄列表
        self.actor_handles = []
        # 初始化环境列表
        self.envs = []
        # 初始化摩擦系数
        self.friction_coef = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # 初始化恢复系数
        self.restitution_coef = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # 初始化基座质量
        self.base_mass = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # 初始化基座质心
        self.base_com = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            # 创建环境实例
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            # 计算位置
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-0.1, 0.1, (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            # 处理刚体形状属性
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            # 设置资产的刚体形状属性
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            # 创建演员
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)   # 开启关节力矩传感器
            # 处理自由度属性
            dof_props = self._process_dof_props(dof_props_asset, i)
            # 设置演员的自由度属性
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            # 获取演员的刚体属性
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            # 处理刚体属性
            body_props = self._process_rigid_body_props(body_props, i)
            # 设置演员的刚体属性
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            # 添加环境到列表
            self.envs.append(env_handle)
            # 添加演员句柄到列表
            self.actor_handles.append(actor_handle)

        # 初始化脚部索引
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        # 初始化需要惩罚接触的索引
        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        # 初始化需要终止接触的索引
        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.priv_obs_scales = self.cfg.normalization.priv_obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def pre_physics_step(self):
        self.rwd_linVelTrackPrev = self._reward_tracking_lin_vel() # 误差
        self.rwd_angVelTrackPrev = self._reward_tracking_ang_vel()
        # self.rwd_linVelTrackEnhancedPrev = self.tracking_integral_lin_vel() # 误差积分
        self.rwd_linVelTrackEnhancedPrev = self._reward_tracking_lin_vel_enhance() # 误差积分
        self.rwd_angVelTrackEnhancedPrev = self._reward_tracking_ang_vel_enhance()

    def tracking_integral_lin_vel(self):
        # 计算期望线性速度与实际线性速度之间的误差
        lin_vel_error = self.commands[:, 0] - self.base_lin_vel_head[:, 0]
        # lin_vel_error = torch.square(self.commands[:, 0] - self.wheel_lin_vel)
        ans = self.cfg.rewards.enhance_factor * self.rwd_linVelTrackEnhancedPrev + lin_vel_error
        # # 使用指数函数计算奖励，奖励值随着误差的减小而增加
        return ans  # Ensure the reward is non-negative
        # return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma / 10) - 1

    def update_stage(self):
        ##stage_buf: ["stand", "gait", "recover"] -> 两轮站立,步态,恢复两轮
        #当前为gait且步态命令false：进入stand
        need_gait = (torch.abs(self.commands[:, 5]) < self.cfg.commands.gait_train_proportion)                          # y方向有命令才使用步态
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
        if not hasattr(self, "last_stage"):
            self.last_stage = current_stage.clone()
        same_stage = current_stage == self.last_stage
        self.stage_time_buf = torch.where(
            same_stage,
            self.stage_time_buf + self.dt,
            torch.zeros_like(self.stage_time_buf)
        )
        self.last_stage = current_stage.clone()

    def update_gait_phase(self):
        period = self.cfg.commands.gait_period
        offset = 0.5
        self.gait_enable = self.stage_buf[:, 1] == 1.0  # gait
        phase = torch.zeros_like(self.stage_time_buf)
        # 只在 gait 阶段推进 phase
        phase[self.gait_enable] = (self.stage_time_buf[self.gait_enable] % period) / period
        self.phase = phase
        self.phase_left = phase
        self.phase_right = torch.where(self.gait_enable, (phase + offset) % 1.0, torch.zeros_like(phase))
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

    def _get_terrain_height_at_feet(self):
        """获取足端正下方的地形高度"""  
        # 足端的世界坐标(x, y)
        feet_xy = self.wheel_pos[:, :, :2]
        
        # 查询地形高度场
        # 需要根据你的地形实现来调整
        terrain_heights = self.terrain.get_heights_at_positions(
            feet_xy.reshape(-1, 2)
        ).reshape(self.num_envs, len(self.feet_indices))
        
        return terrain_heights
    
    def _update_contact_history(self):
        force_xy = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
        contact = (force_xy > 30.0).float()
        # 滑动窗口
        self.contact_history = torch.roll(self.contact_history, shifts=-1, dims=2)
        self.contact_history[:, :, -1] = contact

    # def _get_swing_stance_mask(self):
    #     # ===== 1. 接触检测 =====
    #     force_xy = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
    #     # print("force_xy:", force_xy[0])
    #     contact = force_xy > 30.0
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
    #     swing_duration = 0.3                                # 摆动周期
    #     no_swing = (~self.has_swing) & need_swing           # 当前无摆动腿 & 需要抬腿
    #     self.current_swing[no_swing] = candidate[no_swing]  # 当前摆动腿索引
    #     self.has_swing[no_swing] = True                     # 有摆动腿的环境
    #     self.swing_time[self.has_swing] += self.dt          # 有摆动腿的环境计时
    #     finished = self.swing_time > swing_duration         # 到摆动周期时长再结束摆动
    #     self.has_swing[finished] = False                    # 重置摆动状态
    #     self.swing_time[finished] = 0.0                     # 重置摆动时间

    #     # ===== 4. 输出 =====
    #     self.swing_mask[torch.arange(self.num_envs), self.current_swing] = self.has_swing.float()
    #     self.stance_mask = 1.0 - self.swing_mask
    #     # print("swing_mask:", self.swing_mask[0])

    def _get_swing_stance_mask(self):
        # ===== 1. 接触检测 =====
        force_xy = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
        # print("contact_force:", force_xy[0])
        contact = force_xy > 60.0
        stable = (self.contact_history.sum(dim=2) >= 2)     # 稳定接触（推荐 >=2 更鲁棒）
        need_swing = contact.sum(dim=1) > 0                 # 是否需要抬腿（平地不抬）
        # ===== 2. 选候选腿 =====
        candidate = torch.argmax(force_xy, dim=1)           # 默认：接触力大的,return:0-left_leg,1-right_leg
        one_contact = (contact.sum(dim=1) == 1)             # 如果单脚接触
        if one_contact.any():
            candidate[one_contact] = torch.argmax(contact[one_contact].float(), dim=1) # 抬单脚接触的那只脚
        one_stable = (stable.sum(dim=1) == 1)               # 如果双脚接触 & 单脚稳定接触,选稳定接触的脚
        mask = (contact.sum(dim=1) == 2) & one_stable
        if mask.any():
            candidate[mask] = torch.argmax(stable[mask].float(), dim=1)

        # ===== 3. swing状态机 =====
        swing_duration = 0.3                                # 摆动周期
        ds_duration = 0.1                                   # 双支撑时间
        in_ds = self.in_double_support
        self.ds_time[in_ds] += self.dt
        ds_finished = in_ds & (self.ds_time > ds_duration) & need_swing     # 双支撑结束,开始摆动
        if ds_finished.any():
            self.current_swing[ds_finished] = candidate[ds_finished]        # 当前摆动腿索引
            self.has_swing[ds_finished] = True                              # 有摆动腿的环境
            self.in_double_support[ds_finished] = False                     # 无双腿支撑的环境
            self.swing_time[ds_finished] = 0.0                              # 摆动时间重置
            self.ds_time[ds_finished] = 0.0                                 # 双腿支撑时间重置
        
        in_swing = self.has_swing                                           # 有摆动腿的环境
        self.swing_time[in_swing] += self.dt                                # 摆动时间增加
        swing_finished = in_swing &  (self.swing_time > swing_duration)     # 摆动结束,开始双支撑
        if swing_finished.any():
            self.has_swing[swing_finished] = False                          # 摆动结束
            self.in_double_support[swing_finished] = True                   # 进入双支撑状态
            self.swing_time[swing_finished] = 0.0                           # 摆动时间重置
            self.ds_time[swing_finished] = 0.0                              # 双支撑时间重置

        # ===== 4. 输出 =====
        self.swing_mask.zero_()
        self.swing_mask[torch.arange(self.num_envs), self.current_swing] = self.has_swing.float()
        self.stance_mask = 1.0 - self.swing_mask
        self.stance_mask[self.in_double_support] = 1.0

    def _apply_swing_feedforward(self, actions):
        """
        直接在 actions 上施加 swing 前馈（只作用在 swing 腿的 hip/knee）
        """
        if not self.cfg.commands.stair_command:
            return  actions
        # ===== 1. 没有摆动腿，直接返回 =====
        swing_envs = self.has_swing
        # if self.has_swing[0]:
        #     print("----0 is swing----,swing leg id is:", self.current_swing[0])
        #     print("----swing time----:", self.swing_time)
        if not swing_envs.any():
            return actions

        # ===== 2. swing phase =====
        swing_duration = 0.1
        swing_phase = self.swing_time / swing_duration
        swing_phase = torch.clamp(swing_phase, 0.0, 1.0)

        # ===== 3. 轨迹 =====
        hip_delta  = (0.2 * (1 - torch.cos(2 * np.pi * swing_phase)) - 0.3 * swing_phase) / self.cfg.control.action_scale_pos
        knee_delta = -0.4 * ( 1 - torch.cos(2 *np.pi * swing_phase)) / self.cfg.control.action_scale_pos
        # hip_delta  = 0.7 * (1 - torch.cos(2 * np.pi * swing_phase)) / self.cfg.control.action_scale_pos
        # knee_delta = -1.5 * ( 1 - torch.cos(2 * np.pi * swing_phase)) / self.cfg.control.action_scale_pos

        # ===== 4. 衰减系数 =====
        max_steps = 3000 * 48
        # print("step_counter:", self.common_step_counter)
        alpha = 1.0 - min(self.common_step_counter / max_steps, 1.0)

        # ===== 5. env & leg =====
        env_ids = torch.arange(self.num_envs, device=self.device)[swing_envs]
        leg_ids = self.current_swing[swing_envs]   # (k,)

        # ===== 6. joint mapping =====
        hip_index  = torch.tensor([1, 5], device=self.device)
        knee_index = torch.tensor([2, 6], device=self.device)

        hip_ids  = hip_index[leg_ids]   # (k,)
        knee_ids = knee_index[leg_ids]

        # ===== 7. 取对应 delta =====
        hip_ff  = hip_delta[swing_envs]   # (k,)
        knee_ff = knee_delta[swing_envs]

        # ===== 8. 当前动作 =====
        act_hip  = actions[env_ids, hip_ids]
        act_knee = actions[env_ids, knee_ids]

        # ===== 9. 混合（核心）=====
        actions[env_ids, hip_ids]  = act_hip  + alpha * hip_ff
        actions[env_ids, knee_ids] = act_knee + alpha * knee_ff
        # print("actions:", actions[0])
        return actions

    # ------------ reward functions----------------
    ################## 速度控制 ##################
    def _reward_tracking_goal(self):
        current_pos = self.root_states[:, :2]
        target_pos = self.cur_goals[:, :2]
        # print("cur_goals:", self.cur_goals[0])
        # norm = torch.norm(current_pos - target_pos, dim=-1)
        # rew = torch.exp(-5*norm) - 0.1*norm
        # return rew
        direction = target_pos - current_pos
        direction_norm = torch.norm(direction, dim=-1, keepdim=True) + 1e-6
        direction = direction / direction_norm
        # print("direction:", direction[0])
        vel = self.root_states[:, 7:9]
        # print("vel:", vel[0])
        rew = torch.sum(vel * direction, dim=-1)
        # print("rew:", rew[0])
        return rew

    def _reward_position(self):
        # 仅在回合最后2秒给予奖励，鼓励最终到达目标
        if self.episode_length_buf * self.dt > (self.max_episode_length_s - 2):
            # 计算当前位置与目标位置的距离
            pos_error = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1)
            rew = 10.0 / (1 + pos_error**2)  # 系数10.0来自论文
        else:
            rew = torch.zeros(self.num_envs, device=self.device)
        return rew

    def _reward_position(self):
        pos_error = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1)

        # dense reward（持续引导）
        rew_dense = 1.0 / (1.0 + pos_error)

        # terminal reward（最后强化）
        near_goal = pos_error < 0.2
        rew_terminal = near_goal.float() * 5.0

        return rew_dense + rew_terminal

    # 鼓励缩小与目标距离
    def _reward_goal_progress(self):
        dist = torch.norm(self.target_pos_rel, dim=1)
        progress = (self.last_dist - dist) / self.dt
        self.last_dist = dist
        return progress

    # 鼓励朝向目标方向前进
    def _reward_heading(self):
        current_pos = self.root_states[:, :2]
        target_pos = self.cur_goals[:, :2]
        delta = target_pos - current_pos
        target_yaw = torch.atan2(delta[:, 1], delta[:, 0])
        yaw = self.base_euler_zyx[:, 2]
        yaw_error = target_yaw - yaw
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
        reward = torch.cos(yaw_error)   # 或 exp(-2*yaw_error^2)
        return reward

    # 限制机器人速度不要太快
    def _reward_speed_penalty(self):
        speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        return speed
    # 到达目标给额外奖励
    def _reward_goal_reached(self):
        return self.reached_goal_ids.float()

    def _reward_tracking_lin_vel_x(self):
        """
        计算增强的线性速度跟踪奖励。

        该方法通过计算期望线性速度与实际线性速度之间的误差，并使用指数函数来计算奖励。
        奖励值随着误差的减小而增加，从而鼓励机器人更好地跟踪期望的线性速度。

        Returns:
            torch.Tensor: 增强的线性速度跟踪奖励。
        """
        # Tracking of linear velocity commands (x axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0]) / 0.1
        ans = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        return ans
    
    def _reward_tracking_lin_vel_y(self):
        gait_enable = self.stage_buf[:, 1]                                                          # 只在步态时才跟随y速度
        has_swing = (self.contact_forces[:, self.feet_indices, 2] < 1.0).any(dim=1).float()         # 只有至少有一只脚离地后，才开始跟随y速度
        # lin_vel_y_error = torch.square(self.commands[:, 1]  - self.base_lin_vel[:, 1]) / 0.1
        lin_vel_y_error = torch.abs(self.commands[:, 1]  - self.base_lin_vel[:, 1])
        rew_vy = torch.exp(-lin_vel_y_error / self.cfg.rewards.tracking_sigma)
        return rew_vy * gait_enable * has_swing

    def _reward_tracking_lin_vel_enhance(self):
        # 计算期望线性速度与实际线性速度之间的误差
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        # lin_vel_error = torch.square(self.commands[:, 0] - self.wheel_lin_vel)
        # ans = self.rwd_linVelTrackEnhancedPrev + self.cfg.rewards.enhance_factor * lin_vel_error
        # # 使用指数函数计算奖励，奖励值随着误差的减小而增加
        # return ans.clip(0, 1)  # Ensure the reward is non-negative
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma / 10) - 1
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ans = torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
        return ans

    def _reward_tracking_ang_vel_enhance(self):
        # Tracking of angular velocity commands (x axes)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ans = self.rwd_angVelTrackEnhancedPrev + self.cfg.rewards.enhance_factor * ang_vel_error
        return ans.clip(0, 1)  # Ensure the reward is non-negative

    def _reward_tracking_lin_vel_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_lin_vel() - self.rwd_linVelTrackPrev)
        # return lin_vel_error
        return delta_phi

    def _reward_tracking_ang_vel_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_ang_vel() - self.rwd_angVelTrackPrev)
        # return ang_vel_error
        return delta_phi

################## 位姿控制 ##################
    def _reward_base_height(self):
        # Penalize base height away from target
        if self.reward_scales["base_height"] < 0:
            return torch.abs(self.base_height - self.commands[:, 3])
        else:
            base_height_error = torch.square(self.base_height - self.commands[:, 3])
            ans = torch.exp(-base_height_error / 0.01)
        return ans
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_orientation(self):
        """
        计算保持基座平坦方向的奖励。使用基座欧拉角和投影重力向量来惩罚与期望基座方向的偏差。
        """
        ans = torch.exp(-torch.sum(torch.square(self.projected_gravity[:, :2]) * 5, dim=1))
        return ans
    
    def _reward_base_euler(self):
        roll_pitch = self.base_euler_zyx[:, :2]   # (N,2)
        threshold = 0.2
        exceed = torch.abs(roll_pitch) > threshold
        penalty = torch.sum(exceed.float(), dim=1)
        return penalty

    # def _reward_base_euler(self):
    #     ans = torch.exp(-torch.sum(torch.abs(self.base_euler_zyx[:, :2]), dim=1) / 0.1)
    #     return ans

    def _reward_leg_end_x_diff(self):
        if self.cfg.commands.gait_command:
            stand_enable = self.stage_buf[:, 0]                    
            ans = torch.exp(-torch.abs(self.wheel_pos_left_local_x - self.wheel_pos_right_local_x) / 0.1)
            return ans * stand_enable
        else:
            ans = torch.abs(self.wheel_pos_left_local_x - self.wheel_pos_right_local_x)
            return ans

    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.abs(self.wheel_pos[:, 0, 1] - self.wheel_pos[:, 1, 1])
        too_close_penalty = torch.relu(self.cfg.rewards.min_feet_distance - feet_distance)  # 相当于 max(0, d_min - d)
        too_far_penalty = torch.relu(feet_distance - self.cfg.rewards.max_feet_distance)    # 相当于 max(0, d - d_max)
        reward = too_close_penalty + too_far_penalty
        # # 二值化惩罚：距离小于最小安全距离时惩罚为1，否则为0
        # reward = torch.where(
        #     feet_distance < self.cfg.rewards.min_feet_distance,
        #     torch.ones_like(feet_distance),                             # 距离不足时惩罚=1
        #     torch.zeros_like(feet_distance),                            # 距离足够时惩罚=0
        # )
        return reward
    
    def _reward_hip_pos(self):
        if self.cfg.commands.gait_command:
            gait_enable  = self.stage_buf[:, 1].bool()
            hip_error = torch.sum(torch.square(self.dof_pos[:, [0,4]]), dim=1)
            balance_scale = 5.0
            gait_scale = 0.0
            scale = torch.where(gait_enable, gait_scale, balance_scale)
            rew_hip = scale * hip_error
        else:
            hip_error = torch.sum(torch.square(self.dof_pos[:, [0,4]]), dim=1)
            rew_hip = hip_error
        return rew_hip

################## 步态控制 ##################
    def _reward_enter_gait(self):                                       # 进入步态直接给奖励
        enter_gait = (self.stage_buf[:,1] == 1) & (self.stage_time_buf < self.dt)
        rew_enter = enter_gait.float()
        return rew_enter
    
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
        return rew_airTime * 100
    
    # def _reward_contact(self): 
    #     gait_enable  = self.stage_buf[:, 1]                                 # 只在 gait
    #     fz = self.contact_forces[:, self.feet_indices, 2]                   # 接触力z,维度[N, 2]
    #     # print("fz", fz)
    #     contact = (fz > 1.0).float()                                        # 是否接触
    #     is_stance = (self.leg_phase < 0.55).float()                         # 支撑脚和摆动脚,维度[N, 2]
    #     rew_contact = is_stance * contact + (1 - is_stance) * (1 - contact) # 支撑腿接触,摆动腿不接触
    #     rew_gait = torch.mean(rew_contact, dim=1)                            # 步态奖励
    #     return rew_gait * gait_enable

    def _reward_contact(self): 
        fz = self.contact_forces[:, self.feet_indices, 2]                   # 接触力z,维度[N, 2]
        contact = (fz > 1.0).float()                                        # 是否接触
        rew_contact = - self.swing_mask * contact + self.swing_mask * (1 - contact) # 支撑腿接触,摆动腿不接触
        # print("swing:", self.swing_mask[0])
        rew_gait = torch.mean(rew_contact, dim=1)                            # 步态奖励
        return rew_gait * 20
    
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
    #     height_err = torch.square(z - z_ref)    
    #     track_reward = -20 * height_err                                 # 跟踪轨迹
    #     mid_mask = (torch.abs(swing_phase - 0.5) < 0.15).float()
    #     peak_err = (z - z_target)**2                                    # 尖峰误差
    #     peak_reward = -40.0 * peak_err * mid_mask                          # 尖峰惩罚
    #     lift_reward = 5.0 * torch.clamp(z-z_min, min=0.0)
    #     rew_total = track_reward + peak_reward + lift_reward
    #     rew = torch.sum(rew_total * swing, dim=1)   # 只对摆动脚起作用
    #     return rew * gait_enable

    def _reward_feet_clearance(self):
        """
        改进版：使用连续函数替代二值判断，提供更平滑的梯度。
        """
        foot_height = self.wheel_pos[:, :, 2]
        terrain_height = self._get_terrain_height_at_feet()
        clearance = foot_height - terrain_height - self.cfg.asset.wheel_radius
        # clearance = foot_height - self.cfg.asset.wheel_radius
        # print("clearance:", clearance[0])

        h_min = 0.16
        h_max = 0.18
        swing_mask = self.swing_mask  # (num_envs, 2)
        # in_range = (clearance > h_min) & (clearance < h_max)
        # # 1. 计算“在区间内”的连续得分（使用sigmoid函数产生平滑过渡）
        # # 例如，可以分别计算“高于h_min”和“低于h_max”的得分，然后相乘
        score_min = torch.sigmoid((clearance - h_min) * 20)  # 乘100控制过渡陡峭度
        score_max = torch.sigmoid((h_max - clearance) * 20)
        in_range_score = score_min * score_max  # 当clearance在(h_min, h_max)时接近1，否则接近0
        lift_reward = 5.0 * torch.clamp(clearance, min=0.0, max=h_max)
        # 2. 只对摆动腿应用此得分
        reward_per_leg = ((in_range_score + lift_reward) * swing_mask)
        # 3. 对所有腿的得分求和（鼓励所有摆动腿都达标）
        total_reward = reward_per_leg.sum(dim=1)
        return total_reward

    def _reward_feet_swing_height(self):
        z = self.wheel_pos[:, :, 2]                         # (N,2)
        swing = self.swing_mask                             # (N,2)
        z_min = self.cfg.asset.wheel_radius
        z_target = self.cfg.commands.gait_foot_height
        delta_z = z_target - z_min
        # ===== 用时间定义 swing phase =====
        swing_duration = 0.3
        swing_phase = self.swing_time.unsqueeze(1) / swing_duration
        swing_phase = torch.clamp(swing_phase, 0.0, 1.0)
        # ===== 参考轨迹 =====
        z_ref = z_min + delta_z * torch.sin(np.pi * swing_phase) ** 2
        # ===== 1. 轨迹跟踪 =====
        height_err = (z - z_ref) ** 2
        track_reward = -20.0 * height_err
        # ===== 2. 中间抬高约束 =====
        mid_mask = (torch.abs(swing_phase - 0.5) < 0.15).float()
        peak_err = (z - z_target) ** 2
        peak_reward = -40.0 * peak_err * mid_mask
        # ===== 3. 抬腿激励 =====
        lift_reward = 5.0 * torch.clamp(z - z_min, min=0.0)
        # ===== 总奖励 =====
        rew_total = track_reward + peak_reward + lift_reward
        # 👉 只对 swing 腿生效
        rew = torch.sum(rew_total * swing, dim=1)
        return rew
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.wheel_body_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, 1:3])
        return torch.sum(penalize, dim=(1,2))

    def _reward_swing_no_wheel_vel(self):
        swing_mask = self.swing_mask
        wheel_vel = self.dof_vel[:, self.wheel_indices]                             # 轮子速度
        air_wheel_wheel = torch.square(wheel_vel)                                   # 步态时轮子的速度
        wheel_vel = torch.sum(air_wheel_wheel * swing_mask, dim=1)                               # 两个轮子的速度和
        rew_wheel_vel = torch.exp(-wheel_vel)
        return rew_wheel_vel                                       # 惩罚步态时轮子速度

    def _reward_gait_no_wheel_vel(self):
        gait_enable  = self.stage_buf[:, 1]                                         # 只在 gait
        wheel_vel = self.dof_vel[:, self.wheel_indices]                             # 轮子速度
        air_wheel_wheel = torch.square(wheel_vel)                                   # 步态时轮子的速度
        wheel_vel = torch.sum(air_wheel_wheel, dim=1)                               # 两个轮子的速度和
        rew_wheel_vel = torch.exp(-0.01 * wheel_vel)
        return rew_wheel_vel * gait_enable                                          # 惩罚步态时轮子速度

    def _reward_feet_contact_forces(self):
        """
        惩罚足部垂直接触力超过阈值的情况
        公式：max(0, F_z - F_max) × -5.0
        """
        # 获取足部垂直接触力
        # self.contact_forces 形状: (num_envs, num_bodies, 3)
        # self.feet_indices 应该是包含左右足部索引的列表
        feet_z_forces = self.contact_forces[:, self.feet_indices, 2]  # Z轴力，形状: (num_envs, 2)
        # 设置最大允许力阈值（单位：牛顿）
        F_max = self.cfg.rewards.max_wheel_contact_force  # 例如50N，需要根据你的机器人调整
        # 计算超出阈值的部分
        excess_force = torch.relu(feet_z_forces - F_max)  # relu等同于max(0, x)
        # 对两个足部求和，然后乘以惩罚系数
        penalty = torch.sum(excess_force, dim=1)  # 对两个足部求和
        return penalty

################## 动作柔顺 ##################
    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel[:, self.joint_indices]), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc[:, self.joint_indices]), dim=1)
    
    def _reward_torques(self):
        # Penalize torques
        # print("sum torques:", torch.sum(torch.square(self.torques), dim=1))
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:, :, 0] - self.actions), dim=1)

    def _reward_action_smooth(self):
        """
        鼓励机器人动作的平滑性,通过惩罚连续动作之间的大差异。
        这对于实现流畅的运动和减少机械应力很重要。
        """
        ans = torch.sum(
            torch.square(self.actions[:, self.joint_indices] + self.last_actions[:, self.joint_indices, 1] - 2 * self.last_actions[:, self.joint_indices, 0]), dim=1
        )
        return ans
    
    def _reward_energy(self):
        """能耗惩罚：让空转变得昂贵"""
        # 扭矩和关节速度的乘积
        energy = torch.abs(self.torques * self.dof_vel).sum(dim=1)
        return energy
################## 关节限制 ##################
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        dof_pos_subset = self.dof_pos[:, self.joint_indices]
        lower_limits = self.dof_pos_limits[self.joint_indices, 0]
        upper_limits = self.dof_pos_limits[self.joint_indices, 1]
        lower_penalty = -(dof_pos_subset - lower_limits).clip(max=0.0)  # 下限惩罚
        upper_penalty = (dof_pos_subset - upper_limits).clip(min=0.0)  # 上限惩罚
        penalties = lower_penalty + upper_penalty
        return torch.sum(penalties, dim=1)
    
    def _reward_wheel_slip(self):
        # 轮子角速度
        wheel_vel = self.dof_vel[:, self.wheel_indices]
        # 机器人前向速度
        base_lin_vel_x = self.base_lin_vel[:, 0].unsqueeze(1)
        # 轮子线速度
        wheel_lin_vel = wheel_vel * self.cfg.asset.wheel_radius
        # slip
        slip = torch.abs(wheel_lin_vel - base_lin_vel_x)
        # 平均两个轮子
        return torch.mean(slip, dim=1)
    
    def _reward_stuck(self):
        wheel_vel = torch.abs(self.dof_vel[:, self.wheel_indices])
        base_vel = torch.abs(self.base_lin_vel[:, 0]).unsqueeze(1)
        stuck = ((wheel_vel > 1.5) & (base_vel < 0.05)).float()
        return torch.mean(stuck, dim=1)
    
################## 碰撞惩罚 ##################
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
            dim=1,
        )

    def _reward_alive(self):
        """生存奖励：只要机器人还活着就给予固定奖励"""
        return torch.ones(self.num_envs, device=self.device)

    def _reward_successful_lift(self):
        """奖励成功抬腿越过障碍 - 只在需要时"""
        # 1. 检测是否需要抬腿（撞到东西了）
        force_x = self.contact_forces[:, self.feet_indices, 0]
        horizontal_force = torch.abs(force_x)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        # 当前是否撞到东西（水平力大且接触地面）
        hitting_obstacle = (horizontal_force > 50.0) & contact  # [num_envs, num_feet]
        # 2. 检测抬腿动作
        feet_height = self.wheel_pos[:, :, 2]
        terrain_height = self._get_terrain_height_at_feet()
        lift = feet_height - (terrain_height + self.cfg.asset.wheel_radius)
        # 3. 检测成功越过（抬腿后，之前撞到的腿现在水平力变小了）
        # 记录上一时刻是否在撞
        was_hitting = self.was_hitting.clone()
        self.was_hitting = hitting_obstacle
        # 成功越过 = 之前撞到 + 现在不撞了 + 抬腿足够高
        successful = was_hitting & (~hitting_obstacle) & (lift > 0.05)
        # 4. 奖励
        success_reward = successful.any(dim=1).float() * 200.0
        return success_reward

    def _reward_stumble(self):
        """精细的撞楼梯惩罚"""
        # 1. 水平力
        force_x = self.contact_forces[:, self.feet_indices, 0]  # 前进方向力
        force_y = self.contact_forces[:, self.feet_indices, 1]  # 侧向力
        horizontal_force = torch.sqrt(force_x**2 + force_y**2)
        # 2. 垂直力
        force_z = self.contact_forces[:, self.feet_indices, 2]
        # 3. 只有轮子接触地面
        contact = force_z > 1.0
        # 4. 水平力/垂直力 比例过大（撞到东西）
        # 或者水平力绝对值过大
        stumble = (horizontal_force > 5 * force_z) | (horizontal_force > 60.0)
        stumble = stumble & contact
        # 5. 可选：增加渐变惩罚，而不是二值
        penalty = torch.clamp(horizontal_force - 10.0, min=0.0) / 10.0  # 0-?
        # 返回惩罚（负值）
        return torch.any(stumble, dim=1).float() * 20.0

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
        return total_slip