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

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from wheel_legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg                                                      # 地形配置
        self.num_robots = num_robots                                        # 环境数量
        self.type = cfg.mesh_type                                           # 地形网格类型
        if self.type in ["none", "plane"]:
            return
        self.env_length = cfg.terrain_length                                # 地形长度
        self.env_width = cfg.terrain_width                                  # 地形宽度
        self.proportions = [                                                # 地形比例
            np.sum(cfg.terrain_proportions[: i + 1])
            for i in range(len(cfg.terrain_proportions))
        ]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols             # 子地形数量
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))        # 地形零点
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, cfg.num_goals, 3))               # 每个子地形里定义多个目标点
        self.num_goals = cfg.num_goals                                                      # 目标点数量
        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)              # 地形宽度像素值
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)            # 地形长度像素值

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)                      # 地形边界
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border     # 整个地图列像素
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border    # 整个地图行像素

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)    # 整张地图像素点高度全部置零,相当于一张空地图
        if cfg.curriculum:                                      # 课程学习
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":                              # 把heightfield地形转换成三角网格地形,以便物理引擎使用
            self.vertices, self.triangles = (
                terrain_utils.convert_heightfield_to_trimesh(
                    self.height_field_raw,
                    self.cfg.horizontal_scale,
                    self.cfg.vertical_scale,
                    self.cfg.slope_treshold,
                )
            )
        # 构建高度查询器
        self._build_height_querier()

    def _build_height_querier(self):
            """构建高度查询器"""
            # 将高度场转换为float类型并缩放到实际高度
            self.height_field = self.height_field_raw.astype(np.float32) * self.cfg.vertical_scale
            
            # 计算地形原点（左下角世界坐标）
            self.origin_x = -self.border * self.cfg.horizontal_scale
            self.origin_y = -self.border * self.cfg.horizontal_scale
            
            # 计算网格信息
            self.grid_rows = self.tot_rows
            self.grid_cols = self.tot_cols
            self.resolution = self.cfg.horizontal_scale

    def randomized_terrain(self):                               # 生成随机地形
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows                                  # 地形难度
                choice = j / self.cfg.num_cols + 0.001                              # 选择一种地形

                terrain = self.make_terrain(choice, difficulty)                     # 选择一种地形及难度,相当于给一块地图配置好地形信息
                self.add_terrain_to_map(terrain, i, j)                              # 把配置好的地形加载到总地图

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop("type")                          # 从配置文件读取地形函数名字
        for k in range(self.cfg.num_sub_terrains):                                  # 循环生成多个子地形
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))    # 生成二维坐标

            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.width_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale,
            )

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)   # 调用地形生成函数
            self.add_terrain_to_map(terrain, i, j)                                  # 把子地形放入地图

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,                    # 整张地图的宽度像素
            length=self.width_per_env_pixels,                   # 整张地图的长度像素
            vertical_scale=self.cfg.vertical_scale,             # 垂直缩放比例,对应z方向1代表多少
            horizontal_scale=self.cfg.horizontal_scale,         # 水平缩放比例,对应x-y平面1代表多少
        )
        slope = difficulty * 0.5                                # 斜率
        random_height = 0.05 + difficulty * 0.05                # 随机高度
        step_height = 0.06 + 0.1 * difficulty                  # 台阶高度
        # step_height = 0.08                  # 台阶高度
        discrete_obstacles_height = 0.03 + difficulty * 0.1     # 离散障碍物高度
        stepping_stones_size = 1.5 * (1.05 - difficulty)        # 石头尺寸
        stone_distance = 0.05 if difficulty == 0 else 0.1       # 石头距离
        gap_size = 1.0 * difficulty                             # 沟宽度尺寸
        pit_depth = 1.0 * difficulty                            # 坑深度尺寸
        if choice < self.proportions[0]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=0, platform_size=3.0)       # 完全平地
        elif choice < self.proportions[1]:
            if (
                choice
                < self.proportions[0] + (self.proportions[1] - self.proportions[0]) / 2
            ):
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(                                           # 一半上坡,一半下坡
                terrain, slope=slope, platform_size=3.0
            )
        elif choice < self.proportions[2]:
            if (
                choice
                < self.proportions[1] + (self.proportions[2] - self.proportions[1]) / 2
            ):
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope * 0.5, platform_size=3.0
            )
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-random_height,      # 随机高度范围
                max_height=random_height,
                step=0.005,                     # 高度变化的最小单位
                downsampled_scale=0.2,          # 控制rough terrain的"空间尺度",也就是凸起之间的距离
            )
        elif choice < self.proportions[4]:
            if choice < self.proportions[3]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(       # 生成台阶地形
                terrain, step_width=0.7, step_height=step_height, platform_size=4.0
            )
            num_goals = self.num_goals
            terrain.goals = np.zeros((num_goals,3))
            step_width = 0.7
            for k in range(num_goals):
                # terrain.goals[k] = [5.0 + 0.6 * k * step_width, 4.0, 0.2]
                terrain.goals[k] = [9.0, 4.0, 0.2]

        elif choice < self.proportions[5]:
            num_rectangles = 4                         # 随机矩形障碍的数量
            rectangle_min_size = 2.5                    # 随机矩形障碍的尺寸
            rectangle_max_size = 3.5
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                discrete_obstacles_height,              # 随机矩形障碍的高度
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=4.0,
            )
            # step_height = discrete_obstacles_height
            # # print("step:", terrain.length)
            # step_start = int(terrain.length * 0.7)
            # step_end = int(terrain.length * 0.75)
            # # print("start", step_start)

            # terrain.height_field_raw[step_start:step_end,:] = step_height / 0.005#step_height
            num_goals = self.num_goals
            terrain.goals = np.zeros((num_goals,3))
            step_width = 0.7
            for k in range(num_goals):
                terrain.goals[k] = [5.0 + 0.6 * k * step_width, 4.0, 0.2]
                # terrain.goals[k] = [8.0, 4.0, 0.2]
        elif choice < self.proportions[6]:              
            terrain_utils.stepping_stones_terrain(      # 踏石地形
                terrain,
                stone_size=stepping_stones_size,        # 石头尺寸
                stone_distance=stone_distance,          # 石头间隔
                max_height=0.0,                         # 石头高度随机变化范围,0代表所有石头高度一样
                platform_size=4.0,                      # 平台尺寸
            )
        elif choice < self.proportions[7]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.0)  # 宽沟地形
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.0)    # 深坑地形

        return terrain

    def add_terrain_to_map(self, terrain, row, col):                    # 加载地形,设置原点
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2.0 - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2.0 + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 1) / terrain.horizontal_scale)
        env_origin_z = (
            np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        )
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.goals[i, j, :, :3] = terrain.goals + [i * self.env_length, j * self.env_width, 0]             # 把子地形goals映射到世界坐标,第i,j个环境的所有目标点的x,y值

    def get_heights_at_positions(self, positions):
            """
            获取任意位置的地形高度（双线性插值）
            
            Args:
                positions: [N, 2] or [B, N, 2] tensor - 世界坐标(x, y)
            
            Returns:
                heights: [N] or [B, N] tensor - 地形高度
            """
            import torch
            
            # 保存原始形状
            original_shape = positions.shape[:-1]
            
            # 重塑为 [total_points, 2]
            if len(positions.shape) == 3:
                batch_size, num_points, _ = positions.shape
                positions_flat = positions.reshape(-1, 2)
            else:
                positions_flat = positions
            
            # 转换到设备
            if not isinstance(positions_flat, torch.Tensor):
                positions_flat = torch.tensor(positions_flat, dtype=torch.float)
            
            device = positions_flat.device
            
            # 转换到网格坐标
            # 世界坐标 -> 像素坐标
            px = (positions_flat[:, 0] - self.origin_x) / self.resolution
            py = (positions_flat[:, 1] - self.origin_y) / self.resolution
            
            # 边界裁剪
            px = torch.clamp(px, 0, self.grid_cols - 1.001)
            py = torch.clamp(py, 0, self.grid_rows - 1.001)
            
            # 整数部分和分数部分
            px0 = px.long()
            py0 = py.long()
            px1 = (px0 + 1).clamp(max=self.grid_cols - 1)
            py1 = (py0 + 1).clamp(max=self.grid_rows - 1)
            
            fx = px - px0.float()
            fy = py - py0.float()
            
            # 将高度场转换为tensor（如果还没转换）
            if not hasattr(self, 'height_field_tensor'):
                self.height_field_tensor = torch.tensor(
                    self.height_field, device=device, dtype=torch.float
                )
            elif self.height_field_tensor.device != device:
                self.height_field_tensor = self.height_field_tensor.to(device)
            
            # 获取四个角的高度
            # 注意：height_field的索引是 [row, col] = [y, x]
            h00 = self.height_field_tensor[py0, px0]
            h01 = self.height_field_tensor[py1, px0]
            h10 = self.height_field_tensor[py0, px1]
            h11 = self.height_field_tensor[py1, px1]
            
            # 双线性插值
            h0 = h00 * (1 - fx) + h10 * fx
            h1 = h01 * (1 - fx) + h11 * fx
            heights = h0 * (1 - fy) + h1 * fy
            
            # 恢复原始形状
            heights = heights.reshape(*original_shape)
            
            return heights
        
    # ==================== 新增方法3：单点查询 ====================
    def get_height_at_point(self, x, y):
        """
        获取单点高度（用于调试）
        
        Args:
            x, y: 世界坐标
        Returns:
            height: 地形高度
        """
        import torch
        pos = torch.tensor([[x, y]])
        return self.get_heights_at_positions(pos)[0].item()
    
    # ==================== 新增方法4：查询足端下的地形 ====================
    def get_heights_at_feet_positions(self, feet_positions):
        """
        专门为足端查询优化的方法
        
        Args:
            feet_positions: [num_envs, num_feet, 3] - 足端世界坐标
        Returns:
            terrain_heights: [num_envs, num_feet] - 地形高度
        """
        # 提取x,y坐标
        feet_xy = feet_positions[..., :2]
        
        # 查询高度
        return self.get_heights_at_positions(feet_xy)

def gap_terrain(terrain, gap_size, platform_size=1.0):                  # 挖沟
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[
        center_x - x2 : center_x + x2, center_y - y2 : center_y + y2
    ] = -1000
    terrain.height_field_raw[
        center_x - x1 : center_x + x1, center_y - y1 : center_y + y1
    ] = 0


def pit_terrain(terrain, depth, platform_size=1.0):                     # 挖坑
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
