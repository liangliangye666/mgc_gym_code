import torch
import numpy as np
import os
import math

from isaacgym.torch_utils import *
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.envs.base.legged_robot import LeggedRobot
from wheel_legged_gym.utils.terrain import Terrain
from wheel_legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
    get_euler_zyx_tensor,
    quat_from_euler_zyx,
    get_euler_tensor,
    get_heading_quaternion,
)
from wheel_legged_gym.utils.helpers import class_to_dict
from .y4a_walk_config import Y4A_WALK_Cfg

from tqdm import tqdm


class Y4A_WALK(LeggedRobot):

    def __init__(
        self, cfg: Y4A_WALK_Cfg, sim_params, physics_engine, sim_device, headless
    ):
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
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None

        self.init_done = False
        self._parse_cfg(self.cfg)

        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(
            self.sim_device
        )
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == "cuda" and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = "cpu"

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_critic_obs = cfg.env.num_critic_observations

        self.num_actions = cfg.env.num_actions
        self.obs_history_length = cfg.env.obs_history_length
        self.num_commands = cfg.commands.num_commands

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(
            self.num_envs, self.num_obs, device=self.device, dtype=torch.float
        )
        self.critic_obs_buf = torch.zeros(
            self.num_envs,
            self.num_critic_obs,
            device=self.device,
            dtype=torch.float,
        )
        self.obs_history = torch.zeros(
            self.num_envs,
            self.num_obs * self.obs_history_length,
            device=self.device,
            dtype=torch.float,
        )
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.fail_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.stable_episode_length_count = 0
        self.mean_episode_len = 0
        self.walk_stability = False
        self.stand_still_stability = False
        self.envs_steps_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.edge_reset_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties()
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )

        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2

        self.group_idx = torch.arange(0, self.cfg.env.num_envs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)

        Returns:
            obs (torch.Tensor): Tensor of shape (num_envs, num_observations_per_env)
            rewards (torch.Tensor): Tensor of shape (num_envs)
            dones (torch.Tensor): Tensor of shape (num_envs)
        """

        if self.cfg.domain_rand.randomize_action_delay:
            if (
                self.common_step_counter
                % self.cfg.domain_rand.delay_update_global_steps
                == 0
            ):
                action_delay_idx = torch.round(
                    torch_rand_float(
                        self.cfg.domain_rand.delay_ms_range[0]
                        / 1000
                        / self.sim_params.dt,
                        self.cfg.domain_rand.delay_ms_range[1]
                        / 1000
                        / self.sim_params.dt,
                        (self.num_envs, 1),
                        device=self.device,
                    )
                ).squeeze(-1)
                self.action_delay_idx = action_delay_idx.long()

        self._action_clip(actions)
        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )
            self.envs_steps_buf += 1

            self.torques = self._compute_torques(
                self.action_fifo[
                    torch.arange(self.num_envs), self.action_delay_idx, :
                ]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
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
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
            self.commands[:, :3] * self.commands_scale,
            self.critic_obs_buf,  # make sure critic_obs update in every for loop
        )

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = (
            self.command_ranges["lin_vel_x"][env_ids, 1]
            - self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_x"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 1] = (
            self.command_ranges["lin_vel_y"][env_ids, 1]
            - self.command_ranges["lin_vel_y"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_y"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 2] = (
            self.command_ranges["ang_vel_yaw"][env_ids, 1]
            - self.command_ranges["ang_vel_yaw"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "ang_vel_yaw"
        ][
            env_ids, 0
        ]
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        # set small commands to zero
        # self.commands[env_ids, :2] *= (
        #     torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.min_norm
        # ).unsqueeze(1)
        zero_command_idx = (
            (
                torch_rand_float(0, 1, (len(env_ids), 1), device=self.device)
                < self.cfg.commands.zero_command_prob
            )
            .squeeze(1)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.commands[zero_command_idx, :3] = 0
        if self.cfg.commands.heading_command:
            forward = quat_apply(
                self.base_quat[zero_command_idx],
                self.forward_vec[zero_command_idx],
            )
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[zero_command_idx, 3] = heading

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        self.actions_scaled = actions * self.cfg.control.action_scale

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (
                self.p_gains
                * (self.actions_scaled + self.default_dof_pos - self.dof_pos)
                - self.d_gains * self.dof_vel
            )
        elif control_type == "V":
            torques = (
                self.p_gains * (self.actions_scaled - self.dof_vel)
                - self.d_gains
                * (self.dof_vel - self.last_dof_vel)
                / self.sim_params.dt
            )
        elif control_type == "T":
            torques = self.actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(
            torques * self.torques_scale,
            -self.torque_limits,
            self.torque_limits,
        )

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
        noise_vec[0:3] = (
            noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        )
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:12] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[12:18] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[18:] = 0.0  # previous actions + clock + gait
        return noise_vec

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
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(
                as_tuple=False
            ).flatten()
            self.update_command_curriculum(time_out_env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        # Reset average velocity buffers
        self.avg_base_lin_vel[env_ids] = 0.0
        self.avg_base_ang_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history[env_ids] = 0
        obs_buf, _ = self.compute_group_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(
            1, self.obs_history_length
        )
        self.gait_indices[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids])
                / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["group_terrain_level"] = torch.mean(
                self.terrain_levels[self.group_idx].float()
            )
            self.extras["episode"]["group_terrain_level_stair_up"] = torch.mean(
                self.terrain_levels[self.stair_up_idx].float()
            )
        if self.cfg.terrain.curriculum and self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][
                    self.smooth_slope_idx, 1
                ].float()
            )
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler = get_euler_tensor(self.base_quat)

    def compute_group_observations(self):
        # note that observation noise need to modified accordingly !!!
        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.projected_gravity,  # 3
                (self.dof_pos - self.default_dof_pos)
                * self.obs_scales.dof_pos,  # 6
                self.dof_vel * self.obs_scales.dof_vel,  # 6
                self.actions,  # 6
                self.clock_inputs_sin.view(self.num_envs, 1),  # 1
                self.clock_inputs_cos.view(self.num_envs, 1),  # 1
                self.gaits,  # num_gait_params 4
            ),
            dim=-1,
        )
        critic_obs_buf = torch.cat(
            (self.base_lin_vel * self.obs_scales.lin_vel, self.obs_buf),
            dim=-1,
        )
        return obs_buf, critic_obs_buf

    # ----------------------------------------

    def get_observations(self):
        return (
            self.obs_buf,
            self.obs_history,
            self.commands[:, :3] * self.commands_scale,
            self.critic_obs_buf,
        )

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            rew = torch.clip(
                rew,
                -self.cfg.rewards.clip_single_reward,
                self.cfg.rewards.clip_single_reward,
            )
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        self.rew_buf[:] = torch.clip(
            self.rew_buf[:],
            -self.cfg.rewards.clip_reward,
            self.cfg.rewards.clip_reward,
        )
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def create_sim(self):
        """Creates simulation, terrain and evironments"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]"
            )
        self._create_envs()

    def check_termination(self):
        """Check if environments need to be reset"""
        fail_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :],
                dim=-1,
            )
            > 10.0,
            dim=1,
        )
        fail_buf |= self.projected_gravity[:, 2] > -0.1
        fail_buf |= self.base_height < 0.5  # 检查高度是否低于阈值
        self.fail_buf += fail_buf
        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )  # no terminal reward for time-outs
        self.power_limit_out_buf = (
            torch.sum(self.power, dim=1) > self.cfg.control.max_power
        )
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.edge_reset_buf = (
                self.base_position[:, 0] > self.terrain_x_max - 1
            )
            self.edge_reset_buf |= (
                self.base_position[:, 0] < self.terrain_x_min + 1
            )
            self.edge_reset_buf |= (
                self.base_position[:, 1] > self.terrain_y_max - 1
            )
            self.edge_reset_buf |= (
                self.base_position[:, 1] < self.terrain_y_min + 1
            )
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
            | self.edge_reset_buf
            # | self.power_limit_out_buf
        )

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(
            WHEEL_LEGGED_GYM_ROOT_DIR=WHEEL_LEGGED_GYM_ROOT_DIR
        )
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = (
            self.cfg.asset.default_dof_drive_mode
        )
        asset_options.collapse_fixed_joints = (
            self.cfg.asset.collapse_fixed_joints
        )
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.flip_visual_attachments = (
            self.cfg.asset.flip_visual_attachments
        )
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(
            robot_asset
        )

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        hip_roll_joint_names = [s for s in self.dof_names if "hip_roll" in s]

        print("Body names:", body_names)
        print("Feet names:", feet_names)
        print("hip roll joint names:", hip_roll_joint_names)
        print("DOF names:", self.dof_names)

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend(
                [s for s in body_names if name in s]
            )

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []
        masses_list = []
        local_coms_list = []

        self.friction_coef = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.restitution_coef = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.base_mass = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.base_com = torch.zeros(
            self.num_envs,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.whole_body_mass = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        print("Creating env...")
        for i in tqdm(range(self.num_envs)):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(
                -1.0, 1.0, (2, 1), device=self.device
            ).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i
            )

            self.gym.set_asset_rigid_shape_properties(
                robot_asset, rigid_shape_props
            )
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(
                env_handle, actor_handle, dof_props
            )
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True
            )

            masses_for_this_env = [p.mass for p in body_props]
            local_coms_for_this_env = [
                [p.com.x, p.com.y, p.com.z] for p in body_props
            ]
            masses_list.append(masses_for_this_env)
            local_coms_list.append(local_coms_for_this_env)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.masses_tensor = torch.tensor(
            masses_list, dtype=torch.float, device=self.device
        )
        self.local_coms_tensor = torch.tensor(
            local_coms_list, dtype=torch.float, device=self.device
        )

        self.feet_indices = torch.zeros(
            len(feet_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        print("self.feet_indices: ", self.feet_indices)


        self.hip_roll_joint_indices = torch.zeros(
            len(hip_roll_joint_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(hip_roll_joint_names)):
            self.hip_roll_joint_indices[i] = self.gym.find_actor_dof_handle(
                self.envs[0], self.actor_handles[0], hip_roll_joint_names[i]
            )

        print("self.hip_roll_joint_indices: ", self.hip_roll_joint_indices)

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = (
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0],
                    self.actor_handles[0],
                    penalized_contact_names[i],
                )
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = (
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0],
                    self.actor_handles[0],
                    termination_contact_names[i],
                )
            )

    def get_foot_contacts(self):
        contact = (
            torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.0
        )
        # 如果至少一个输入为True，则输出为True；否则输出为False
        foot_contacts_bool = torch.logical_or(
            contact, self.last_contacts
        )  # 减少假阴性
        self.last_contacts = contact
        return foot_contacts_bool

    def get_com_dir_vec(self) -> torch.Tensor:

        feet_midpoint = torch.mean(self.foot_positions, dim=1)

        vec_base_to_feet_W = feet_midpoint - self.base_position

        # 朝向系下轮足中点的位置
        feet_H = quat_rotate_inverse(self.q_WH, vec_base_to_feet_W)

        vec_feet_to_com = self.robot_com - feet_H

        unit_vector = torch.nn.functional.normalize(
            vec_feet_to_com, p=2, dim=-1, eps=1e-8
        )

        return unit_vector

    def get_robot_com_pos_heading(self) -> torch.Tensor:
        """
        获取朝向系下机器人质心的位置
        """
        # self.rigid_body_state 是 (self.num_envs, self.num_bodies, 13) 的形状
        body_pos = self.rigid_body_state[
            :, :, 0:3
        ]  # (self.num_envs, self.num_bodies, 3)
        body_rot = self.rigid_body_state[
            :, :, 3:7
        ]  # (self.num_envs, self.num_bodies, 4)

        # 将局部质心向量从局部坐标系旋转到世界坐标系
        # quat_rotate 需要的输入形状是 (K, 4) 和 (K, 3)
        # 我们的 local_coms 是 (self.num_envs, self.num_bodies, 3)，需要 reshape
        world_com_offset = quat_rotate(
            body_rot.view(-1, 4), self.local_coms_tensor.view(-1, 3)
        )

        # 计算每个刚体在世界坐标系下的质心
        # body_pos 是 (self.num_envs, self.num_bodies, 3)，将其和 offset 相加
        world_body_coms = body_pos + world_com_offset.view(
            self.num_envs, self.num_bodies, 3
        )

        # 计算总质量和加权的质心位置
        # body_masses 形状是 (self.num_envs, self.num_bodies)，需要扩展维度以进行乘法
        total_mass = torch.sum(self.masses_tensor, dim=1)  # (self.num_envs)

        # (self.num_envs, self.num_bodies, 3) * (self.num_envs, self.num_bodies, 1) -> (self.num_envs, self.num_bodies, 3)
        weighted_coms = world_body_coms * self.masses_tensor.unsqueeze(-1)

        # 沿着刚体维度求和，得到每个环境的总加权质心
        total_weighted_com = torch.sum(
            weighted_coms, dim=1
        )  # (self.num_envs, 3)

        # 计算最终的机器人质心，并防止除以零
        robot_com_W = total_weighted_com / (total_mass.unsqueeze(-1) + 1e-8)

        vector_w = robot_com_W - self.base_position

        self.q_WH = get_heading_quaternion(self.base_quat)

        robot_com_H = quat_rotate_inverse(self.q_WH, vector_w)

        return robot_com_H

    def _get_heights(self, env_ids=None):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs,
                self.num_height_points,
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError(
                "Can't measure height with terrain mesh type 'none'"
            )

        if env_ids:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points),
                self.height_points[env_ids],
            ) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(
                self.base_quat.repeat(1, self.num_height_points),
                self.height_points,
            ) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _process_rigid_shape_props(self, props, env_id):
        """Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(
                    friction_range[0],
                    friction_range[1],
                    (num_buckets, 1),
                    device="cpu",
                )
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        if self.cfg.domain_rand.randomize_restitution:
            if env_id == 0:
                (
                    min_restitution,
                    max_restitution,
                ) = self.cfg.domain_rand.restitution_range
                self.restitution_coef = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (max_restitution - min_restitution)
                    + min_restitution
                )
            for s in range(len(props)):
                props[s].restitution = self.restitution_coef[env_id]
        return props

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
            self.dof_vel_limits = torch.zeros(
                self.num_dof,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.torque_limits = torch.zeros(
                self.num_dof,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = (
                    m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                )
                self.dof_pos_limits[i, 1] = (
                    m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                )
        return props

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            if env_id == 0:
                min_add_mass, max_add_mass = (
                    self.cfg.domain_rand.added_mass_range
                )
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
                self.base_mass = props[0].mass + self.base_add_mass
            props[0].mass += self.base_add_mass[env_id]
        else:
            self.base_mass[:] = props[0].mass
        if self.cfg.domain_rand.randomize_base_com:
            if env_id == 0:
                com_x, com_y, com_z = self.cfg.domain_rand.rand_com_vec
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
            props[0].com.x += self.base_com[env_id, 0]
            props[0].com.y += self.base_com[env_id, 1]
            props[0].com.z += self.base_com[env_id, 2]
        if self.cfg.domain_rand.randomize_inertia:
            for i in range(len(props)):
                low_bound, high_bound = (
                    self.cfg.domain_rand.randomize_inertia_range
                )
                inertia_scale = np.random.uniform(low_bound, high_bound)
                props[i].mass *= inertia_scale
                props[i].inertia.x.x *= inertia_scale
                props[i].inertia.y.y *= inertia_scale
                props[i].inertia.z.z *= inertia_scale
        return props

    def _step_contact_targets(self):
        frequencies = self.gaits[:, 0]
        offsets = self.gaits[:, 1]
        durations = torch.cat(
            [
                self.gaits[:, 2].view(self.num_envs, 1),
                self.gaits[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )
        # 1. 更新步态时钟 (Gait Clock)
        self.gait_indices = torch.remainder(
            self.gait_indices + self.dt * frequencies, 1.0
        )  # 0~1之间循环

        # 2. 生成时钟信号 (Clock Signals)
        self.clock_inputs_sin = torch.sin(2 * np.pi * self.gait_indices)
        self.clock_inputs_cos = torch.cos(2 * np.pi * self.gait_indices)
        # self.doubletime_clock_inputs_sin = torch.sin(4 * np.pi * foot_indices)
        # self.halftime_clock_inputs_sin = torch.sin(np.pi * foot_indices)

        # von mises distribution
        kappa = self.cfg.rewards.kappa_gait_probs
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf
        # 3. 计算每只脚各自的相位 (Per-Foot Phase)
        foot_indices = torch.remainder(
            torch.cat(
                [
                    self.gait_indices.view(self.num_envs, 1),
                    (self.gait_indices + offsets + 1).view(self.num_envs, 1),
                ],
                dim=1,
            ),
            1.0,
        )

        # 4. 重新映射相位并生成平滑的接触信号
        stance_idxs = foot_indices < durations  # 判断脚是否应处于支撑相
        swing_idxs = foot_indices > durations  # 判断脚是否应处于摆动相

        # 将支撑相的时间段 [0, duration] 映射到 [0, 0.5]
        foot_indices[stance_idxs] = torch.remainder(
            foot_indices[stance_idxs], 1
        ) * (0.5 / durations[stance_idxs])
        # 将摆动相的时间段 [duration, 1.0] 映射到 [0.5, 1.0]
        foot_indices[swing_idxs] = 0.5 + (
            torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]
        ) * (0.5 / (1 - durations[swing_idxs]))

        self.desired_contact_states = smoothing_cdf_start(foot_indices) * (
            1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (
            1 - smoothing_cdf_start(foot_indices - 1.5)
        )
        # 图像 img/self.desired_contact_states.jpg   kappa<0.1才有明晰的支撑和摆动相

    def _get_foot_heights(self):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs,
                len(self.feet_indices),
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError(
                "Can't measure height with terrain mesh type 'none'"
            )

        points = self.foot_positions[:, :, :2] + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        heights = (
            heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        )

        return heights

    def _post_physics_step_callback(self):
        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)
        self._step_contact_targets()

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = 1.0 * wrap_to_pi(
                self.commands[:, 3] - heading
            )

        if (
            self.cfg.terrain.measure_heights
            or self.cfg.terrain.critic_measure_heights
        ):
            self.measured_heights = self._get_heights()

        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(
            self.cfg.domain_rand.push_interval_s / self.dt
        )
        self.gaits_ranges = class_to_dict(self.cfg.gait.ranges)

    def _resample_gaits(self, env_ids):
        if len(env_ids) == 0:
            return
        self.gaits[env_ids, 0] = torch_rand_float(
            self.gaits_ranges["frequencies"][0],
            self.gaits_ranges["frequencies"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.gaits[env_ids, 1] = torch_rand_float(
            self.gaits_ranges["offsets"][0],
            self.gaits_ranges["offsets"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        # parts = 4
        # self.gaits[env_ids, 1] = (self.gaits[env_ids, 1] * parts).round() / parts
        self.gaits[env_ids, 1] = 0.5

        self.gaits[env_ids, 2] = torch_rand_float(
            self.gaits_ranges["durations"][0],
            self.gaits_ranges["durations"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        # parts = 2
        # self.gaits[env_ids, 2] = (self.gaits[env_ids, 2] * parts).round() / parts

        self.gaits[env_ids, 3] = torch_rand_float(
            self.gaits_ranges["swing_height"][0],
            self.gaits_ranges["swing_height"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

    def _create_ground_plane(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(
            self.sim, self.terrain.heightsamples, hf_params
        )
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

    def _create_trimesh(self):
        """Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        #"""
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order="C"),
            self.terrain.triangles.flatten(order="C"),
            tm_params,
        )
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

    def _init_height_points(self):
        """Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(
            self.cfg.terrain.measured_points_y,
            device=self.device,
            requires_grad=False,
        )
        x = torch.tensor(
            self.cfg.terrain.measured_points_x,
            device=self.device,
            requires_grad=False,
        )
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs,
            self.num_height_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def pre_physics_step(self):
        self.rwd_linVelTrackPrev = self._reward_tracking_lin_vel()
        self.rwd_angVelTrackPrev = self._reward_tracking_ang_vel()
        if "tracking_contacts_shaped_height" in self.reward_scales.keys():
            self.rwd_swingHeightPrev = (
                self._reward_tracking_contacts_shaped_height()
            )

    def update_command_curriculum(self, env_ids):
        """Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if self.cfg.terrain.curriculum and len(self.success_ids) != 0:
            if "tracking_lin_vel" in self.episode_sums.keys():
                mask = (
                    self.episode_sums["tracking_lin_vel"][self.success_ids]
                    / self.max_episode_length
                    > self.cfg.commands.curriculum_threshold
                    * self.reward_scales["tracking_lin_vel"]
                )
            elif "tracking_lin_vel_x" in self.episode_sums.keys():
                mask = (
                    self.episode_sums["tracking_lin_vel_x"][self.success_ids]
                    / self.max_episode_length
                    > self.cfg.commands.curriculum_threshold
                    * self.reward_scales["tracking_lin_vel_x"]
                )
            success_ids = self.success_ids[mask]
            slope_ids = torch.any(
                success_ids.unsqueeze(1) == self.smooth_slope_idx.unsqueeze(0),
                dim=1,
            )
            slope_ids = success_ids[slope_ids]
            self.command_ranges["lin_vel_x"][success_ids, 0] -= 0.05
            self.command_ranges["lin_vel_x"][success_ids, 1] += 0.05
            self.command_ranges["lin_vel_x"][slope_ids, 0] -= 0.02
            self.command_ranges["lin_vel_x"][slope_ids, 1] += 0.02
            self.command_ranges["lin_vel_y"][success_ids, 0] -= 0.05
            self.command_ranges["lin_vel_y"][success_ids, 1] += 0.05
            self.command_ranges["lin_vel_y"][slope_ids, 0] -= 0.02
            self.command_ranges["lin_vel_y"][slope_ids, 1] += 0.02

            self.command_ranges["lin_vel_x"][self.smooth_slope_idx, :] = (
                torch.clip(
                    self.command_ranges["lin_vel_x"][self.smooth_slope_idx, :],
                    -self.cfg.commands.smooth_max_lin_vel_x,
                    self.cfg.commands.smooth_max_lin_vel_x,
                )
            )
            self.command_ranges["lin_vel_y"][self.smooth_slope_idx, :] = (
                torch.clip(
                    self.command_ranges["lin_vel_y"][self.smooth_slope_idx, :],
                    -self.cfg.commands.smooth_max_lin_vel_y,
                    self.cfg.commands.smooth_max_lin_vel_y,
                )
            )
            self.command_ranges["lin_vel_x"][self.none_smooth_idx, :] = (
                torch.clip(
                    self.command_ranges["lin_vel_x"][self.none_smooth_idx, :],
                    -self.cfg.commands.non_smooth_max_lin_vel_x,
                    self.cfg.commands.non_smooth_max_lin_vel_x,
                )
            )
            self.command_ranges["lin_vel_y"][self.none_smooth_idx, :] = (
                torch.clip(
                    self.command_ranges["lin_vel_y"][self.none_smooth_idx, :],
                    -self.cfg.commands.non_smooth_max_lin_vel_y,
                    self.cfg.commands.non_smooth_max_lin_vel_y,
                )
            )

    def _update_terrain_curriculum(self, env_ids):
        """Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1
        )
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        if "tracking_lin_vel" in self.episode_sums.keys():
            move_down = (
                self.episode_sums["tracking_lin_vel"][env_ids]
                / self.max_episode_length_s
                < (self.reward_scales["tracking_lin_vel"] / self.dt) * 0.5
            ) * ~move_up
            self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # for bipedal
        elif "tracking_lin_vel_x" in self.episode_sums.keys():
            move_down = (
                self.episode_sums["tracking_lin_vel_x"][env_ids]
                / self.max_episode_length_s
                < (self.reward_scales["tracking_lin_vel_x"] / self.dt) * 0.5
            ) * ~move_up
            self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        mask = self.terrain_levels[env_ids] >= self.max_terrain_level
        self.success_ids = env_ids[mask]
        mask = self.terrain_levels[env_ids] < 0
        self.fail_ids = env_ids[mask]
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(
                self.terrain_levels[env_ids], self.max_terrain_level
            ),
            torch.clip(self.terrain_levels[env_ids], 0),
        )  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]
        if self.cfg.commands.curriculum:
            self.command_ranges["lin_vel_x"][self.fail_ids, 0] = torch.clip(
                self.command_ranges["lin_vel_x"][self.fail_ids, 0] + 0.05,
                -self.cfg.commands.smooth_max_lin_vel_x,
                -0.0,
            )
            self.command_ranges["lin_vel_x"][self.fail_ids, 1] = torch.clip(
                self.command_ranges["lin_vel_x"][self.fail_ids, 1] - 0.05,
                0.0,
                self.cfg.commands.smooth_max_lin_vel_x,
            )
            self.command_ranges["lin_vel_y"][self.fail_ids, 0] = torch.clip(
                self.command_ranges["lin_vel_y"][self.fail_ids, 0] + 0.02,
                -self.cfg.commands.smooth_max_lin_vel_y,
                -0.0,
            )
            self.command_ranges["lin_vel_y"][self.fail_ids, 1] = torch.clip(
                self.command_ranges["lin_vel_y"][self.fail_ids, 1] - 0.02,
                0.0,
                self.cfg.commands.smooth_max_lin_vel_y,
            )

    def _push_robots(self):
        """Random pushes the robots."""
        env_ids = (
            (
                self.envs_steps_buf
                % int(self.cfg.domain_rand.push_interval_s / self.sim_params.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(env_ids) == 0:
            return

        max_push_force = (
            self.base_mass.mean().item()
            * self.cfg.domain_rand.max_push_vel_xy
            / self.sim_params.dt
        )
        max_push_torque = (
            self.base_mass.mean().item()
            * self.cfg.domain_rand.max_push_ang_vel
            / self.sim_params.dt
        )

        self.rigid_body_external_forces[:] = 0
        self.rigid_body_external_torques[:] = 0

        rigid_body_external_forces = torch_rand_float(
            -max_push_force,
            max_push_force,
            (self.num_envs, 3),
            device=self.device,
        )
        rigid_body_external_torques = torch_rand_float(
            -max_push_torque,
            max_push_torque,
            (self.num_envs, 3),
            device=self.device,
        )

        self.rigid_body_external_forces[env_ids, 0, :] = quat_rotate(
            self.base_quat[env_ids], rigid_body_external_forces[env_ids]
        )
        self.rigid_body_external_torques[env_ids, 0, :] = quat_rotate(
            self.base_quat[env_ids], rigid_body_external_torques[env_ids]
        )

        self.rigid_body_external_forces[env_ids, 0, 2] *= 0.5

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.rigid_body_external_forces),
            gymtorch.unwrap_tensor(self.rigid_body_external_torques),
            gymapi.ENV_SPACE,
        )

    def compute_observations(self):
        """Computes observations"""
        self.obs_buf, self.critic_obs_buf = self.compute_group_observations()

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

        self.obs_history = torch.cat(
            (self.obs_history[:, self.num_obs :], self.obs_buf), dim=-1
        )
        # add imu noise if needed
        if self.cfg.domain_rand.randomize_imu_offset:
            randomized_base_quat = quat_mul(
                self.random_imu_offset, self.base_quat
            )
            self.obs_buf[:, :3] = (
                quat_rotate_inverse(
                    randomized_base_quat, self.root_states[:, 10:13]
                )
                * self.obs_scales.ang_vel
            )
            self.obs_buf[:, 3:6] = quat_rotate_inverse(
                randomized_base_quat, self.gravity_vec
            )

    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -0.7, 0.7, (len(env_ids), 2), device=self.device
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

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids, :]
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def compute_foot_state(self):
        self.feet_state = self.rigid_body_state[:, self.feet_indices, :]
        self.foot_quat = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 3:7]
        self.foot_positions = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 0:3]
        self.foot_velocities = (
            self.foot_positions - self.last_foot_positions
        ) / self.dt

        self.foot_ang_vel = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 10:13]
        for i in range(len(self.feet_indices)):
            self.foot_ang_vel[:, i] = quat_rotate_inverse(
                self.foot_quat[:, i], self.foot_ang_vel[:, i]
            )
            self.foot_velocities_f[:, i] = quat_rotate_inverse(
                self.foot_quat[:, i], self.foot_velocities[:, i]
            )

        foot_relative_velocities = (
            self.foot_velocities
            - (self.base_position - self.last_base_position)
            .unsqueeze(1)
            .repeat(1, len(self.feet_indices), 1)
            / self.dt
        )
        for i in range(len(self.feet_indices)):
            self.foot_relative_velocities[:, i, :] = quat_rotate_inverse(
                self.base_quat, foot_relative_velocities[:, i, :]
            )
        self.foot_heights = torch.clip(
            (
                self.foot_positions[:, :, 2]
                - self.cfg.asset.foot_radius
                - self._get_foot_heights()
            ),
            0,
            1,
        )

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1  # 整体的训练时间

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_position = self.root_states[:, :3]
        self.base_lin_vel = (
            self.base_position - self.last_base_position
        ) / self.dt
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        self.power = torch.abs(self.torques * self.dof_vel)

        # 获取基座的欧拉角
        self.base_euler = get_euler_tensor(self.base_quat)

        self.compute_foot_state()

        self.contact_filt = self.get_foot_contacts()

        self.robot_com = self.get_robot_com_pos_heading()

        self.com_vec = self.get_com_dir_vec()

        self.base_lin_vel_H = quat_rotate_inverse(self.q_WH, self.base_lin_vel)
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.base_lin_vel
        )

        alpha = self.dt * self.gaits[:, 0].unsqueeze(1)
        self.avg_base_lin_vel = alpha * self.base_lin_vel + (1 - alpha) * self.avg_base_lin_vel
        self.avg_base_ang_vel = alpha * self.base_ang_vel + (1 - alpha) * self.avg_base_ang_vel

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:, :, 1] = self.last_actions[:, :, 0]
        self.last_actions[:, :, 0] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_base_position[:] = self.base_position[:]
        self.last_foot_positions[:] = self.foot_positions[:]
        self.last_contact_filt = self.contact_filt.clone()

    def _action_clip(self, actions):
        target_pos = torch.clip(
            actions * self.cfg.control.action_scale,
            self.dof_pos
            - self.default_dof_pos
            + (
                self.d_gains.mean() * self.dof_vel
                - self.cfg.control.user_torque_limit
            )
            / self.p_gains.mean(),
            self.dof_pos
            - self.default_dof_pos
            + (
                self.d_gains.mean() * self.dof_vel
                + self.cfg.control.user_torque_limit
            )
            / self.p_gains.mean(),
        )
        self.actions = target_pos / self.cfg.control.action_scale

    def compute_dof_vel(self):
        diff = (
            torch.remainder(
                self.dof_pos - self.last_dof_pos + self.pi, 2 * self.pi
            )
            - self.pi
        )
        self.dof_pos_dot = diff / self.sim_params.dt

        if self.cfg.env.dof_vel_use_pos_diff:
            self.dof_vel = self.dof_pos_dot

        self.last_dof_pos[:] = self.dof_pos[:]

    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[
            ..., 0
        ]  # equal [:,:, 0]
        print("self.dof_pos.size: ",self.dof_pos.size())
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[
            ..., 1
        ]
        self.dof_acc = torch.zeros_like(self.dof_vel)
        self.base_quat = self.root_states[:, 3:7]

        self.base_euler = get_euler_tensor(self.base_quat)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, -1
        )
        self.feet_state = self.rigid_body_state[:, self.feet_indices, :]
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, -1
        )
        self.foot_positions = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 0:3]
        self.last_foot_positions = torch.zeros_like(self.foot_positions)
        self.foot_heights = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.foot_velocities = torch.zeros_like(self.foot_positions)
        self.foot_velocities_f = torch.zeros_like(self.foot_positions)
        self.foot_relative_velocities = torch.zeros_like(self.foot_velocities)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.power = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.torques = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.torques_scale = torch.ones(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.d_gains = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.action_delay_idx = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        delay_max = np.int64(
            np.round(
                self.cfg.domain_rand.delay_ms_range[1]
                / 1000
                / self.sim_params.dt
            )
            + 1
        )
        self.action_fifo = torch.zeros(
            (self.num_envs, delay_max, self.cfg.env.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        if self.cfg.commands.heading_command:
            self.commands = torch.zeros(
                self.num_envs,
                self.cfg.commands.num_commands + 1,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )  # x vel, y vel, yaw vel, heading
        else:
            self.commands = torch.zeros(
                self.num_envs,
                self.cfg.commands.num_commands,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
            ],
            device=self.device,
            requires_grad=False,
        )  # TODO change this
        self.command_ranges["lin_vel_x"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["lin_vel_x"][:] = torch.tensor(
            self.cfg.commands.ranges.lin_vel_x
        )
        self.command_ranges["lin_vel_y"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["lin_vel_y"][:] = torch.tensor(
            self.cfg.commands.ranges.lin_vel_y
        )
        self.command_ranges["ang_vel_yaw"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["ang_vel_yaw"][:] = torch.tensor(
            self.cfg.commands.ranges.ang_vel_yaw
        )
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.contact_filt = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.last_contact_filt = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.base_position = self.root_states[:, :3]
        self.last_base_position = self.base_position.clone()
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_lin_vel_H = self.base_lin_vel.clone()
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.avg_base_lin_vel = self.base_lin_vel.clone()
        self.avg_base_ang_vel = self.base_ang_vel.clone()
        self.rigid_body_external_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3),
            device=self.device,
            requires_grad=False,
        )
        self.rigid_body_external_torques = torch.zeros(
            (self.num_envs, self.num_bodies, 3),
            device=self.device,
            requires_grad=False,
        )
        self.projected_gravity = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self.base_height = torch.zeros_like(self.root_states[:, 2])
        if (
            self.cfg.terrain.measure_heights
            or self.cfg.terrain.critic_measure_heights
        ):
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.raw_default_dof_pos = torch.zeros(
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.default_dof_pos = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_dofs):
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
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero"
                    )
        if self.cfg.domain_rand.randomize_Kp:
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
        if self.cfg.domain_rand.randomize_Kd:
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
        if self.cfg.domain_rand.randomize_motor_torque:
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
        if self.cfg.domain_rand.randomize_default_dof_pos:
            self.default_dof_pos += torch_rand_float(
                self.cfg.domain_rand.randomize_default_dof_pos_range[0],
                self.cfg.domain_rand.randomize_default_dof_pos_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            )

        if self.cfg.domain_rand.randomize_imu_offset:
            min_angle, max_angle = (
                self.cfg.domain_rand.randomize_imu_offset_range
            )

            min_angle_rad = math.radians(min_angle)
            max_angle_rad = math.radians(max_angle)

            pitch = (
                torch.rand(self.num_envs, device=self.device)
                * (max_angle_rad - min_angle_rad)
                + min_angle_rad
            )
            roll = (
                torch.rand(self.num_envs, device=self.device)
                * (max_angle_rad - min_angle_rad)
                + min_angle_rad
            )

            pitch_quat = torch.stack(
                [
                    torch.zeros_like(pitch),
                    torch.sin(pitch / 2),
                    torch.zeros_like(pitch),
                    torch.cos(pitch / 2),
                ],
                dim=-1,
            )
            roll_quat = torch.stack(
                [
                    torch.sin(roll / 2),
                    torch.zeros_like(roll),
                    torch.zeros_like(roll),
                    torch.cos(roll / 2),
                ],
                dim=-1,
            )

            self.random_imu_offset = quat_mul(pitch_quat, roll_quat)
        else:
            self.random_imu_offset = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], device=self.device
            ).repeat(self.num_envs, 1)

        self.gaits = torch.zeros(
            self.num_envs,
            self.cfg.gait.num_gait_params,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.desired_contact_states = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.gait_indices = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.clock_inputs_sin = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.clock_inputs_cos = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.doubletime_clock_inputs_sin = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.halftime_clock_inputs_sin = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.robot_com = torch.zeros(
            self.num_envs,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.com_vec = torch.zeros(
            self.num_envs,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

    def _get_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(
                0, max_init_level + 1, (self.num_envs,), device=self.device
            )
            self.terrain_types = torch.zeros(
                self.num_envs,
                dtype=torch.long,
                device=self.device,
                requires_grad=False,
            )
            self.terrain_types[-self.num_envs :] = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.cfg.terrain.num_cols),
                rounding_mode="floor",
            ).to(torch.long)

            self.smooth_slope_idx = (
                (self.terrain_types < 2).nonzero(as_tuple=False).flatten()
            )
            self.rough_slope_idx = (
                ((2 <= self.terrain_types) * (self.terrain_types < 4))
                .nonzero(as_tuple=False)
                .flatten()
            )
            self.stair_up_idx = (
                ((4 <= self.terrain_types) * (self.terrain_types < 11))
                .nonzero(as_tuple=False)
                .flatten()
            )
            self.stair_down_idx = (
                ((11 <= self.terrain_types) * (self.terrain_types < 16))
                .nonzero(as_tuple=False)
                .flatten()
            )
            self.discrete_idx = (
                ((16 <= self.terrain_types) * (self.terrain_types < 20))
                .nonzero(as_tuple=False)
                .flatten()
            )
            self.none_smooth_idx = torch.cat(
                (
                    self.rough_slope_idx,
                    self.stair_up_idx,
                    self.stair_down_idx,
                    self.discrete_idx,
                )
            )
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = (
                torch.from_numpy(self.terrain.env_origins)
                .to(self.device)
                .to(torch.float)
            )
            self.env_origins[:] = self.terrain_origins[
                self.terrain_levels, self.terrain_types
            ]
            self.terrain_x_max = (
                self.cfg.terrain.num_rows * self.cfg.terrain.terrain_length
                + self.cfg.terrain.border_size
            )
            self.terrain_x_min = -self.cfg.terrain.border_size
            self.terrain_y_max = (
                self.cfg.terrain.num_cols * self.cfg.terrain.terrain_length
                + self.cfg.terrain.border_size
            )
            self.terrain_y_min = -self.cfg.terrain.border_size
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(
                torch.arange(num_rows), torch.arange(num_cols)
            )
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0

    # --------------------------- reward functions---------------------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        # return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        return torch.norm(self.base_ang_vel[:, :2], dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        # reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        # return reward
        quat_mismatch = torch.exp(
            -torch.sum(torch.abs(self.base_euler[:, :2]), dim=1)
            * self.cfg.rewards.quat_sigma
        )
        orientation = torch.exp(
            -torch.norm(self.projected_gravity[:, :2], dim=1)
            * self.cfg.rewards.orientation_sigma
        )
        # return (
        #     quat_mismatch + quat_mismatch**5 + orientation + orientation**5
        # ) / 4
        return (orientation + orientation**5
        ) / 2

    def _reward_base_height(self):
        # Penalize base height away from target
        # base_height = torch.mean(
        #     self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        # )
        # return torch.square(
        #     base_height
        #     - self.cfg.rewards.base_height_target
        #     - self.cfg.asset.foot_radius
        # )
        rew = 1 - torch.exp(
            -torch.abs(
                self.base_height
                - self.cfg.rewards.base_height_target
                - self.cfg.asset.foot_radius
            )
            / self.cfg.rewards.base_height_sigma
        )
        return rew

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(self.actions - self.last_actions[:, :, 0]), dim=1
        )

    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions
                - 2 * self.last_actions[:, :, 0]
                + self.last_actions[:, :, 1]
            ),
            dim=1,
        )

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(
            min=0.0
        )
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]),
            dim=1,
        )
        # return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        ans = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        return (ans + ans**5) / 2
    
    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        ans = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        return (ans + ans**5) / 2
    
    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axes)
        lin_vel_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        
        ans = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        return (ans + ans**5) / 2

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2]
        )
        # return torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)
        ans = torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)
        return (ans + ans**5) / 2

    def _reward_tracking_contacts_shaped_force(self):
        # 计算每只脚的接触力大小（L2范数），形状为 (num_envs, num_feet)
        foot_forces = torch.norm(
            self.contact_forces[:, self.feet_indices, :], dim=-1
        )
        # 获取期望的接触状态，值在 [0, 1] 之间，接近 1 表示 stance，接近 0 表示 swing
        desired_contact = self.desired_contact_states

        # 初始化奖励值为 0
        reward = 0
        # 根据奖励权重判断使用正向奖励还是负向惩罚
        if self.reward_scales["tracking_contacts_shaped_force"] > 0:
            # 正向奖励：鼓励 swing 阶段（desired_contact 接近 0）接触力接近 0
            for i in range(len(self.feet_indices)):
                # 使用高斯函数对接触力进行平滑惩罚，gait_force_sigma 控制平滑程度
                # (1 - desired_contact) 确保奖励针对 swing 阶段
                reward += (1 - desired_contact[:, i]) * torch.exp(
                    -foot_forces[:, i] / self.cfg.rewards.gait_force_sigma
                )
        else:
            # 负向惩罚：惩罚 swing 阶段非零接触力
            for i in range(len(self.feet_indices)):
                # (1 - desired_contact) * (1 - exp(-force^2 / sigma)) 使非零接触力导致奖励接近 0
                reward += (1 - desired_contact[:, i]) * (
                    1
                    - torch.exp(
                        -foot_forces[:, i] / self.cfg.rewards.gait_force_sigma
                    )
                )

        # 返回平均奖励，除以脚的数量以归一化
        return reward / len(self.feet_indices)

    def _reward_tracking_contacts_shaped_vel(self):
        # 计算每只脚的速度大小（L2范数），形状为 (num_envs, num_feet)
        foot_velocities = torch.norm(self.foot_velocities, dim=-1)
        # 获取期望的接触状态，值在 [0, 1] 之间，接近 1 表示 stance，接近 0 表示 swing
        desired_contact = self.desired_contact_states
        # 初始化奖励值为 0
        reward = 0
        # 根据奖励权重判断使用正向奖励还是负向惩罚
        if self.reward_scales["tracking_contacts_shaped_vel"] > 0:
            # 正向奖励：鼓励 stance 阶段（desired_contact 接近 1）速度接近 0
            for i in range(len(self.feet_indices)):
                # 使用高斯函数对速度进行平滑惩罚，gait_vel_sigma 控制平滑程度
                # desired_contact 确保奖励针对 stance 阶段
                reward += desired_contact[:, i] * torch.exp(
                    -foot_velocities[:, i] / self.cfg.rewards.gait_vel_sigma
                )
        else:
            # 负向惩罚：惩罚 stance 阶段非零速度
            for i in range(len(self.feet_indices)):
                # desired_contact * (1 - exp(-velocity^2 / sigma)) 使非零速度导致奖励接近 0
                reward += desired_contact[:, i] * (
                    1
                    - torch.exp(
                        -foot_velocities[:, i] / self.cfg.rewards.gait_vel_sigma
                    )
                )
        # 返回平均奖励，除以脚的数量以归一化
        return reward / len(self.feet_indices)

    def _reward_tracking_contacts_shaped_height(self):
        # 计算每只脚与目标高度的差距，形状为 (num_envs, num_feet)
        foot_height_errors = torch.abs(
            self.foot_heights - self.gaits[:, 3].view(-1, 1)
        )

        # 获取期望的接触状态，值在 [0, 1] 之间，接近 1 表示 stance，接近 0 表示 swing
        desired_contact = self.desired_contact_states

        # 初始化奖励值为 0
        reward = 0

        # 根据奖励权重判断使用正向奖励还是负向惩罚
        if self.reward_scales["tracking_contacts_shaped_height"] > 0:
            # 正向奖励：鼓励 swing 阶段（desired_contact 接近 0）足部高度接近目标高度
            for i in range(len(self.feet_indices)):
                # 使用高斯函数对高度误差进行平滑奖励，gait_height_sigma 控制平滑程度
                # (1 - desired_contact) 确保奖励针对 swing 阶段（抬脚时）
                reward += (1 - desired_contact[:, i]) * torch.exp(
                    -foot_height_errors[:, i]
                    / self.cfg.rewards.gait_height_sigma
                )
        else:
            # 负向惩罚：惩罚 swing 阶段高度偏离目标的情况
            for i in range(len(self.feet_indices)):
                # (1 - desired_contact) * (1 - exp(-error^2 / sigma)) 使高度偏离导致奖励接近 0
                reward += (1 - desired_contact[:, i]) * (
                    1
                    - torch.exp(
                        -foot_height_errors[:, i]
                        / self.cfg.rewards.gait_height_sigma
                    )
                )

        # 返回平均奖励，除以脚的数量以归一化
        return reward / len(self.feet_indices)

    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.norm(
            self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2],
            dim=-1,
        )
        # reward = torch.clip(
        #     self.cfg.rewards.min_feet_distance - feet_distance, 0, 1
        # )
        # return reward
        # 二值化惩罚：距离小于最小安全距离时惩罚为1，否则为0
        reward = torch.where(
            feet_distance < self.cfg.rewards.min_feet_distance,
            torch.ones_like(feet_distance),  # 距离不足时惩罚=1
            torch.zeros_like(feet_distance),  # 距离足够时惩罚=0
        )
        return reward

    def _reward_leg_symmetry(self):
        foot_positions_base = self.foot_positions - (
            self.base_position
        ).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(
                self.base_quat, foot_positions_base[:, i, :]
            )
        leg_symmetry_err = abs(
            abs(foot_positions_base[:, 0, 1])
            - abs(foot_positions_base[:, 1, 1])
        )
        return torch.exp(
            -leg_symmetry_err / self.cfg.rewards.leg_symmetry_tracking_sigma
        )

    def _reward_feet_regulation(self):
        feet_height = (self.cfg.rewards.base_height_target) * 0.001
        reward = torch.sum(
            torch.exp(-self.foot_heights / feet_height)
            * torch.square(torch.norm(self.foot_velocities[:, :, :2], dim=-1)),
            dim=1,
        )
        return reward

    def _reward_collision(self):
        return torch.sum(
            torch.norm(
                self.contact_forces[:, self.penalised_contact_indices, :],
                dim=-1,
            )
            > 1.0,
            dim=1,
        )

    def _reward_foot_landing_vel(self):
        # 获取所有足部的垂直速度（z方向）
        z_vels = self.foot_velocities[:, :, 2]

        # 识别"即将着地"的足部状态
        about_to_land = (
            # 条件1：足部高度低于着地阈值
            (self.foot_heights < self.cfg.rewards.about_landing_threshold)
            # 条件2：足部当前未接触地面
            & (~self.contact_filt)
            # 条件3：足部正在向下运动（垂直速度为负）
            & (z_vels < 0.0)
        )

        # 只对即将着地的足部提取垂直速度的正值，其他设为0
        landing_z_vels = torch.where(
            about_to_land, z_vels, torch.zeros_like(z_vels)
        )

        # 计算垂直着地速度的平方和作为惩罚
        # 速度越大（向下越快），惩罚越重
        # reward = torch.sum(torch.square(landing_z_vels), dim=1)
        # reward = torch.norm(landing_z_vels, dim=1)
        reward = 1.0 - torch.exp(
            -torch.norm(landing_z_vels, dim=1)
            / self.cfg.rewards.foot_landing_vel_sigma
        )

        return reward

    def _reward_foot_contact_force_impact(self):
        """
        强力惩罚落地瞬间冲击力过大（> threshold）
        """
        # 当前足端 z 向接触力（向上为正）
        foot_forces_z = self.contact_forces[
            :, self.feet_indices, 2
        ]  # (num_envs, num_feet)

        # 超过阈值的冲击力（超过部分才罚）
        impact_excess = torch.clamp(
            foot_forces_z
            - self.cfg.rewards.foot_contact_force_impact_threshold,
            min=0.0,
        )  # 只罚超出的部分

        in_contact = self.contact_filt
        impact_penalty = impact_excess * in_contact.float()

        # 归一化 + 取平均（每只脚等权）
        reward = torch.sum(impact_penalty) / len(self.feet_indices)
        return reward

    def _reward_hip_roll_smooth(self):
        return torch.sum(
            torch.square(
                self.actions[:,self.hip_roll_joint_indices]
                - 2 * self.last_actions[:, self.hip_roll_joint_indices, 0]
                + self.last_actions[:, self.hip_roll_joint_indices, 1]
            ),
            dim=1,
        )/len(self.hip_roll_joint_indices)

    def _reward_hip_roll_rate(self):
        return torch.sum(
            torch.square(
                self.actions[:,self.hip_roll_joint_indices]
                - self.last_actions[:, self.hip_roll_joint_indices, 0]
            ),
            dim=1,
        )/len(self.hip_roll_joint_indices)
    
    def _reward_hip_roll_deviation(self):
        return torch.sum(
            torch.abs(
                self.actions_scaled[:,self.hip_roll_joint_indices]
                # self.dof_pos[:,self.hip_roll_joint_indices] - self.default_dof_pos[:,self.hip_roll_joint_indices]
            )
        )/ len(self.hip_roll_joint_indices)