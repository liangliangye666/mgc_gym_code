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

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR, WHEEL_LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .wheel_legged.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO
from .wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC
from .wheel_legged_vmc.wheel_legged_vmc_config import (
    WheelLeggedVMCCfg,
    WheelLeggedVMCCfgPPO,
)
from .wheel_legged_vmc_flat.wheel_legged_vmc_flat_config import (
    WheelLeggedVMCFlatCfg,
    WheelLeggedVMCFlatCfgPPO,
)

from .y4a_2wheel.y4a_2wheel import Y4A_2WHEEL
from .y4a_2wheel.y4a_2wheel_config import (Y4A_2WHEEL_Cfg, Y4A_2WHEEL_CfgPPO,)

from .y4b_2wheel.y4b_2wheel import Y4B_2WHEEL
from .y4b_2wheel.y4b_2wheel_config import (
    Y4B_2WHEEL_Cfg,
    Y4B_2WHEEL_CfgPPO,
)

from .l5a_2wheel.l5a_2wheel import L5A_2WHEEL
from .l5a_2wheel.l5a_2wheel_config import (
    L5A_2WHEEL_Cfg,
    L5A_2WHEEL_CfgPPO,
)

import os

from wheel_legged_gym.utils.task_registry import task_registry

task_registry.register(
    "wheel_legged",
    LeggedRobot,
    WheelLeggedCfg(),
    WheelLeggedCfgPPO(),
)
task_registry.register(
    "wheel_legged_vmc",
    LeggedRobotVMC,
    WheelLeggedVMCCfg(),
    WheelLeggedVMCCfgPPO(),
)
task_registry.register(
    "wheel_legged_vmc_flat",
    LeggedRobotVMC,
    WheelLeggedVMCFlatCfg(),
    WheelLeggedVMCFlatCfgPPO(),
)

task_registry.register(
    "y4a_2wheel",
    Y4A_2WHEEL,
    Y4A_2WHEEL_Cfg(),
    Y4A_2WHEEL_CfgPPO(),
)

task_registry.register(
    "y4b_2wheel",
    Y4B_2WHEEL,
    Y4B_2WHEEL_Cfg(),
    Y4B_2WHEEL_CfgPPO(),
)

task_registry.register(
    "l5a_2wheel",
    L5A_2WHEEL,
    L5A_2WHEEL_Cfg(),
    L5A_2WHEEL_CfgPPO(),
)