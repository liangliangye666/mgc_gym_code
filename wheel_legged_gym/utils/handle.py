from dataclasses import dataclass
import time
import numpy as np

@dataclass
class JoystickStatus:
    command: np.ndarray
    motion_enable: bool
    gait_enable: bool
    rl_run: bool
    brake_translation: bool
    brake_yaw: bool

class JoystickCommandSource:
    AXIS_CENTER = 0.5
    DEAD_ZONE = 0.1
    BUTTON_THRESHOLD = 0.5

    # F310 D mode / Logitech Dual Action
    VEL_X_SCALE = -1.0
    VEL_Y_SCALE = -1.0
    OMEGA_SCALE = -1.0
    VEL_X_ACCEL = 0.5
    VEL_X_BRAKE = 1.5

    AXIS_LEFT_X = 0
    AXIS_LEFT_Y = 1
    AXIS_RIGHT_X = 2
    AXIS_RIGHT_Y = 3

    BUTTON_A = 0
    BUTTON_B = 1
    BUTTON_X = 2
    BUTTON_Y = 3
    BUTTON_LB = 4
    BUTTON_RB = 5
    BUTTON_LT = 6
    BUTTON_RT = 7
    BUTTON_BACK = 8
    BUTTON_START = 9

    def __init__(self, pygame, joystick_index: int, debug: bool):
        self.pygame = pygame
        self.debug = debug
        self.vel_x = 0.0
        self.status = JoystickStatus(
            command=np.zeros(3, dtype=np.float32),
            motion_enable=False,
            gait_enable=False,
            rl_run=False,
            brake_translation=False,
            brake_yaw=False,
        )
        self._next_debug_time = 0.0

        self.pygame.init()
        self.pygame.joystick.init()
        joystick_count = self.pygame.joystick.get_count()
        if joystick_count <= joystick_index:
            raise IndexError(
                f"pygame detected {joystick_count} joystick(s), "
                f"but --joystick-index={joystick_index}"
            )

        self.joystick = self.pygame.joystick.Joystick(joystick_index)
        self.joystick.init()
        self.name = self.joystick.get_name()
        self.index = joystick_index

    @staticmethod
    def clamp01(value: float) -> float:
        return min(1.0, max(0.0, value))

    @classmethod
    def axis_in_dead_zone(cls, value: float) -> bool:
        return abs(value - cls.AXIS_CENTER) <= cls.DEAD_ZONE

    @classmethod
    def pressed(cls, value: float) -> bool:
        return value >= cls.BUTTON_THRESHOLD

    def _raw_axis(self, axis_index: int, default: float = 0.0) -> float:
        if axis_index < 0 or axis_index >= self.joystick.get_numaxes():
            return default
        return float(self.joystick.get_axis(axis_index))

    def _stick_axis(self, axis_index: int) -> float:
        return self.clamp01(0.5 + 0.5 * self._raw_axis(axis_index))

    def _button(self, button_index: int) -> float:
        if button_index < 0 or button_index >= self.joystick.get_numbuttons():
            return 0.0
        return 1.0 if self.joystick.get_button(button_index) else 0.0

    def read_command(self, policy_dt: float) -> np.ndarray:
        self.pygame.event.pump()

        a0 = self._stick_axis(self.AXIS_LEFT_X)
        a1 = self._stick_axis(self.AXIS_LEFT_Y)
        a2 = self._stick_axis(self.AXIS_RIGHT_X)
        a3 = self._stick_axis(self.AXIS_RIGHT_Y)

        b0 = self._button(self.BUTTON_A)
        b1 = self._button(self.BUTTON_B)
        b2 = self._button(self.BUTTON_X)
        b3 = self._button(self.BUTTON_Y)
        b4 = self._button(self.BUTTON_LB)
        b5 = self._button(self.BUTTON_RB)
        b6 = self._button(self.BUTTON_LT)
        b7 = self._button(self.BUTTON_RT)
        b8 = self._button(self.BUTTON_BACK)
        b9 = self._button(self.BUTTON_START)

        # D模式没有LT/RT轴，这里改成按钮逻辑
        motion_enable = self.pressed(b6)
        gait_enable = self.pressed(b7)

        # 安全开关：按住 A+B 才允许策略运动
        rl_run = self.pressed(b0) and self.pressed(b1)

        brake_translation = self.pressed(b4)
        brake_yaw = self.pressed(b5)

        if self.axis_in_dead_zone(a1) or brake_translation:
            target_vel_x = 0.0
        else:
            target_vel_x = (a1 - self.AXIS_CENTER) * self.VEL_X_SCALE * motion_enable

        self.vel_x = target_vel_x
        # self.vel_x = self.update_with_smart_brake(
        #     self.vel_x,
        #     target_vel_x,
        #     self.VEL_X_ACCEL,
        #     self.VEL_X_BRAKE,
        #     policy_dt,
        # )

        if self.axis_in_dead_zone(a0) or brake_translation:
            vel_y = 0.0
        else:
            vel_y = (a0 - self.AXIS_CENTER) * self.VEL_Y_SCALE * motion_enable

        if self.axis_in_dead_zone(a2) or brake_yaw:
            omega = 0.0
        else:
            omega = (a2 - self.AXIS_CENTER) * self.OMEGA_SCALE * motion_enable

        command = np.array([self.vel_x, vel_y, omega], dtype=np.float32)
        self.status = JoystickStatus(
            command=command,
            motion_enable=motion_enable,
            gait_enable=gait_enable,
            rl_run=rl_run,
            brake_translation=brake_translation,
            brake_yaw=brake_yaw,
        )

        self._maybe_print_debug(a0, a1, a2, a3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9)
        return command

    def _maybe_print_debug(self, a0, a1, a2, a3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9):
        now = time.time()
        if not self.debug or now < self._next_debug_time:
            return

        self._next_debug_time = now + 0.2
        print(
            f"[Joystick {self.index}: {self.name}] "
            f"a0={a0:.2f}, a1={a1:.2f}, a2={a2:.2f}, a3={a3:.2f}, "
            f"b0={b0:.0f}, b1={b1:.0f}, b2={b2:.0f}, b3={b3:.0f}, "
            f"b4={b4:.0f}, b5={b5:.0f}, b6={b6:.0f}, b7={b7:.0f}, "
            f"b8={b8:.0f}, b9={b9:.0f}, "
            f"cmd={self.status.command}, "
            f"motion={self.status.motion_enable}, "
            f"gait={self.status.gait_enable}, "
            f"rl_run={self.status.rl_run}"
        )

    def update_with_smart_brake(self, current: float, target: float, accel: float, brake: float, dt: float) -> float:
        rate = brake if abs(target) < abs(current) else accel
        delta = np.clip(target - current, -rate * dt, rate * dt)
        return current + delta

# class JoystickCommandSource:
#     AXIS_CENTER = 0.5
#     DEAD_ZONE = 0.1
#     BUTTON_THRESHOLD = 0.5
#     VEL_X_SCALE = -2.0
#     VEL_Y_SCALE = -1.0
#     OMEGA_SCALE = -1.0
#     VEL_X_ACCEL = 0.5
#     VEL_X_BRAKE = 1.5

#     AXIS_LEFT_X = 0
#     AXIS_LEFT_Y = 1
#     AXIS_TRIGGER_LEFT = 2
#     AXIS_RIGHT_X = 3
#     AXIS_TRIGGER_RIGHT = 5

#     BUTTON_A = 0
#     BUTTON_B = 1
#     BUTTON_LB = 4
#     BUTTON_RB = 5

#     def __init__(self, pygame, joystick_index: int, debug: bool):
#         self.pygame = pygame
#         self.debug = debug
#         self.vel_x = 0.0
#         self.status = JoystickStatus(
#             command=np.zeros(3, dtype=np.float32),
#             motion_enable=False,
#             gait_enable=False,
#             rl_run=False,
#             brake_translation=False,
#             brake_yaw=False,
#         )
#         self._next_debug_time = 0.0
#         self._signed_trigger_axes = set()

#         self.pygame.init()
#         self.pygame.joystick.init()
#         joystick_count = self.pygame.joystick.get_count()
#         if joystick_count <= joystick_index:
#             raise IndexError(
#                 f"pygame detected {joystick_count} joystick(s), "
#                 f"but --joystick-index={joystick_index}"
#             )

#         self.joystick = self.pygame.joystick.Joystick(joystick_index)
#         self.joystick.init()
#         self.name = self.joystick.get_name()
#         self.index = joystick_index

#     @staticmethod
#     def clamp01(value: float) -> float:
#         return min(1.0, max(0.0, value))

#     @classmethod
#     def axis_in_dead_zone(cls, value: float) -> bool:
#         return abs(value - cls.AXIS_CENTER) <= cls.DEAD_ZONE

#     @classmethod
#     def pressed(cls, value: float) -> bool:
#         return value >= cls.BUTTON_THRESHOLD

#     def _raw_axis(self, axis_index: int, default: float = 0.0) -> float:
#         if axis_index < 0 or axis_index >= self.joystick.get_numaxes():
#             return default
#         return float(self.joystick.get_axis(axis_index))

#     def _stick_axis(self, axis_index: int) -> float:
#         return self.clamp01(0.5 + 0.5 * self._raw_axis(axis_index))

#     def _trigger_axis(self, axis_index: int) -> float:
#         raw = self._raw_axis(axis_index)
#         if raw < -0.05:
#             self._signed_trigger_axes.add(axis_index)
#         if axis_index in self._signed_trigger_axes:
#             return self.clamp01(0.5 + 0.5 * raw)
#         return self.clamp01(raw)

#     def _button(self, button_index: int) -> float:
#         if button_index < 0 or button_index >= self.joystick.get_numbuttons():
#             return 0.0
#         return 1.0 if self.joystick.get_button(button_index) else 0.0

#     def read_command(self, policy_dt: float) -> np.ndarray:
#         self.pygame.event.pump()

#         a0 = self._stick_axis(self.AXIS_LEFT_X)
#         a1 = self._stick_axis(self.AXIS_LEFT_Y)
#         a2 = self._trigger_axis(self.AXIS_TRIGGER_LEFT)
#         a3 = self._stick_axis(self.AXIS_RIGHT_X)
#         a5 = self._trigger_axis(self.AXIS_TRIGGER_RIGHT)
#         b0 = self._button(self.BUTTON_A)
#         b1 = self._button(self.BUTTON_B)
#         b4 = self._button(self.BUTTON_LB)
#         b5 = self._button(self.BUTTON_RB)

#         motion_enable = self.pressed(a2)
#         gait_enable = self.pressed(a5)
#         rl_run = self.pressed(b0) and self.pressed(b1)
#         brake_translation = self.pressed(b4)
#         brake_yaw = self.pressed(b5)

#         if self.axis_in_dead_zone(a1) or brake_translation:
#             target_vel_x = 0.0
#         else:
#             target_vel_x = (
#                 (a1 - self.AXIS_CENTER) * self.VEL_X_SCALE * motion_enable
#             )
#         self.vel_x = self.update_with_smart_brake(
#             self.vel_x,
#             target_vel_x,
#             self.VEL_X_ACCEL,
#             self.VEL_X_BRAKE,
#             policy_dt,
#         )

#         if self.axis_in_dead_zone(a0) or brake_translation:
#             vel_y = 0.0
#         else:
#             vel_y = (a0 - self.AXIS_CENTER) * self.VEL_Y_SCALE * motion_enable

#         if self.axis_in_dead_zone(a3) or brake_yaw:
#             omega = 0.0
#         else:
#             omega = (a3 - self.AXIS_CENTER) * self.OMEGA_SCALE * motion_enable

#         command = np.array([self.vel_x, vel_y, omega], dtype=np.float32)
#         self.status = JoystickStatus(
#             command=command,
#             motion_enable=motion_enable,
#             gait_enable=gait_enable,
#             rl_run=rl_run,
#             brake_translation=brake_translation,
#             brake_yaw=brake_yaw,
#         )
#         self._maybe_print_debug(a0, a1, a2, a3, a5, b0, b1, b4, b5)
#         return command
    

#     def _maybe_print_debug(self, a0, a1, a2, a3, a5, b0, b1, b4, b5):
#         now = time.time()
#         if not self.debug or now < self._next_debug_time:
#             return

#         self._next_debug_time = now + 0.2
#         print(
#             f"[Joystick {self.index}: {self.name}] "
#             f"a0={a0:.2f}, a1={a1:.2f}, a2={a2:.2f}, "
#             f"a3={a3:.2f}, a5={a5:.2f}, "
#             f"A={b0:.0f}, B={b1:.0f}, LB={b4:.0f}, RB={b5:.0f}, "
#             f"cmd={self.status.command}, "
#             f"motion={self.status.motion_enable}, "
#             f"gait={self.status.gait_enable}, "
#             f"rl_run={self.status.rl_run}"
#         )

#     def update_with_smart_brake(self, current: float, target: float, accel: float, brake: float, dt: float) -> float:
#         rate = brake if abs(target) < abs(current) else accel
#         delta = np.clip(target - current, -rate * dt, rate * dt)
#         return current + delta