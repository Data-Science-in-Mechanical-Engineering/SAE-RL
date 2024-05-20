from typing import Optional

import numpy as np
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.push import Push
from panda_gym.pybullet import PyBullet


class PushCustom(Push):
    """Custom Panda Push task with target position in blue and shown simply as rectangle on the table.
    """

    def _create_scene(self):

        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name='object',
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name='target',
            half_extents=np.array([1.0, 1.0, 0.01]) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.0]),
            rgba_color=np.array([0.1, 0.1, 0.9, 1.0]),
        )

    def reset(self):

        self.goal = self._sample_goal()
        goal_position = self.goal - np.array([0.0, 0.0, self.object_size / 2])  # shift so only top is visible
        object_position = self._sample_object()
        self.sim.set_base_pose('target', goal_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose('object', object_position, np.array([0.0, 0.0, 0.0, 1.0]))


class PandaPushCustomEnv(RobotTaskEnv):
    """Custom Push environment wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to 'rgb_array'.
        reward_type (str, optional): 'sparse' or 'dense'. Defaults to 'sparse'.
        control_type (str, optional): 'ee' to control end-effector position or 'joints' to control joint values. Defaults to 'ee'.
        renderer (str, optional): Renderer, either 'Tiny' or OpenGL'. Defaults to 'Tiny' if render mode is 'human' and 'OpenGL' if render mode is 'rgb_array'. Only 'OpenGL' is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z). Defaults to [0.0, 0.0, 0.0].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Rool of the camera. Defaults to 0.
    """

    def __init__(self, render_mode: str = 'rgb_array', reward_type: str = 'sparse', control_type: str = 'ee', renderer: str = 'Tiny', render_width: int = 720, render_height: int = 480, render_target_position: Optional[np.ndarray] = None, render_distance: float = 1.4, render_yaw: float = 45, render_pitch: float = -30, render_roll: float = 0):

        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PushCustom(sim, reward_type=reward_type)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )

    def seed(self, seed: int):
        """Reset environment with specified seed.

        Used for compatibility with `stable_baselines3.common.vec_env.SubprocVecEnv.seed`.

        Args:
            seed (int): Random seed.
        """

        self.reset(seed)
