import sys
from contextlib import nullcontext
from typing import Dict, List, Tuple

import gymnasium
import gymnasium as gym
import numpy as np
import pybullet as p
import torch
import torch.nn as nn
from gymnasium.core import ObsType

from ..utils import Bunch, gl
from .utils import antialias, compute_projection_matrix, compute_view_matrix, project_2D

sys.modules['gym'] = gymnasium  # see [PR](https://github.com/DLR-RM/stable-baselines3/pull/780)
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper


class NotInstantlySolvedWrapper(gym.Wrapper):
    """Wrapper ensuring that a newly reset environment does not randomly fulfill the requirements for a solved task.
    """

    def reset(self, **kwargs) -> Tuple[ObsType, Dict]:
        """Resets the environment potentially multiple times until it does not directly fulfill success requirements.

        Keyworded arguments are passed to the underlying environment's reset function.

        Returns:
            Tuple[ObsType, Dict]: Observation of the initial state and dictionary containing auxiliary information.
        """

        observation, info = self.env.reset(**kwargs)

        while self.task.is_success(observation['achieved_goal'], observation['desired_goal']):
            observation, info = self.env.reset(**kwargs)

        return observation, info


class CameraUtilsWrapper(gym.Wrapper):
    """Wrapper adding interfaces for camera-related functions to the wrapped env, like positioning the camera and retrieving images, feature points or keypoint.
    """

    def __init__(self, env: gym.Env, **kwargs):
        """Initializes camera utils wrapper.

        Note: Keyworded arguments are passed to the `setup_camera(...)` method.

        Args:
            env (gym.Env): Environment to wrap.
        """

        super().__init__(env)

        self.setup_camera(**kwargs)

    def setup_camera(self, height: int = 480, width: int = 720, target_position: List[float] = [0, 0, 0], distance: float = 1.4, yaw: float = 45, pitch: float = -30, roll: float = 0, antialias_factor: int = 2, **_):
        """Sets up camera positioning and recording parameters.

        Args:
            height (int, optional): Image height. Defaults to 480.
            width (int, optional): Image width. Defaults to 720.
            target_position (List[float], optional): Position targeted by the camera, as (x, y, z). Defaults to [0, 0, 0].
            distance (float, optional): Distance of the camera. Defaults to 1.4.
            yaw (float, optional): Yaw of the camera. Defaults to 45.
            pitch (float, optional): Pitch of the camera. Defaults to -30.
            roll (float, optional): Roll of the camera. Defaults to 0.
            antialias_factor (int, optional): Antialiasing factor for average-pooling kernel size. Defaults to 2.
        """

        self.rec = Bunch(
            height=(height * antialias_factor),
            width=(width * antialias_factor),
            aspect_ratio=(width / height),
            antialias_factor=antialias_factor
        )

        self.camera = Bunch(
            target_position=target_position,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll
        )

        self.view_matrix = compute_view_matrix(**self.camera)
        self.projection_matrix = compute_projection_matrix(self.rec.aspect_ratio)

    def get_image(self, channels_first: bool = True) -> torch.Tensor:
        """Renders an antialiased RGB image of the environment.

        Returns:
            torch.Tensor: Rendered image.
        """

        image = self.env.render()
        image = torch.tensor(image, device=gl.device) / 255
        image = antialias(image, factor=self.rec.antialias_factor)

        if channels_first:
            image = torch.movedim(image, -1, -3)

        return image

    @torch.no_grad()
    def get_feature_points(self, encoder: nn.Module) -> torch.Tensor:
        """Encodes an image of the environment via the given encoder to extract 2D feature points.

        Args:
            encoder (nn.Module): Encoder to use for feature point extraction.

        Returns:
            torch.Tensor: Extracted 2D feature points.
        """

        image = self.get_image()
        feature_points = encoder(image.unsqueeze(dim=0)).squeeze(dim=0)

        return feature_points

    def get_keypoints(self, objects: List[str], links: List[int]) -> torch.Tensor:
        """Extracts ground truth 2D keypoints of given environment objects and roboter links.

        Args:
            objects (List[str]): Objects to determine the base position keypoints of
            links (List[int]): Roboter links to determine the position keypoints of

        Returns:
            torch.Tensor: Extracted 2D keypoints.
        """

        keypoints = []

        # get object positions
        for object in objects:
            point_3D = torch.tensor(self.env.sim.get_base_position(object).astype(np.float32), device=gl.device)
            point_2D = project_2D(point_3D, self.view_matrix, self.projection_matrix)
            keypoints.append(point_2D)

        # get roboter link positions
        for link in links:
            point_3D = torch.tensor(self.env.sim.get_link_position('panda', link).astype(np.float32), device=gl.device)
            point_2D = project_2D(point_3D, self.view_matrix, self.projection_matrix)
            keypoints.append(point_2D)

        keypoints = torch.stack(keypoints)

        return keypoints


class MeasurableObservationWrapper(gym.ObservationWrapper):
    """Wrapper splitting observation into measurable robot state and immeasurable object states, including only the requested parts in the final observation.
    """

    def __init__(self, env: gym.Env, include_measurable: bool, include_immeasurable: bool = False):
        """Initializes measurable observation wrapper.

        Args:
            env (gym.Env): Environment to wrap.
            include_measurable (bool): Whether to include measurable state information (robot) in the observation.
            include_immeasurable (bool, optional): Whether to include immeasurable state information (objects) in the observation. Defaults to False.
        """

        super().__init__(env)

        self.include_measurable = include_measurable
        self.include_immeasurable = include_immeasurable

        obs_space = {}
        if self.include_measurable:
            obs_space['measurable'] = self.env.robot.get_obs().shape
        if self.include_immeasurable:
            obs_space['immeasurable'] = np.concatenate([self.env.task.get_obs(), self.env.task.get_goal()]).shape

        self.observation_space = gym.spaces.Dict({
            key: gym.spaces.Box(low=-10.0, high=10.0, shape=value) for key, value in obs_space.items()
        })

    def observation(self, obs: Dict) -> Dict:
        """Filters observation for measurable robot state and immeasurable object states.

        Args:
            obs (Dict): Original state observation, completely replaced here.

        Returns:
            Dict: Modified state observation.
        """

        obs = {}
        if self.include_measurable:
            obs['measurable'] = self.env.robot.get_obs()
        if self.include_immeasurable:
            obs['immeasurable'] = np.concatenate([self.env.task.get_obs(), self.env.task.get_goal()])

        return obs


class KeypointObservationWrapper(gym.ObservationWrapper):
    """Wrapper extending the state observation with keypoints extracted from a wrapped environment.
    """

    def __init__(self, env: gym.Env, objects: List[str], links: List[int], **kwargs):
        """Initializes keypoint observation wrapper.

        Note: Environment `env` is expected to be wrapped in a `CameraUtilsWrapper`. Otherwise, wrapping is performed automatically and keyworded arguments are passed to the `CameraUtilsWrapper`.

        Args:
            env (gym.Env): Environment to wrap.
            objects (List[str]): Objects to determine the base position keypoints of.
            links (List[int]): Roboter links to determine the position keypoints of.
        """

        # wrap with camera utils wrapper if needed
        if not isinstance(env, CameraUtilsWrapper):
            env = CameraUtilsWrapper(env, **kwargs)

        super().__init__(env)

        # set attributes
        self.objects = objects
        self.links = links

        # adjust observation space
        n_keypoints = len(objects) + len(links)
        keypoint_obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(n_keypoints, 2))
        self.observation_space = gym.spaces.Dict({
            **self.observation_space,
            'keypoints': keypoint_obs_space
        })

    def observation(self, obs: Dict) -> Dict:
        """Extends existing observations with extracted keypoints.

        Args:
            obs (Dict): Original state observation.

        Returns:
            Dict: Observations with keypoints.
        """

        keypoints = self.get_keypoints(self.objects, self.links).cpu().numpy()
        obs['keypoints'] = keypoints

        return obs


class ImageObservationWrapper(gym.ObservationWrapper):
    """Wrapper extending the state observation with an image of the environment.
    """

    def __init__(self, env: gym.Env, height: int, width: int):
        """Initializes image observation wrapper.

        Args:
            env (gym.Env): Environment to wrap
            height (int): Height of images to observe.
            width (int): Width of images to observe.
        """

        super().__init__(env)

        image_obs_space = gym.spaces.Box(low=0, high=255, shape=(height, width, 3))
        self.observation_space = gym.spaces.Dict({
            **self.observation_space,
            'image': image_obs_space
        })

    def observation(self, obs: Dict) -> Dict:
        """Extends existing observations with image.

        Args:
            obs (Dict): Original state observation.

        Returns:
            Dict: Observations with image.
        """

        obs['image'] = self.render()

        return obs


class VecFeaturePointObservationWrapper(VecEnvWrapper):
    """Vector Wrapper extending the state observation with feature points encoded from wrapped environments' images.
    """

    def __init__(self, venv: VecEnv, encoder: nn.Module, n_feature_points: int, antialias_factor: int = 1):
        """Initializes vector feature point observation wrapper.

        Note: Individual environments in `venv` are expected to be wrapped in a `ImageObservationWrapper`.

        Args:
            venv (VecEnv): Vector environment to wrap
            encoder (nn.Module): Encoder to use for image encoding to feature points
            n_feature_points (int): Number of feature points the encoder outputs.
            antialias_factor (int, optional): Antialias factor for image smoothing. Defaults to 1.
        """

        self.encoder = encoder
        self.antialias_factor = antialias_factor

        feature_point_obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(n_feature_points, 2))
        observation_space = gym.spaces.Dict({
            **{k: v for k, v in venv.observation_space.items() if k != 'image'},
            'feature_points': feature_point_obs_space
        })

        super().__init__(venv=venv, observation_space=observation_space)

    def reset(self) -> Dict:
        """Resets environment, extending existing observations with encoded feature points and removes images.

        Returns:
            Dict: Observations with feature points.
        """

        obs = self.venv.reset()

        images = np.array(obs['image'])
        obs['feature_points'] = self._get_feature_points(images)
        obs.pop('image', None)

        return obs

    def step_wait(self) -> VecEnvStepReturn:
        """Performs environment step, extending existing observations with encoded feature points and removes images.

        Returns:
            VecEnvStepReturn: Resulting observations with feature points, rewards, dones and infos.
        """

        obs, rewards, dones, infos = self.venv.step_wait()

        images = np.array(obs['image'])
        obs['feature_points'] = self._get_feature_points(images)
        obs.pop('image', None)

        for i, done in enumerate(dones):
            if not done:
                continue
            image = np.expand_dims(infos[i]['terminal_observation']['image'], axis=0)
            infos[i]['terminal_observation']['feature_points'] = self._get_feature_points(image)
            infos[i]['terminal_observation'].pop('image', None)

        return obs, rewards, dones, infos

    @torch.no_grad()
    def _get_feature_points(self, images: np.array) -> np.array:
        """Encodes images into feature points.

        Args:
            images (np.array): Batch of images to encode.

        Returns:
            np.array: Resulting feature points.
        """

        images = torch.tensor(images, device=gl.device) / 255
        images = antialias(images, factor=self.antialias_factor)
        images = torch.movedim(images, -1, -3)

        feature_points = self.encoder(images).cpu().numpy()

        return feature_points
