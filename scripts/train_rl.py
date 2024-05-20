import logging
import sys
from collections import deque
from typing import Union

import gymnasium
import gymnasium as gym
import hydra
import panda_gym
import torch
from omegaconf import DictConfig

import __init__
from src.environments.wrappers import (ImageObservationWrapper, KeypointObservationWrapper,
                                       MeasurableObservationWrapper, NotInstantlySolvedWrapper,
                                       VecFeaturePointObservationWrapper)
from src.models.sae import assemble_sae
from src.utils import Bunch, check_gpu, get_display, get_hydra_run_dir, gl, is_rerun, print_cfg, root, setup_wandb

sys.modules['gym'] = gymnasium  # see [PR](https://github.com/DLR-RM/stable-baselines3/pull/780)
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback, EventCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor

log = logging.getLogger(__name__)


def train_rl(cfg: DictConfig, model: Union[SAC, PPO], env_eval: gym.Env):
    """Runs the main learning method.

    Args:
        cfg (DictConfig): Hydra configuration object.
        model (Union[SAC, PPO]): Model instance to train.
        env_eval (gym.Env): Environment instance to run evaluation on.
    """

    model.learn(
        cfg.training.steps,
        log_interval=cfg.training.n_environments,  # episodes
        reset_num_timesteps=False,
        progress_bar=True,
        callback=[
            AdjustBufferCallback(),
            EvalCallback(env_eval,
                         n_eval_episodes=cfg.training.evaluation.n_episodes,  # complete vector env once
                         eval_freq=(cfg.training.evaluation.frequency // cfg.training.n_environments),  # steps
                         best_model_save_path=get_hydra_run_dir())
        ]
    )


class AdjustBufferCallback(EventCallback):
    """Adjusts episode info buffers to have as many entries as there are parallel environments to prevent rollout log smoothing.
    """

    def _on_training_start(self):

        if self.model.ep_info_buffer.maxlen != self.training_env.num_envs:
            self.model.ep_info_buffer = deque(maxlen=self.training_env.num_envs)
            self.model.ep_success_buffer = deque(maxlen=self.training_env.num_envs)


def setup_environment(cfg: DictConfig) -> gym.Env:
    """Sets up a single instance of the environment with all needed wrappers, depending on the desired state space.

    Args:
        cfg (DictConfig): Hydra configuration object.

    Returns:
        gym.Env: Initialized environment.
    """

    # set up environment
    env = gym.make(cfg.environment.id,
                   render_mode='rgb_array',
                   max_episode_steps=cfg.environment.time_limit,
                   renderer=cfg.environment.camera.renderer,
                   render_height=(cfg.environment.camera.height * cfg.environment.camera.antialias_factor),
                   render_width=(cfg.environment.camera.width * cfg.environment.camera.antialias_factor),
                   render_target_position=cfg.environment.camera.target_position,
                   render_distance=cfg.environment.camera.distance,
                   render_yaw=cfg.environment.camera.yaw,
                   render_pitch=cfg.environment.camera.pitch,
                   render_roll=cfg.environment.camera.roll)

    # prevent resetting environments to a solved state
    env = NotInstantlySolvedWrapper(env)

    # strip measurable or immeasurable information from observation
    env = MeasurableObservationWrapper(env,
                                       include_measurable=cfg.training.observation.measurable,
                                       include_immeasurable=cfg.training.observation.immeasurable)

    if cfg.training.observation.keypoints:
        # add keypoints to state observation
        env = KeypointObservationWrapper(env,
                                         objects=cfg.environment.keypoints.objects,
                                         links=cfg.environment.keypoints.links,
                                         **cfg.environment.camera)

    if cfg.training.observation.feature_points:
        # add image observation to encode into feature points
        env = ImageObservationWrapper(env,
                                      height=(cfg.environment.camera.width * cfg.environment.camera.antialias_factor),
                                      width=(cfg.environment.camera.height * cfg.environment.camera.antialias_factor))

    return env


def setup_environments(cfg: DictConfig, n_environments: int) -> SubprocVecEnv:
    """Sets up a vector environment according to the passed configuration.

    Args:
        cfg (DictConfig): Hydra configuration object.
        n_environments(int): Number of individual environments to initialize.

    Returns:
        SubprocVecEnv: Set up vector environment.
    """

    venv = SubprocVecEnv([(lambda: setup_environment(cfg))
                          for _ in range(n_environments)])

    if cfg.training.observation.feature_points:
        # add feature points to state observation
        if cfg.training.sae_checkpoint is not None:
            # load model checkpoint
            checkpoint = Bunch(**torch.load(root / cfg.training.sae_checkpoint,
                                            map_location=gl.device))
            sae = assemble_sae(checkpoint.cfg)
            sae.load_state_dict(checkpoint.model_state_dict)
            encoder = sae.encoder
            encoder.eval()
            n_feature_points = checkpoint.cfg.model.encoder.settings.n_coordinates
        else:
            raise ValueError(
                'cfg.training.observation.feature_points is set to true but cfg.training.sae_checkpoint is None.')

        # use feature points for state observations
        venv = VecFeaturePointObservationWrapper(venv,
                                                 encoder=encoder,
                                                 n_feature_points=n_feature_points,
                                                 antialias_factor=cfg.environment.camera.antialias_factor)

    if  cfg.training.observation.stack_frames is not None:
        # stack state observation of multiple frames
        venv = VecFrameStack(venv, n_stack=cfg.training.observation.stack_frames)

    venv = VecMonitor(venv)

    return venv


def setup_model(cfg: DictConfig, env: gym.Env, rerun: bool) -> Union[SAC, PPO]:
    """Sets up the desired Stable Baselines3 model (SAC or PPO) with the needed policy.

    Args:
        cfg (DictConfig): Hydra configuration object.
        env (gym.Env): Environment this model is to be trained on.
        rerun (bool): Whether this is a resumed run and the model should be loaded.

    Raises:
        ValueError: Raised if selected model is not SAC or PPO.

    Returns:
        Union[SAC, PPO]: Initialized SAC or PPO model.
    """

    # determine configured training algorithm
    if cfg.algorithm.id == 'sac':
        algorithm = SAC
    elif cfg.algorithm.id == 'ppo':
        algorithm = PPO
    else:
        raise ValueError(
            f'Expected "sac" or "ppo" in cfg.algorithm.id. Got {cfg.algorithm} instead.')
    keywords = cfg.algorithm.settings or {}

    if rerun:
        # restore saved model and replay buffer
        model = algorithm.load(get_hydra_run_dir() / 'final_model.zip', env=env)
        if hasattr(model, 'load_replay_buffer'):  # in case of off-policy algorithm
            model.load_replay_buffer(get_hydra_run_dir() / 'final_replay_buffer.pickle')

    else:
        # choose suitable policy for configured observation space
        if isinstance(env.observation_space, gym.spaces.dict.Dict):
            policy = 'MultiInputPolicy'
        else:
            policy = 'MlpPolicy'

        # instantiate RL model
        model = algorithm(policy, env, **keywords, device=gl.device)

    # setup tensorboard logger
    logger = configure(str(get_hydra_run_dir() / 'tb'), ['tensorboard'])
    model.set_logger(logger)

    return model


def store_model(model: BaseAlgorithm, tag: str = None):
    """Stores model and it's replay buffer as `tag_model.zip` and `tag_replay_buffer.pickle`.

    Args:
        model (BaseAlgorithm): The model to store.
        tag (str, optional): Tag to prepend the stored model file with. Defaults to None.
    """

    # save model and replay buffer
    model_name = (tag + '_model.zip') if tag is not None else 'model.zip'
    model.save(get_hydra_run_dir() / model_name)
    if hasattr(model, 'save_replay_buffer'):  # in case of off-policy algorithm
        replay_buffer_name = (tag + '_replay_buffer.zip') if tag is not None else 'replay_buffer.zip'
        model.save_replay_buffer(get_hydra_run_dir() / replay_buffer_name)


@hydra.main(version_base=None, config_path='../configs', config_name='train_rl')
def main(cfg: DictConfig):
    """Main program entry point.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """

    print_cfg(cfg)
    check_gpu(cfg.gpu)
    rerun = is_rerun()
    if rerun:
        log.info('Resuming previous run')
    run = setup_wandb(cfg, rerun)

    with get_display(cfg.display):

        # set up environments and model
        log.info('Setting up environments')
        venv_train = setup_environments(cfg, cfg.training.n_environments)
        venv_eval = setup_environments(cfg, cfg.training.evaluation.n_environments)

        log.info('Setting up model')
        model = setup_model(cfg, venv_train, rerun)

        # perform learning
        try:
            log.info('Learning starts')
            train_rl(cfg, model, venv_eval)

            log.info('Saving model and replay buffer')
            store_model(model, tag='final')
        except Exception as e:
            log.error('Encountered error')
            log.info('Saving model and replay buffer')
            store_model(model, tag=e.__class__.__name__)
            raise e
        finally:
            log.info('Closing environments')
            venv_train.close()
            venv_eval.close()


if __name__ == '__main__':
    main()
