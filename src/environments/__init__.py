from gymnasium.envs.registration import register


for reward_type in ['sparse', 'dense']:
    for control_type in ['ee', 'joints']:
        reward_suffix = 'Dense' if reward_type == 'dense' else ''
        control_suffix = 'Joints' if control_type == 'joints' else ''
        kwargs = {'reward_type': reward_type, 'control_type': control_type}

        register(
            id='PandaPush{}{}-custom'.format(control_suffix, reward_suffix),
            entry_point='src.environments.panda:PandaPushCustomEnv',
            kwargs=kwargs,
            max_episode_steps=50
        )
