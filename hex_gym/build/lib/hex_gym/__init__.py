from gym.envs.registration import register

register(
    id='hexpod-v1',
    entry_point='hex_gym.envs:HexSimulator',
)