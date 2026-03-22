try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces
import numpy as np
from envs.env_core import EnvCore


class ContinuousActionEnv(object):
    """
    连续动作环境封装, 支持异构动作维度 (用户和无人机动作维度不同)
    """

    def __init__(self):
        self.env = EnvCore()
        self.num_agent = self.env.agent_num
        self.signal_obs_dim = self.env.obs_dim

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0
        for agent_id in range(self.num_agent):
            action_dim = self.env.action_dims[agent_id]
            self.action_space.append(spaces.Box(
                low=-np.inf, high=+np.inf,
                shape=(action_dim,), dtype=np.float32))

            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf,
                shape=(self.signal_obs_dim,), dtype=np.float32))

        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf,
                       shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agent)
        ]

    def step(self, actions):
        results = self.env.step(actions)
        obs, rews, dones, infos = results
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = self.env.reset()
        return np.stack(obs)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass
