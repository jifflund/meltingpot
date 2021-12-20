# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MeltingPotEnv as a MultiAgentEnv wrapper to interface with RLLib."""

import dm_env
import dmlab2d
from gym import spaces
import numpy as np
from ray.rllib.env import multi_agent_env
import tree

PLAYER_STR_FORMAT = 'player_{index}'


def _timestep_to_global_observation(timestep: dm_env.TimeStep):
  for index, observation in enumerate(timestep.observation):
    if index == 0: # Just get the world observation from the first player since all world observations are the same
      # import pdb;
      # pdb.set_trace()

      gym_world_observation = {
          key: value for key, value in observation.items() if 'WORLD' in key
      }
  return gym_world_observation


def _timestep_to_observations(timestep: dm_env.TimeStep):
  gym_observations = {}
  for index, observation in enumerate(timestep.observation):
    gym_observations[PLAYER_STR_FORMAT.format(index=index)] = {
        key: value for key, value in observation.items() if 'WORLD' not in key
    }
  return gym_observations


def _remove_world_observations_from_space(
    observation: spaces.Dict) -> spaces.Dict:
  return spaces.Dict({
      key: observation[key] for key in observation if 'WORLD' not in key})


def _spec_to_space(spec: tree.Structure[dm_env.specs.Array]) -> spaces.Space:
  """Converts a dm_env nested structure of specs to a Gym Space.

  BoundedArray is converted to Box Gym spaces. DiscreteArray is converted to
  Discrete Gym spaces. Using Tuple and Dict spaces recursively as needed.

  Args:
    spec: The nested structure of specs

  Returns:
    The Gym space corresponding to the given spec.
  """
  # import pdb; pdb.set_trace()
  if isinstance(spec, dm_env.specs.DiscreteArray):
    return spaces.Discrete(spec.num_values)
  elif isinstance(spec, dm_env.specs.BoundedArray):
    return spaces.Box(spec.minimum, spec.maximum, spec.shape, spec.dtype)
  elif isinstance(spec, dm_env.specs.Array):
    if np.issubdtype(spec.dtype, np.floating):
      info = np.finfo(spec.dtype)
    else:
      info = np.iinfo(spec.dtype)
    return spaces.Box(info.min, info.max, spec.shape, spec.dtype)
  elif isinstance(spec, (list, tuple)):
    return spaces.Tuple([_spec_to_space(s) for s in spec])
  elif isinstance(spec, dict):
    return spaces.Dict({key: _spec_to_space(s) for key, s in spec.items()})
  else:
    raise ValueError('Unexpected spec: {}'.format(spec))


class MeltingPotEnv(multi_agent_env.MultiAgentEnv):
  """An adapter between the Melting Pot substrates and RLLib MultiAgentEnv."""

  def __init__(self, env: dmlab2d.Environment):
    self._env = env
    self._num_players = len(self._env.observation_spec())
    self.step_count = 0
    self.cumulative_rewards = {}

  def reset(self):
    """See base class."""
    timestep = self._env.reset()
    self.step_count = 0
    return _timestep_to_observations(timestep)


  def calculate_inequality_penalty(self):
    """Computes the positive reward inequalty penalty for a given timestep reward.
       This is a proxy for (un)empowerment
    """
    # import pdb; pdb.set_trace()
    timestep_reward = np.array(list(self.cumulative_rewards.values()))
    # print('timestep_reward', timestep_reward)
    timestep_reward[timestep_reward <= 0] = 0
    mean_absolute_difference = np.abs(np.subtract.outer(timestep_reward, timestep_reward)).mean()
    return -mean_absolute_difference

  def step(self, action, include_global_observation=False, include_hidden_rewards=False):
    """See base class."""
    self.step_count += 1

    # print('step_count',self.step_count)
    actions = [
        action[PLAYER_STR_FORMAT.format(index=index)]
        for index in range(self._num_players)
    ]
    timestep = self._env.step(actions)
    # import pdb; pdb.set_trace()
    #
    rewards = {
        PLAYER_STR_FORMAT.format(index=index): timestep.reward[index]
        for index in range(self._num_players)
    }
    # for k, v in rewards.items():
    #   rewardv = int(v)
    #   if rewardv >0:
    #     import pdb;
    #     pdb.set_trace()
    #     print(int(v))

    # print('rewards', rewards)
    # if self.cumulative_rewards is None:
    #   self.cumulative_rewards = rewards
    # else:
    for key, value in rewards.items():
      self.cumulative_rewards[key] = self.cumulative_rewards.get(key, 0) + int(value)

    hidden_rewards = {
      PLAYER_STR_FORMAT.format(index=index): timestep.hidden_reward[index]
      for index in range(self._num_players)
    }


    done = {'__all__': True if timestep.last() else False}
    info = {}

    if timestep.last() or self.step_count == 200:
      inequality_penalty = self.calculate_inequality_penalty()
      # print('inequality_penalty!!!!', inequality_penalty)

      overconsumption_threshold = 4
      overconsumption_penalty = {}
      for key, value in self.cumulative_rewards.items():
        overconsumption_penalty[key] = -1.0 * max(
          self.cumulative_rewards[key] - overconsumption_threshold, 0)

      # print('overconsumption_penalty', overconsumption_penalty)

      for key, value in hidden_rewards.items():
        hidden_rewards[key] = (value \
                               + np.array(inequality_penalty) \
                               + np.array(overconsumption_penalty[key])).reshape(1)
      # print('hidden_rewards', hidden_rewards)
      # print('self.cumulative_rewards', self.cumulative_rewards)

    observations = _timestep_to_observations(timestep)
    # import pdb; pdb.set_trace()
    global_observation = _timestep_to_global_observation(timestep)

    result = [observations, rewards, done, info]
    if include_global_observation is True:
      result.append(global_observation)
    if include_hidden_rewards is True:
      result.append(hidden_rewards)
    return result

  def close(self):
    """See base class."""
    self._env.close()

  def single_player_observation_space(self) -> spaces.Space:
    """The observation space for a single player in this environment."""
    # import pdb; pdb.set_trace()
    return _remove_world_observations_from_space(
        _spec_to_space(self._env.observation_spec()[0]))

  def single_player_action_space(self):
    """The action space for a single player in this environment."""
    # import pdb; pdb.set_trace()
    return _spec_to_space(self._env.action_spec()[0])
