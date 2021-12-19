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
"""Runs an example of a self-play training experiment."""

import copy
import os
from meltingpot.python import substrate
from ml_collections import config_dict
import ray
from ray import tune
from ray.rllib.agents.registry import get_trainer_class
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
import argparse
import numpy as np
import pygame
import multiagent_wrapper  # pylint: disable=g-bad-import-order
import time


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--num_players', type=int, default=16, help='Number of players')
  parser.add_argument(
    '--map_size', type=str, default='large', help='Map size')
  parser.add_argument(
    '--checkpoint_path', type=str, required=True, help='checkpoint to trained agents')
  parser.add_argument(
    '--render_env', type=str, default='False', help='Render the environment using pygame if True'
  )
  parser.add_argument(
    '--num_replays', type=int, default=1, help='Number of replays of the game')

  args = parser.parse_args()



  # function that outputs the environment you wish to register.
  # TODO add in way ot build a scenario in additional to a substrate
  def env_creator(env_config):
    env = substrate.build(config_dict.ConfigDict(env_config))
    env = multiagent_wrapper.MeltingPotEnv(env)
    return env


  # We need the following 3 pieces to run the training:
  # 1. The agent algorithm to use.
  agent_algorithm = "PPO"
  # 2. The name of the MeltingPot substrate, coming
  # from substrate.AVAILABLE_SUBSTRATES.
  # substrate_name = "allelopathic_harvest"
  substrate_name = "commons_harvest_open"
  # 3. The number of CPUs to be used to run the training.
  num_cpus = 1

  # 4. Gets default training configuration and specifies the POMgame to load.
  config = copy.deepcopy(
      get_trainer_class(agent_algorithm)._default_config)  # pylint: disable=protected-access

  # 5. Set environment config. This will be passed to
  # the env_creator function via the register env lambda below.
  config["env_config"] = substrate.get_config(substrate_name, num_players=args.num_players, map_size=args.map_size)

  # 6. Register env
  register_env("meltingpot", env_creator)

  # 7. Extract space dimensions
  test_env = env_creator(config["env_config"])
  obs_space = test_env.single_player_observation_space()
  # print('obs_space', obs_space)
  act_space = test_env.single_player_action_space()

  # 8. Configuration for multiagent setup with policy sharing:
  config["multiagent"] = {
      "policies": {
          # the first tuple value is None -> uses default policy
          "av": (None, obs_space, act_space, {}),
      },
      "policy_mapping_fn": lambda agent_id, **kwargs: "av"
  }

  config['framework'] = 'torch'

  # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
  config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
  config["log_level"] = "DEBUG"
  config["num_workers"] = 1
  # Fragment length, collected at once from each worker and for each agent!
  config["rollout_fragment_length"] = 30
  # Training batch size -> Fragments are concatenated up to this point.
  config["train_batch_size"] = 200
  # After n steps, force reset simulation
  config["horizon"] = 200
  # Default: False
  config["no_done_at_end"] = False
  # Info: If False, each agents trajectory is expected to have
  # maximum one done=True in the last step of the trajectory.
  # If no_done_at_end = True, environment is not resetted
  # when dones[__all__]= True.
  config['env'] = 'meltingpot'

  # 9. Initialize pygame
  if args.render_env == 'True':
    pygame.init()
    pygame.display.set_caption('Melting Pot: {}'.format(substrate_name))
    text_font_size = 10
    font = pygame.font.SysFont(None, text_font_size)
    scale = 6
    observation_width = 800
    observation_height = 600
    game_display = pygame.display.set_mode(
      (observation_width * scale, observation_height * scale))

  # 10. Initialize ray and trainer object
  ray.init(num_cpus=num_cpus + 1)

  # checkpoint_path = '/Users/allisterlundberg/ray_results/trainer_meltingpot_v1/PPO_meltingpot_374d8_00000_0_2021-08-02_13-07-08/checkpoint_000020/checkpoint-20'
  # checkpoint_path = '/Users/allisterlundberg/ray_results/trainer_meltingpot_v1/PPO_meltingpot_374d8_00000_0_2021-08-02_13-07-08/checkpoint_000016/checkpoint-16'
  agent = ppo.PPOTrainer(config=config, env='meltingpot')
  agent.restore(args.checkpoint_path)

  done = {'agent_0': False}
  episode_length = 25
  length_count = 0

  max_step = 100

  for i in range(args.num_replays):
    print(f'Replay number {i} of {args.num_replays}')
    episode_reward = 0
    episode_hidden_reward = 0
    step_count = 0
    # import pdb;
    # pdb.set_trace()

    obs = test_env.reset()
    # import pdb;
    # pdb.set_trace()

    while step_count <= max_step:
      # if done['agent_0'] is not False:
      #   obs = test_env.reset()
      #   print('RESETTING')

      actions = {}
      for player in obs.keys():
        # print(player)
        
        # import pdb; pdb.set_trace()
        action = agent.compute_action(obs[player], policy_id="av", explore=True) # This is required otherise trainged agents may get stuck
        actions[player] = action


      obs, reward, done, info, global_observation, hidden_reward = test_env.step(
        actions, include_global_observation=True, include_hidden_rewards=True
      )

      # import pdb;   pdb.set_trace()

      if args.render_env == 'True':
        global_observation = global_observation['WORLD.RGB']
        global_observation = np.transpose(global_observation, (1, 0, 2))  # PyGame is column major!
        surface = pygame.surfarray.make_surface(global_observation)
        rect = surface.get_rect()
        surf = pygame.transform.scale(surface, (rect[2] * scale, rect[3] * scale))
        game_display.blit(surf, dest=(0, 0))
        pygame.display.update()
        # time.sleep(.1)

      step_count += 1
      episode_reward += sum(reward.values())
      episode_hidden_reward += sum(hidden_reward.values())
      print('episode_reward', episode_reward)
      print('episode_hidden_reward', episode_hidden_reward)

    print('total episode_reward', episode_reward)
    print('total episode_hidden_reward', episode_hidden_reward)



  test_env.close()
  print('finished!')


if __name__ == "__main__":
  main()
