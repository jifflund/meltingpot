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
from meltingpot.python import substrate, scenario
from ml_collections import config_dict
import ray
from ray import tune
from ray.rllib.agents.registry import get_trainer_class
from ray.tune.registry import register_env
import argparse

import multiagent_wrapper  # pylint: disable=g-bad-import-order


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--num_players', type=int, default=16, help='Number of players')
  parser.add_argument(
    '--map_size', type=str, default='large', help='Map size')
  parser.add_argument(
    '--checkpoint_path', type=str, default=None, help='Optional checkpoint to start training from')

  args = parser.parse_args()


  # function that outputs the environment you wish to register.
  def env_creator(env_config):
    # env = substrate.build(config_dict.ConfigDict(env_config))

    env = scenario.build(config_dict.ConfigDict(env_config))

    env = multiagent_wrapper.MeltingPotEnv(env)
    return env

  # We need the following pieces to run the training:
  # 1. The agent algorithm to use.
  agent_algorithm = "PPO"
  # 2. The name of the MeltingPot substrate, coming
  # from substrate.AVAILABLE_SUBSTRATES.
  # substrate_name = "allelopathic_harvest"
  # substrate_name = "commons_harvest_open"

  scenario_name = "commons_harvest_open_0"

  # 3. The number of CPUs to be used to run the training.
  num_cpus = 6

  # 4. Gets default training configuration and specifies the POMgame to load.
  config = copy.deepcopy(
      get_trainer_class(agent_algorithm)._default_config)  # pylint: disable=protected-access

  # 5. Set environment config. This will be passed to
  # the env_creator function via the register env lambda below.
  # config["env_config"] = substrate.get_config(substrate_name, num_players=args.num_players, map_size=args.map_size)

# TODO
  config["env_config"] = scenario.get_config(scenario_name)

  # 6. Register env
  register_env("meltingpot", env_creator)

  # 7. Extract space dimensions
  test_env = env_creator(config["env_config"])
  obs_space = test_env.single_player_observation_space()
  act_space = test_env.single_player_action_space()

  # 8. Configuration for multiagent setup with policy sharing:
  config["multiagent"] = {
      "policies": {
          # the first tuple value is None -> uses default policy
          "av": (None, obs_space, act_space, {}),
      },
      "policy_mapping_fn": lambda agent_id, **kwargs: "av",
      "test_param1": 123
  }
  config['framework'] = 'torch'
  config["test_param12"]: 345
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

  # 9. Initialize ray and trainer object
  ray.init(num_cpus=num_cpus + 1)

  # 10. Set stopper function
  # This will run fairly quickly when 3 agents
  # def stopper(trial_id, result):
  #   return result["episode_reward_mean"] > 24

  # Use this to train forever
  def stopper(trial_id, result):
    return False


  checkpoint_path = args.checkpoint_path
  # checkpoint_path = '/Users/allisterlundberg/ray_results/trainer_meltingpot_v1/PPO_meltingpot_374d8_00000_0_2021-08-02_13-07-08/checkpoint_000020/checkpoint-20'

  analysis = tune.run(
    "PPO",
    name='trainer_meltingpot_v1',
    stop=stopper,
    verbose=3,
    checkpoint_at_end=True,
    keep_checkpoints_num=100,
    # TODO May use this for further analysis on how it learns, but for now use just most recent N using keep_checkpoints param
    # checkpoint_score_attr='min-validation_loss',
    checkpoint_freq=1,
    config=config,
    restore=checkpoint_path
  )

  last_checkpoint = analysis.get_last_checkpoint(
    metric="episode_reward_mean", mode="max"
  )
  print('last_checkpoint', last_checkpoint)


if __name__ == "__main__":
  main()
