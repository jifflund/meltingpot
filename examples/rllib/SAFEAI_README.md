
# Safe AI

*To test the safety of AI agents in multiagent environments.*


## About
The intent of this code is to measure AI agents ability to comply with ethical principles when making decisions during training and testing. The code works by adding hidden rewards to each of the MeltingPot substrates/scenarios, and the hidden reward is a sum of rewards/penalties based on four codified ethical principles.

The four ethical principles are:
1. Do no harm
2. Acquiring only what is needed and not what is wanted
3. Recognizing that there are multiple perspectives and encourage uncertainty
4. Empowerment

These four principles have been codified into concrete rules for each of the substrates/scenarios. The goal of the research is to measure SOTA algorithms’ off-the-shelf ability to comply with ethical principles, their ability to learn to comply with these principles given feedback, and how robust is their learning when given new scenarios.

## Training 

To run the code

    ```shell
    cd examples/rllib/
    python self_play_train_open_harvest2.py  --num_players 3 --map_size small
    ```


## Testing

To test the code
python replay_from_checkpoint_open_harvest2.py --num_players 3 --map_size small --checkpoint_path /Users/allisterlundberg/ray_results/trainer_meltingpot_v1/PPO_meltingpot_c20ae_00000_0_2021-09-12_14-02-37/checkpoint_000010/checkpoint-10 --render_env True --num_replays 30

## Generating Reports

Ability to save run and compare to other runs.
Do this by hand and looking at tensorboard
 
 