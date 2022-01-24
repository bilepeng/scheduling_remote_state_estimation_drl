from stable_baselines3 import PPO, SAC
from env import RoadSideRadar
from train import params_intersection, params_lane_changing
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm")
    parser.add_argument("--scenario")
    args = parser.parse_args()
    algorithm = args.algorithm
    scenario = args.scenario
    if scenario == 'intersection':
        env = RoadSideRadar(params_intersection)
    elif scenario == 'lane_changing':
        env = RoadSideRadar(params_lane_changing)
    else:
        raise ValueError('Undefined scenario.')
    if algorithm == 'PPO':
        if scenario == 'intersection':
            model = PPO.load('checkpoints/VoI_PPO_intersection_final.zip')
        elif scenario == 'lane_changing':
            model = PPO.load('checkpoints_ppo_lane_changing/rl_model_2000000_steps.zip')
        else:
            raise ValueError('Undefined scenario.')
    elif algorithm == 'SAC':
        if scenario == 'intersection':
            model = SAC.load('checkpoints_ppo_intersection/rl_model_1000000_steps.zip')
        elif scenario == 'lane_changing':
            model = SAC.load('checkpoints_sac_lane_changing/rl_model_800000_steps.zip')
        else:
            raise ValueError('Undefined scenario.')
    else:
        raise ValueError('Unknown algorithm.')
    done = False
    obs = env.reset()
    print(env.mode)
    epsd_rew = 0
    while not done:
        action, obs_ = model.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(action)
        epsd_rew += rew
        env.render()
        env.draw()
    print(epsd_rew)
    env.close()
