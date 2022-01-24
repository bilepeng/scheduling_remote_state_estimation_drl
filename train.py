from env import RoadSideRadar
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import argparse


params_intersection = {'scenario': 'intersection',
                       'pi': np.eye(2),
                       'initial_model_probabilities': np.array([0.5, 0.5])}
params_lane_changing = {'scenario': 'lane_changing',
                        'pi': np.array([[0.999, 0.0005, 0.0005], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]),
                        'initial_model_probabilities': np.array([0.999, 0.0005, 0.0005])}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm")
    parser.add_argument("--scenario")
    args = parser.parse_args()
    if args.scenario == "intersection":
        env = RoadSideRadar(params_intersection)
    elif args.scenario == "lane_changing":
        env = RoadSideRadar(params_lane_changing)
    else:
        raise ValueError("Undefined scenario.")

    if args.algorithm == "PPO":
        checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='checkpoints_ppo_' + args.scenario)
        model = PPO(policy='MlpPolicy',
                    env=env,
                    batch_size=2048,
                    clip_range=0.1,
                    learning_rate=1e-5,
                    tensorboard_log='tblog')
        model.learn(total_timesteps=int(2e6), tb_log_name='RoadSideRadar_PPO_' + args.scenario,
                    callback=checkpoint_callback)
        model.save('checkpoints/VoI_PPO_' + args.scenario + "_final")
    elif args.algorithm == "SAC":
        checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='checkpoints_sac_' + args.scenario)
        model = SAC(policy="MlpPolicy",
                    env=env,
                    learning_rate=1e-5,
                    batch_size=2048,
                    tensorboard_log="tblog",
                    ent_coef=3)
        model.learn(total_timesteps=int(2e6), tb_log_name="RoadSideRadar_SAC_" + args.scenario,
                    callback=checkpoint_callback)
        model.save("checkpoints/VoI_SAC_" + args.scenario + "_final")
