import numpy as np
from env import RoadSideRadar
from train import params_intersection, params_lane_changing

if __name__ == '__main__':
    scenario = 'lane_changing'
    if scenario == 'lane_changing':
        env = RoadSideRadar(params_lane_changing)
    elif scenario == 'intersection':
        env = RoadSideRadar(params_intersection)
    else:
        raise ValueError("Undefined scenario")
    obs = env.reset()
    done = False
    counter = 0
    if scenario == 'lane_changing':
        states = np.empty((10000, 11))
    elif scenario == 'intersection':
        states = np.empty((10000, 9))
    else:
        raise ValueError("Undefined scenario")
    for _ in range(500):
        while not done:
            action = np.random.rand(1)
            obs, rew, done, info = env.step(action)
            states[counter, :] = obs
            counter += 1

    print(np.mean(states[: counter, :], axis=0))
    print(np.std(states[: counter, :], axis=0))
