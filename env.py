import numpy as np
from gym import Env, spaces
from vehicle_movement import update_true_state_intersection, update_state_straight, update_state_turning, \
    mix_state_v2, update_model_probabilities_v2, output_estimate_v2, update_true_state_lane_changing, \
    update_state_changing_left, update_state_changing_right
from time import sleep
try:
    from gym.envs.classic_control import rendering
except:
    no_rendering = True
else:
    no_rendering = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class RoadSideRadar(Env):

    def __init__(self, params):
        super(RoadSideRadar, self).__init__()
        self.delta_theta = 0.1
        self.delta_t = 0.1  # seconds between state updates
        self.true_state = np.array([0, -10, 0, 5])
        self.x_imm = np.array([0, -10, 0, 5])
        self.H = np.eye(4)
        self.R = np.eye(4) * 0.0005  # Measurement noise
        self.Q = np.eye(4) * 0.0005  # Process noise
        self.z = self.H.dot(self.true_state) + np.random.randn(4) * np.sqrt(np.diag(self.R))
        self.K = np.zeros(4)
        self.pi = params['pi']
        self.initial_u = params['initial_model_probabilities']
        self.u = params['initial_model_probabilities']
        self.previous_u = params['initial_model_probabilities']
        self.p_th = 0.01  # threshold of tolerance
        self.daterate = 640  # date rate
        self.W = 8500  # bandwidth
        self.N_0 = 1
        self.R_t = 0.5  # A random number to compare with the p_t_c_out
        self.p_t_c_out = 0.3  # The probability of communication outage
        if params['scenario'] == 'intersection':
            self.scenario = 'intersection'
            self.mode = np.random.randint(1, 3)
        elif params['scenario'] == 'lane_changing':
            self.scenario = 'lane_changing'
            self.mode = np.random.randint(0, 3)
            self.y_lane_changing = 0
        self.n_histories_max = 512 * 1
        self.n_histories_resampling = 128 * 1

        self.min_x_position = -5
        self.min_y_position = -10
        self.max_x_position = 20
        self.max_y_position = 20
        self.min_x_speed = 0
        self.min_y_speed = 0
        self.max_x_speed = 10
        self.max_y_speed = 10
        self.mean_channel_gain = 0.0001

        # rendering related
        self.tx_power = 0
        self.viewer = None
        self.power_line = None
        self.positions_x = np.zeros(5000)
        self.positions_y = np.zeros(5000)
        self.estimated_positions_x = np.zeros(5000)
        self.estimated_positions_y = np.zeros(5000)
        self.t = 0
        self.track = None
        self.scale = 15
        self.offset_x = -150
        self.offset_y = -220
        self.offset = np.array([self.offset_x, self.offset_y])
        self.list_a = []
        self.list_b = []
        self.list = []
        self.p1s = list()
        self.p2s = list()
        self.histories = list()
        self.low_state = np.array(
            [self.min_x_position, self.min_y_position,
             self.min_x_speed, self.min_y_speed], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_x_position, self.max_y_position,
             self.max_x_speed, self.max_y_speed], dtype=np.float32
        )

        self.action_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)  # transmitting power

        if params['scenario'] == 'intersection':
            self.obs_mean = np.array([6.42987356e+00, 1.80811250e+00, 3.01491120e+00, 2.31884103e+00,
                                      3.91051186e-03,
                                      3.68011503e-01, 6.31988492e-01,
                                      3.68011503e-01, 6.31988492e-01,
                                      1.75173870e-01])
            self.obs_std = np.array([6.74205928e+00, 4.67602043e+00, 2.19415501e+00, 2.24681539e+00,
                                     5.62962714e-03,
                                     1.96174367e-01, 1.96174369e-01,
                                     1.96174367e-01, 1.96174369e-01,
                                     1.83451270e-01])
            self.observation_space = spaces.Box(
                low=-10,
                high=10,
                dtype=np.float32,
                shape=(10,)
            )
        elif params['scenario'] == 'lane_changing':
            self.obs_mean = np.array([-2.47733325e-01,  5.32199113e+00, -5.29116361e-02,  4.93618407e+00,
                                      2.13777165e-03,
                                      9.99605158e-01,  1.97553825e-04,  1.97286036e-04,
                                      9.99605158e-01,  1.97553825e-04,  1.97286036e-04,
                                      5.95310674e-02])
            self.obs_std = np.array([1.44408992e-01, 8.64371711e+00, 5.17086300e-02, 4.99023876e-02,
                                     1.43734134e-04,
                                     5.05563600e-04, 2.52718155e-04, 2.52895869e-04,
                                     5.05563600e-04, 2.52718155e-04, 2.52895869e-04,
                                     2.04303221e-02])
            self.observation_space = spaces.Box(
                low=-10,
                high=10,
                dtype=np.float32,
                shape=(12,)
        )

    def step(self, action: np.ndarray):
        self.tx_power = action[0]
        # The probability of communication outage
        self.p_t_c_out = 1 - np.exp(-(2 ** (self.daterate / self.W) - 1) /
                                    (action[0] * self.mean_channel_gain / self.N_0 * self.W + 1e-9))
        if self.scenario == 'intersection':
            self.true_state = update_true_state_intersection(self.true_state, self.mode)
        else:
            self.true_state = update_true_state_lane_changing(self.true_state, self.mode, self.y_lane_changing)
        self.positions_x[self.t] = self.true_state[0]
        self.positions_y[self.t] = self.true_state[1]
        if self.t == 0:
            self.estimated_positions_x[0] = self.positions_x[0]
            self.estimated_positions_y[0] = self.positions_y[0]
        self.z = self.H.dot(self.true_state) + np.random.normal(0, 1, 4) * np.sqrt(np.diag(self.R))

        self.previous_u = self.histories[-1]['u']
        updated_histories = list()
        for history in self.histories:
            p, x_hat, u, P = history['p'], history['x_hat'], history['u'], history['P']

            x_hat_mix, P_mix, c = mix_state_v2(x_hat, self.pi, u, P)
            if self.scenario == 'intersection':
                x_hat_turn, P_turn, r_1 = update_state_turning(x_hat_mix[1], P_mix[1], None)
                x_hat_straight, P_straight, r_0 = update_state_straight(x_hat_mix[0], P_mix[0], None)
                new_p = p * self.p_t_c_out
                new_history = {'p': new_p,
                               'x_hat': [x_hat_straight, x_hat_turn],
                               'u': c,
                               'r': np.array([0, 0]),
                               'P': [P_straight, P_turn],}
                updated_histories.append(new_history)
            elif self.scenario == 'lane_changing':
                x_hat_straight, P_straight, r_straight = update_state_straight(x_hat_mix[0], P_mix[0], None)
                x_hat_left, P_left, r_left = update_state_changing_left(x_hat_mix[1], P_mix[1], None)
                x_hat_right, P_right, r_right = update_state_changing_right(x_hat_mix[2], P_mix[2], None)
                new_p = p * self.p_t_c_out
                new_history = {'p': new_p,
                               'x_hat': [x_hat_straight, x_hat_left, x_hat_right],
                               'u': c,
                               'r': np.array([0, 0, 0]),
                               'P': [P_straight, P_left, P_right], }
                updated_histories.append(new_history)
            else:
                raise ValueError('Scenario undefined.')

        # History with latest measurement
        x_hat, u, P = self.histories[-1]['x_hat'], self.histories[-1]['u'], \
                      self.histories[-1]['P']
        x_hat_mix, P_mix, c = mix_state_v2(x_hat, self.pi, u, P)
        if self.scenario == 'intersection':
            x_hat_turn_m, P_turn_m, r_1_m = update_state_turning(x_hat_mix[1], P_mix[1], self.z)
            x_hat_straight_m, P_straight_m, r_0_m = update_state_straight(x_hat_mix[0], P_mix[0], self.z)
            u = update_model_probabilities_v2([r_0_m, r_1_m], [P_straight_m, P_turn_m], c)
            new_history_m = {'p': 1 - self.p_t_c_out,
                             'x_hat': [x_hat_straight_m, x_hat_turn_m],
                             'u': u,
                             'r': np.array([np.linalg.norm(r_0_m), np.linalg.norm(r_1_m)]),
                             'P': [P_straight_m, P_turn_m],}
            updated_histories.append(new_history_m)

        elif self.scenario == 'lane_changing':
            x_hat_straight_m, P_straight_m, r_straight_m = update_state_straight(x_hat_mix[0], P_mix[0], self.z)
            x_hat_left_m, P_left_m, r_left_m = update_state_changing_left(x_hat_mix[1], P_mix[1], self.z)
            x_hat_right_m, P_right_m, r_right_m = update_state_changing_right(x_hat_mix[2], P_mix[2], self.z)
            u = update_model_probabilities_v2([r_straight_m, r_left_m, r_right_m],
                                              [P_straight_m, P_left_m, P_right_m], c)
            new_history_m = {'p': 1 - self.p_t_c_out,
                             'x_hat': [x_hat_straight_m, x_hat_left_m, x_hat_right_m],
                             'u': u,
                             'r': np.array([np.linalg.norm(r_straight_m),
                                            np.linalg.norm(r_left_m), np.linalg.norm(r_right_m)]),
                             'P': [P_straight_m, P_left_m, P_right_m], }
            updated_histories.append(new_history_m)
        else:
            raise ValueError('Scenario undefined.')

        self.histories = updated_histories

        self.x_imm, self.P_imm, self.u, self.r = 0, 0, 0, 0
        for history in self.histories:
            x_imm_p, P_imm_p = output_estimate_v2(history['x_hat'], history['u'], history['P'])
            self.x_imm += x_imm_p * history['p']
            self.P_imm += P_imm_p * history['p']
            self.u += history['u'] * history['p']
            self.r += history['r'] * history['p']

        self.estimated_positions_x[self.t] = self.x_imm[0]
        self.estimated_positions_y[self.t] = self.x_imm[1]

        rew = -action[0] - 100 * max(0, np.trace(self.P_imm) - self.p_th) - 10 * (np.trace(self.P_imm) > self.p_th)
        # if np.trace(self.P_imm) - self.p_th > 0:
        #     print("Oops")
        self.done = bool(self.true_state[0] > 20 or self.true_state[1] > 20)  # out of bounds

        self.t += 1

        return self.obs, rew, self.done, {}

    @property
    def obs(self):
        obs = np.hstack((self.z,
                         np.trace(self.P_imm),
                         self.previous_u,
                         self.histories[-1]['u'],
                         np.sum(self.r * self.u)))
        obs = (obs - self.obs_mean) / self.obs_std
        return obs

    def reset(self):
        if self.scenario == 'intersection':
            P_straight = np.eye(4) * 0.001
            P_turn = np.eye(4) * 0.001
            self.P_imm = np.eye(4) * 0.001
            x_hat_straight = self.true_state + np.random.randn(4) * np.sqrt(np.diag(P_straight))
            x_hat_turn = x_hat_straight
            self.mode = np.random.randint(1, 3)
            self.r = np.zeros(2)
            new_history = {'p': 1.,
                           'x_hat': [x_hat_straight, x_hat_turn],
                           'u': self.initial_u,
                           'r': np.array([0, 0]),
                           'P': [P_straight, P_turn],}
        elif self.scenario == 'lane_changing':
            self.y_lane_changing = np.random.random() * 20 - 10
            P_straight = np.eye(4) * 0.001
            P_left = np.eye(4) * 0.001
            P_right = np.eye(4) * 0.001
            self.P_imm = np.eye(4) * 0.001
            x_hat_straight = self.true_state + np.random.randn(4) * np.sqrt(np.diag(P_straight))
            x_hat_left = self.true_state + np.random.randn(4) * np.sqrt(np.diag(P_left))
            x_hat_right = self.true_state + np.random.randn(4) * np.sqrt(np.diag(P_right))
            self.mode = np.random.randint(0, 3)
            self.r = np.zeros(3)
            new_history = {'p': 1.,
                           'x_hat': [x_hat_straight, x_hat_left, x_hat_right],
                           'u': self.initial_u,
                           'r': np.array([0, 0, 0]),
                           'P': [P_straight, P_left, P_right], }
        else:
            raise ValueError('Undefined scenario.')
        self.true_state = np.array([0, -10, 0, 5])
        self.z = self.H.dot(self.true_state) + np.random.normal(0, 1, 4) * np.sqrt(np.diag(self.R))
        self.u = self.initial_u
        self.previous_u = self.initial_u
        self.power_line = None

        self.positions_x = np.zeros(5000)
        self.positions_y = np.zeros(5000)
        self.estimated_positions_x = np.zeros(5000)
        self.estimated_positions_y = np.zeros(5000)
        self.p1s = list()
        self.p2s = list()
        self.t = 0
        self.histories = list()
        self.histories.append(new_history)

        return self.obs

    def resample(self):
        resampled_histories = list()
        pmf = np.array([history['p'] for history in self.histories])
        cmf = np.cumsum(pmf)
        sampled_histories = list()
        p = 1 / self.n_histories_resampling
        for sample_index in range(self.n_histories_resampling):
            seed = np.random.rand()
            index = int(np.sum(seed > cmf))
            while index >= len(self.histories):
                seed = np.random.rand()
                index = int(np.sum(seed > cmf))
            if index in sampled_histories:
                i = np.where(index == np.array(sampled_histories))[0][0]
                resampled_histories[i]['p'] += p
            else:
                sampled_histories.append(index)
                s = self.histories[index]
                s['p'] = p
                resampled_histories.append(s)

        return resampled_histories

    def render(self, mode='human'):
        if not no_rendering:
            sleep(0.1)
            screen_width = 700
            screen_height = 700
            if self.viewer is None:
                self.viewer = rendering.Viewer(screen_width, screen_height)

            # Estimation standard deviation
            normal = np.array([self.x_imm[3], -self.x_imm[2]])
            normal = normal / np.linalg.norm(normal)
            std = np.sqrt(self.P_imm[0, 0] + self.P_imm[1, 1])
            p1 = self.x_imm[:2] + normal / 2 * std * 30
            p2 = self.x_imm[:2] - normal / 2 * std * 30
            # confidence_range = list((p1 * self.scale - self.offset, p2 * self.scale - self.offset))
            self.list_a.append(tuple(p1 * self.scale - self.offset))
            self.list_b.append(tuple(p2 * self.scale - self.offset))

            self.list = self.list_a[-2:] + self.list_b[-2:][::-1]

            confidence_range = rendering.make_polygon(self.list, True)
            confidence_range.set_color(0.7, 0.7, 1)
            self.viewer.add_geom(confidence_range)

            # True track
            track = list(zip(self.positions_x[: self.t] * self.scale - self.offset_x,
                             self.positions_y[: self.t] * self.scale - self.offset_y))
            track = rendering.make_polyline(track)
            track.set_linewidth(2)
            self.viewer.add_geom(track)

            # Estimated track
            estimated_track = list(zip(self.estimated_positions_x[self.t - 2: self.t] * self.scale - self.offset_x,
                                       self.estimated_positions_y[self.t - 2: self.t] * self.scale - self.offset_y))
            estimated_track = rendering.make_polyline(estimated_track)
            estimated_track.set_linewidth(2)
            estimated_track.set_color(0, 0, 1)
            self.viewer.add_geom(estimated_track)

            # Power allocation
            if self.power_line is not None:
                self.viewer.add_geom(self.power_line)
            p1 = self.x_imm[:2] + normal / 2 * self.tx_power
            p2 = self.x_imm[:2] - normal / 2 * self.tx_power
            self.p1s.append(self.x_imm[:2] + normal * 1 * self.tx_power)
            self.p2s.append(self.x_imm[:2] - normal * 1 * self.tx_power)
            tx_power = list((p1 * self.scale - self.offset, p2 * self.scale - self.offset))
            self.power_line = rendering.make_polyline(tx_power)
            self.power_line.set_linewidth(3)
            self.power_line.set_color(1, 0, 0)
            self.viewer.add_geom(self.power_line)

            # if self.done:
            #     input()

            return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def draw(self):
        if self.done:
            polygon_x = (np.array([p[0] for p in (self.list_a + self.list_b[::-1])]) + self.offset[0]) / self.scale
            polygon_y = (np.array([p[1] for p in (self.list_a + self.list_b[::-1])]) + self.offset[1]) / self.scale
            std = plt.fill(polygon_x, polygon_y, color='#ff7f0e', alpha=0.2, linewidth=0, label="30 x std")
            true_track = plt.plot(self.positions_x[:self.t], self.positions_y[:self.t], label="True track")
            estimated_track = plt.plot(self.estimated_positions_x[:self.t],
                                       self.estimated_positions_y[:self.t], label="Estimated track")

            for p1, p2 in zip(self.p1s, self.p2s):
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=2)

            red_patch = mpatches.mlines.Line2D([], [], color='red', label='Transmission power', linewidth=2)
            plt.legend(handles=[true_track[0], estimated_track[0], red_patch, std[0]])
            plt.axis("equal")
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.show()
