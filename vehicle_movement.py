import numpy as np

delta_t = 0.1
delta_theta = 0.1
delta_theta_lane_changing = 0.05
t = np.arange(0, 7, delta_t)
n_timesteps = len(t)
sz = (4, n_timesteps)  # size of array
true_state = np.array([0, -10, 0, 5])   # Px, Py, Vx, Vy
H = np.eye(4)
R = np.eye(4) * 0.001  # Measurement noise
Q = np.eye(4) * 0.0005  # Process noise
z = H.dot(true_state) + np.random.randn(4) * np.sqrt(np.diag(R))  # + v_t(observation noise)
K = np.zeros(4)
I = np.eye(4)

all_true_states = np.empty(sz)
all_estimated_states_straight = np.empty(sz)
all_estimated_states_turning = np.empty(sz)
all_estimated_states_left = np.empty(sz)
all_estimated_states_right = np.empty(sz)
all_observed_states = np.empty(sz)
all_imm_states = np.empty(sz)
collect_data = np.empty([n_timesteps, 14])

timestep = 0
model_weights = np.ones(2) / 2
P_straight = np.eye(4) * 0.1
P_turn = np.eye(4) * 0.1
P_left = np.eye(4) * 0.1
P_right = np.eye(4) * 0.1
x_hat_straight = true_state + np.random.randn(4) * np.sqrt(np.diag(P_straight))
x_hat_left = true_state + np.random.randn(4) * np.sqrt(np.diag(P_straight))
x_hat_right = true_state + np.random.randn(4) * np.sqrt(np.diag(P_straight))
x_hat_turn = x_hat_straight
x_hat_minus_straight = np.zeros(4)
x_hat_minus_turn = np.zeros(4)
P_minus_straight = np.zeros((4, 4))
P_minus_turn = np.zeros((4, 4))
pi_intersection = np.eye(2)
pi_lane_changing = np.array([[0.8, 0.1, 0.1],
                             [0.2, 0.8, 0],
                             [0, 0.2, 0.8]])

u0 = 0.5
u1 = 0.5
u_intersection = np.array([0.5, 0.5])
u_lane_changing = np.array([0.99, 0.005, 0.005])
r = 640  # date rate
W = 8500  # bandwidth
P_t = 0.01  # transmission power
N_0 = 1
R_t = 0.5  # A random number to compare with the p_t_c_out
p_t_c_out = 0.3  # The probability of communication outage


def update_state_straight(x_hat_straight, P_straight, z=None):

        f = np.array([[1, 0, delta_t, 0],
                      [0, 1, 0, delta_t],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        # Prediction
        x_hat_minus_straight = f.dot(x_hat_straight)
        P_minus_straight = f.dot(P_straight).dot(f.T) + Q  # Predicted (a priori) estimate

        if z is None:
            return x_hat_minus_straight, P_minus_straight, None
        else:
            # Combine prediction and measurement
            K = P_minus_straight.dot(H.T).dot(np.linalg.inv(H.dot(P_minus_straight).dot(H.T) + R))  # Optimal Kalman gain
            x_hat_straight = x_hat_minus_straight + K.dot((z - H.dot(x_hat_minus_straight)))  # Updated (a posteriori) state estimate
            P_straight = (I - K.dot(H)).dot(P_minus_straight)  # Updated (a posteriori) estimate covariance
            r_0 = z - H.dot(x_hat_minus_straight)  # measurement residual

            return x_hat_straight, P_straight, r_0


def update_state_turning(x_hat_turn, P_turn, z=None):

    if -5 <= x_hat_turn[0] <= 5 and 0 <= x_hat_turn[1] <= 10:

        f = np.array([[1, 0, delta_t, 0],
                      [0, 1, 0, delta_t],
                      [0, 0, np.cos(delta_theta), np.sin(delta_theta)],
                      [0, 0, -np.sin(delta_theta), np.cos(delta_theta)]])
    else:

        f = np.array([[1, 0, delta_t, 0],
                      [0, 1, 0, delta_t],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    # Prediction
    x_hat_minus_turn = f.dot(x_hat_turn)
    P_minus_turn = f.dot(P_turn).dot(f.T) + Q  # Predicted (a priori) estimate

    if z is None:
        return x_hat_minus_turn, P_minus_turn, None
    else:
        # Combine prediction and measurement
        K = P_minus_turn.dot(H.T).dot(np.linalg.inv(H.dot(P_minus_turn).dot(H.T) + R))  # Optimal Kalman gain
        x_hat_turn = x_hat_minus_turn + K.dot((z - H.dot(x_hat_minus_turn)))  # Updated (a posteriori) state estimate
        P_turn = (I - K.dot(H)).dot(P_minus_turn)  # Updated (a posteriori) estimate covariance
        r_1 = z - H.dot(x_hat_minus_turn)  # measurement residual

        return x_hat_turn, P_turn, r_1


def update_state_changing_left(x_hat, P, z=None):
    f = np.array([[1, 0, delta_t, 0],
                  [0, 1, 0, delta_t],
                  [0, 0, np.cos(delta_theta_lane_changing), -np.sin(delta_theta_lane_changing)],
                  [0, 0, np.sin(delta_theta_lane_changing), np.cos(delta_theta_lane_changing)]])
    # Prediction
    x_hat_minus = f.dot(x_hat)
    P_minus = f.dot(P).dot(f.T) + Q  # Predicted (a priori) estimate

    if z is None:
        return x_hat_minus, P_minus, None
    else:
        # Combine prediction and measurement
        K = P_minus.dot(H.T).dot(np.linalg.inv(H.dot(P_minus).dot(H.T) + R))  # Optimal Kalman gain
        x_hat = x_hat_minus + K.dot((z - H.dot(x_hat_minus)))  # Updated (a posteriori) state estimate
        P = (I - K.dot(H)).dot(P_minus)  # Updated (a posteriori) estimate covariance
        r = z - H.dot(x_hat_minus)  # measurement residual

        return x_hat, P, r


def update_state_changing_right(x_hat, P, z=None):
    f = np.array([[1, 0, delta_t, 0],
                  [0, 1, 0, delta_t],
                  [0, 0, np.cos(delta_theta_lane_changing), np.sin(delta_theta_lane_changing)],
                  [0, 0, -np.sin(delta_theta_lane_changing), np.cos(delta_theta_lane_changing)]])

    # Prediction
    x_hat_minus = f.dot(x_hat)
    P_minus = f.dot(P).dot(f.T) + Q  # Predicted (a priori) estimate

    if z is None:
        return x_hat_minus, P_minus, None
    else:
        # Combine prediction and measurement
        K = P_minus.dot(H.T).dot(np.linalg.inv(H.dot(P_minus).dot(H.T) + R))  # Optimal Kalman gain
        x_hat = x_hat_minus + K.dot((z - H.dot(x_hat_minus)))  # Updated (a posteriori) state estimate
        P = (I - K.dot(H)).dot(P_minus)  # Updated (a posteriori) estimate covariance
        r = z - H.dot(x_hat_minus)  # measurement residual

        return x_hat, P, r


def turn_velocity(vx, vy, delta_theta):
    v = np.sqrt(vx ** 2 + vy ** 2)
    theta = np.arctan2(vy, vx)
    if theta > 0:
        theta += delta_theta

    vx = v * np.cos(theta)
    vy = v * np.sin(theta)
    return vx, vy


def update_true_state_intersection(true_state, K):
    w = np.random.normal(0, 1, 4) * np.sqrt(np.diag(Q))
    if K == 1:
        f = np.array([[1, 0, delta_t, 0],
                      [0, 1, 0, delta_t],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        true_state = f.dot(true_state) + w
    else:
        if -5 <= true_state[0] <= 5 and 0 <= true_state[1] <= 10:
            vx, vy = true_state[2], true_state[3]
            vx, vy = turn_velocity(vx, vy, -delta_theta)

            true_state = np.array([true_state[0] + delta_t * vx,
                                   true_state[1] + delta_t * vy,
                                   vx,
                                   vy]) + w
        else:
            f = np.array([[1, 0, delta_t, 0],
                          [0, 1, 0, delta_t],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
            true_state = f.dot(true_state) + w

    return true_state


def update_true_state_lane_changing(true_state, K, y_lane_changing):
    w = np.random.normal(0, 1, 4) * np.sqrt(np.diag(Q))

    lane_changing_finished = np.abs(true_state[0]) > 2 and np.abs(true_state[2]) < 0.2

    if K == 0 or true_state[1] < y_lane_changing or lane_changing_finished:
        f = np.array([[1, 0, delta_t, 0],
                      [0, 1, 0, delta_t],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        true_state = f.dot(true_state) + w
        if lane_changing_finished:
            true_state[2] *= 0.9
    else:
        if K == 1:
            vx, vy = true_state[2], true_state[3]
            if true_state[0] > -2:
                vx, vy = turn_velocity(vx, vy, delta_theta_lane_changing)
            else:
                vx, vy = turn_velocity(vx, vy, -delta_theta_lane_changing)
        elif K == 2:
            vx, vy = true_state[2], true_state[3]
            if true_state[0] < 2:
                vx, vy = turn_velocity(vx, vy, -delta_theta_lane_changing)
            else:
                vx, vy = turn_velocity(vx, vy, delta_theta_lane_changing)
        else:
            raise ValueError("Unknown mode.")

        true_state = np.array([true_state[0] + delta_t * vx,
                               true_state[1] + delta_t * vy,
                               vx,
                               vy]) + w
    return true_state


def mix_state_v2(x_hat, pi, u, P):
    # Eq. 41
    u = pi * u[:, np.newaxis] + 1e-30
    c = np.sum(u, axis=0, keepdims=True)
    u = u / c

    # Eq. 42
    x_hat_mix, P_mix = list(), list()
    for model_idx in range(len(x_hat)):
        x_hat_mix.append(sum([uu * xx for uu, xx in zip(u[:, model_idx], x_hat)]))

    for i in range(len(x_hat)):
        p = 0
        for j in range(len(x_hat)):
            p += u[j, i] * (P[j] + (x_hat[j] - x_hat_mix[i]).T.dot(x_hat[j] - x_hat_mix[i]))
        P_mix.append(p)

    return x_hat_mix, P_mix, c[0, :]


def update_model_probabilities_v2(r, P_minus, c):
    if r is not None:
        S = [H.dot(pp).dot(H.T) + R for pp in P_minus]
        Lambda = [1 / np.sqrt(np.linalg.det((2 * np.pi) ** 4 * ss)) *
                  np.exp(-0.5 * rr.T.dot(np.linalg.inv(ss)).dot(rr)) + 1e-30 for ss, rr in zip(S, r)]
        C = np.sum(Lambda * c)
        u = np.array([ll * cc / C for ll, cc in zip(Lambda, c)])
        return u
    else:
        return c


def output_estimate_v2(x_hat, u, P):
    x_imm = sum([xx * uu for xx, uu in zip(x_hat, u)])
    P_imm = sum([uu * (pp + ((xx - x_imm)[:, np.newaxis]).dot((xx - x_imm)[np.newaxis, :]))
                 for uu, pp, xx in zip(u, P, x_hat)])
    return x_imm, P_imm
