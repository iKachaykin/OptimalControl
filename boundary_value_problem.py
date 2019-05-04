import numpy as np


def solve_boundary_value_problem_state(u, nu, grid_interval_num_space, grid_interval_num_time, space_max, time_max):
    x_res = np.zeros((grid_interval_num_space + 1, grid_interval_num_time + 1))
    time_step, space_step = time_max / grid_interval_num_time, space_max / grid_interval_num_space
    time_space_relation = time_step / space_step / space_step
    alpha, beta = np.ones(grid_interval_num_space), np.zeros(grid_interval_num_space + 1)
    for j in range(1, grid_interval_num_time+1):
        alpha[0] = 1.0
        beta[0] = 0.0
        for i in range(grid_interval_num_space - 1):
            alpha[i+1] = time_space_relation / (1 + time_space_relation * (2 - alpha[i]))
            beta[i+1] = (time_space_relation * beta[i] + x_res[i+1, j-1]) / (1 + time_space_relation * (2 - alpha[i]))
        beta[grid_interval_num_space] = (space_step * nu * u[j] + beta[grid_interval_num_space - 1]) / \
                                        (1 + space_step * nu - alpha[grid_interval_num_space - 1])
        x_res[grid_interval_num_space, j] = beta[grid_interval_num_space]
        for i in range(grid_interval_num_space - 1, -1, -1):
            x_res[i, j] = alpha[i] * x_res[i+1, j] + beta[i]
    return x_res


def solve_boundary_value_problem_costate(u, nu, grid_interval_num_space, grid_interval_num_time, space_max, time_max,
                                         costate_terminal):
    psi_res = np.zeros((grid_interval_num_space + 1, grid_interval_num_time + 1))
    psi_res[:, -1] = costate_terminal.copy()
    time_step, space_step = time_max / grid_interval_num_time, space_max / grid_interval_num_space
    time_space_relation = time_step / space_step / space_step
    alpha, beta = np.ones(grid_interval_num_space), np.zeros(grid_interval_num_space + 1)
    for j in range(grid_interval_num_time-1, -1, -1):
        alpha[0] = 1.0
        beta[0] = 0.0
        for i in range(grid_interval_num_space - 1):
            alpha[i+1] = time_space_relation / (1 + time_space_relation * (2 - alpha[i]))
            beta[i+1] = (time_space_relation * beta[i] + psi_res[i+1, j+1]) / (1 + time_space_relation * (2 - alpha[i]))
        beta[grid_interval_num_space] = (beta[grid_interval_num_space - 1]) / \
                                        (1 + space_step * nu - alpha[grid_interval_num_space - 1])
        psi_res[grid_interval_num_space, j] = beta[grid_interval_num_space]
        for i in range(grid_interval_num_space - 1, -1, -1):
            psi_res[i, j] = alpha[i] * psi_res[i+1, j] + beta[i]
    return psi_res
