import numpy as np
import optimal_control as oc
import matplotlib.pyplot as plt


def Lagrangian(x, u, t):
    return np.linalg.norm(x) ** 2 + np.linalg.norm(u) ** 2


def endpoint_cost(x):
    return np.linalg.norm(x) ** 2


def right_side_of_state_equation(x, u, t, A, B):
    return np.dot(A, x) + np.dot(B, u)


def deriv_Lagrangian_state(x, u, t):
    return 2.0 * x


def deriv_Lagrangian_control(x, u, t):
    return 2.0 * u


def deriv_endpoint_cost_state(x):
    return 2.0 * x


def deriv_right_side_of_state_equation_state(x, u, t, A):
    return A


def deriv_right_side_of_state_equation_control(x, u, t, B):
    return B


def control_lower_bound(t, control_dim):
    return -1.0 * np.ones(control_dim)


def control_upper_bound(t, control_dim):
    return 1.0 * np.ones(control_dim)


if __name__ == '__main__':

    grid_dot_num = 1001
    control_dim, state_dim = 3, 2
    rand_low, rand_high = 0, 10
    initial_time, terminal_time = 0.0, 2.0
    initial_state = np.zeros(state_dim)
    t = np.linspace(initial_time, terminal_time, grid_dot_num)

    A = np.ones((state_dim, state_dim))
    B = np.ones((state_dim, control_dim))

    # A = np.random.randint(rand_low, rand_high, (state_dim, state_dim))
    # B = np.random.randint(rand_low, rand_high, (state_dim, control_dim))

    # initial_control = np.ones((control_dim, grid_dot_num)) * 0.05
    initial_control = np.array([np.sin(t), np.cos(t), np.sin(2*t)])

    default_step = 0.01
    mode = 'projection'
    figsize = (15.0, 7.5)
    fignum = 4
    # line_styles = tuple('k-' for i in range(np.maximum(control_dim, state_dim)))
    line_styles = ('k-', 'b-', 'r-')
    figtitles = 'Оптимальное управление', 'Оптимальная фазовая траектория',\
                'Оптимальные сопряженные переменные', 'Значения целевого функционала'
    result_keys = ('control', 'state', 'costate', 'gradient', 'functional')
    iter_num = 0
    var_synmbols = 'u', 'x', '\u03C8', 'J'

    results = oc.solve_optimal_control_problem(Lagrangian, endpoint_cost, initial_time, terminal_time,
                                               lambda x, u, t: right_side_of_state_equation(x, u, t, A, B),
                                               initial_state, initial_control,
                                               deriv_Lagrangian_state, deriv_Lagrangian_control,
                                               deriv_endpoint_cost_state,
                                               lambda x, u, t: deriv_right_side_of_state_equation_state(x, u, t, A),
                                               lambda x, u, t: deriv_right_side_of_state_equation_control(x, u, t, B),
                                               calc_error_arg=1e-10, calc_error_func=1e-10,
                                               calc_error_grad=1e-10,
                                               control_lower_bound=lambda t: control_lower_bound(t, control_dim),
                                               control_upper_bound=lambda t: control_upper_bound(t, control_dim),
                                               ode_error=1e-2, default_step=default_step, mode=mode,
                                               result_keys=result_keys, print_iter=False)
    for key in results.keys():
        print(key)
        if key == result_keys[4]:
            print(results[key])
            iter_num = len(results[key])
        else:
            print(results[key][-1])

    print('Количество итераций вычислительного процесса: %d' % iter_num)

    for figi in range(fignum):
        plt.figure(figi+1, figsize)
        plt.title(figtitles[figi])
        plt.grid(True)
        if figi == 0:
            for i in range(control_dim):
                plt.plot(t, results[result_keys[figi]][-1][i], line_styles[i],
                         label='%s%s(t)' % (var_synmbols[figi], '' if control_dim == 1 else str(i)))
            plt.legend()
            plt.xlabel('t')
            plt.ylabel(var_synmbols[figi])
        elif figi == 1 or figi == 2:
            for i in range(state_dim):
                plt.plot(t, results[result_keys[figi]][-1][i], line_styles[i],
                         label='%s%s(t)' % (var_synmbols[figi], '' if state_dim == 1 else str(i)))
            plt.legend()
            plt.xlabel('t')
            plt.ylabel(var_synmbols[figi])
        elif figi == 3:
            plt.plot(np.arange(results[result_keys[figi+1]].size), results[result_keys[figi+1]], line_styles[0],
                     label='%s(u)' % (var_synmbols[figi]))
            plt.xlabel('k, индекс итерации')
            plt.ylabel(var_synmbols[figi])

    plt.show()
    plt.close()
