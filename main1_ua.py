import numpy as np
import optimal_control as oc
import matplotlib.pyplot as plt


def Lagrangian(x, u, t):
    return np.sin(t) + x[0] * u[0] ** 2


def endpoint_cost(x):
    return x[0]


def right_side_of_state_equation(x, u, t):
    return np.array([x[0] + (t + 1) * u[0] + np.exp(t)])


def deriv_Lagrangian_state(x, u, t):
    return np.array([u[0]**2])


def deriv_Lagrangian_control(x, u, t):
    return np.array([2.0 * x[0] * u[0]])


def deriv_endpoint_cost_state(x):
    return np.array([1.0])


def deriv_right_side_of_state_equation_state(x, u, t):
    return np.array([
        [1.0]
    ])


def deriv_right_side_of_state_equation_control(x, u, t):
    return np.array([
        [t + 1.0]
    ])


def control_lower_bound(t):
    return np.array([1.0])


def control_upper_bound(t):
    return np.array([2.0])


if __name__ == '__main__':

    grid_dot_num = 101
    control_dim, state_dim = 1, 1
    initial_time, terminal_time = 0.0, 2.0
    initial_state = np.array([1.0])
    t = np.linspace(initial_time, terminal_time, grid_dot_num)
    initial_control = np.zeros((control_dim, grid_dot_num)) + 1.5
    default_step = 0.01
    mode = 'conditional'
    figsize = (15.0, 7.5)
    fignum = 4
    line_styles = tuple('k-' for i in range(np.maximum(control_dim, state_dim)))
    figtitles = '', '',\
                '', ''
    result_keys = ('control', 'state', 'costate', 'gradient', 'functional')
    iter_num = 0
    var_symbols = 'u', 'x', '\u03C8', 'J'

    results = oc.solve_optimal_control_problem(Lagrangian, endpoint_cost, initial_time, terminal_time,
                                               right_side_of_state_equation, initial_state, initial_control,
                                               deriv_Lagrangian_state, deriv_Lagrangian_control,
                                               deriv_endpoint_cost_state, deriv_right_side_of_state_equation_state,
                                               deriv_right_side_of_state_equation_control,
                                               calc_error_arg=1e-2, calc_error_func=1e-2,
                                               calc_error_grad=1e-2,
                                               control_lower_bound=control_lower_bound,
                                               control_upper_bound=control_upper_bound, default_step=default_step,
                                               mode=mode, result_keys=result_keys)
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
                         label='%s%s(t)' % (var_symbols[figi], '' if control_dim == 1 else str(i)))
            plt.legend()
            plt.xlabel('t')
            plt.ylabel(var_symbols[figi])
        elif figi == 1 or figi == 2:
            for i in range(state_dim):
                plt.plot(t, results[result_keys[figi]][-1][i], line_styles[i],
                         label='%s%s(t)' % (var_symbols[figi], '' if state_dim == 1 else str(i)))
            plt.legend()
            plt.xlabel('t')
            plt.ylabel(var_symbols[figi])
        elif figi == 3:
            plt.plot(np.arange(results[result_keys[figi+1]].size), results[result_keys[figi+1]], line_styles[0],
                     label='%s(u)' % (var_symbols[figi]))
            plt.xlabel('k, індекс ітерації')
            plt.ylabel(var_symbols[figi])

    plt.show()
    plt.close()
