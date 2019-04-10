import numpy as np
import optimal_control as oc
import RungeKutta4 as rk4
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d as interp
from scipy.constants import gravitational_constant

number_of_dimension = int(3)


def other_inds(index, index_min=0, index_max=3):
    if index < index_min or index >= index_max:
        raise ValueError('index value is out of range!')
    return np.array(np.concatenate((np.arange(index_min, index), np.arange(index+1, index_max))), dtype=np.int32)


def vector_to_momenta_and_coordinates(vector):

    if len(vector.shape) != 1 or vector.size % (2 * number_of_dimension) != 0:
        raise ValueError('"vector" must have type "ndarray" with shape=(%d * N,), where N is a positive integer!' %
                         (2 * number_of_dimension))

    body_number = vector.size // (2 * number_of_dimension)
    return (vector[:body_number * number_of_dimension].reshape(body_number, number_of_dimension),
            vector[body_number * number_of_dimension:].reshape(body_number, number_of_dimension))


def momenta_and_coordinates_to_vector(momenta, coordinates):

    if len(momenta.shape) != 2 or len(coordinates.shape) != 2 or momenta.shape[1] != number_of_dimension or \
            coordinates.shape[1] != number_of_dimension or momenta.shape[0] != coordinates.shape[0]:
        raise ValueError('"momenta" and "coordinates" must have type "ndarray" with shape=(N, %d),'
                         'where N is a positive integer!' % number_of_dimension)

    return np.concatenate((momenta.ravel(), coordinates.ravel()))


def Lagrangian(x, m, t, t_given, x_given, interpolation_kind):
    x_given_interpolated = interp(t_given, x_given, kind=interpolation_kind) \
            if t_given.size > 2 else interp(t_given, x_given)
    return np.linalg.norm(x - x_given_interpolated(t).reshape(x_given_interpolated(t).size, -1)) ** 2


def endpoint_cost(x):
    return 0.0


def right_side_of_state_equation(x, m, t):

    right_side = np.zeros_like(x)
    body_num = right_side.size // (2 * number_of_dimension)

    for k in range(right_side.size):

        if k < number_of_dimension * body_num:

            sum_inds = np.concatenate((np.arange(np.floor(k / number_of_dimension)),
                                       np.arange(np.floor(k / number_of_dimension)+1, body_num)))
            sum_inds = np.array(sum_inds, dtype=np.int32)
            right_side[k] = -gravitational_constant * m[int(np.floor(k / number_of_dimension))] * np.sum(
                m[sum_inds] * (x[number_of_dimension * body_num + k] -
                               x[number_of_dimension * body_num + k % number_of_dimension +
                                 number_of_dimension * sum_inds]) /
                np.sqrt((x[number_of_dimension * body_num + number_of_dimension *
                           int(np.floor(k / number_of_dimension))] -
                         x[number_of_dimension * body_num + number_of_dimension * sum_inds])**2 +
                        (x[number_of_dimension * body_num +
                           number_of_dimension * int(np.floor(k / number_of_dimension)) + 1] -
                         x[number_of_dimension * body_num + 1 + number_of_dimension * sum_inds])**2 +
                        (x[number_of_dimension * body_num +
                           number_of_dimension * int(np.floor(k / number_of_dimension)) + 2] -
                         x[number_of_dimension * body_num + 2 + number_of_dimension * sum_inds])**2) ** 3)

        else:
            right_side[k] = x[k - number_of_dimension * body_num] / \
                            m[int(np.floor((k - number_of_dimension * body_num) / number_of_dimension))]

    return right_side


def deriv_Lagrangian_state(x, m, t, t_given, x_given, interpolation_kind):
    x_given_interpolated = interp(t_given, x_given, kind=interpolation_kind) \
        if t_given.size > 3 else interp(t_given, x_given)
    return 2 * (x - x_given_interpolated(t))


def deriv_Lagrangian_control(x, m, t):
    return np.zeros_like(m)


def deriv_endpoint_cost_state(x):
    return np.zeros_like(x)


def deriv_right_side_of_state_equation_state(x, m, t, print_iter=False):

    body_num = x.size // (2 * number_of_dimension)
    deriv = np.zeros((x.size, x.size))

    for i in range(body_num):

        all_inds = np.arange(body_num)
        i_other_inds = other_inds(i, 0, body_num)
        distances_from_i = np.sqrt((x[number_of_dimension * body_num + number_of_dimension * i] -
                                    x[number_of_dimension * body_num + number_of_dimension * all_inds]) ** 2 +
                                   (x[number_of_dimension * body_num + number_of_dimension * i + 1] -
                                    x[number_of_dimension * body_num + number_of_dimension * all_inds + 1]) ** 2 +
                                   (x[number_of_dimension * body_num + number_of_dimension * i + 2] -
                                    x[number_of_dimension * body_num + number_of_dimension * all_inds + 2]) ** 2
                                   ) ** 5
        deriv[number_of_dimension * body_num + i, i] = 1 / m[int(np.floor(i / number_of_dimension))]
        for j in range(body_num):
            for p in range(number_of_dimension):
                for s in range(number_of_dimension):
                    if p == s:

                        p_other_inds = other_inds(p, 0, number_of_dimension)
                        p_other0, p_other1 = p_other_inds[0], p_other_inds[1]
                        if i == j:
                            deriv[p + number_of_dimension * i,
                                  number_of_dimension * body_num + s + number_of_dimension * j] = \
                                -gravitational_constant * m[i] * np.sum(
                                    m[i_other_inds] / distances_from_i[i_other_inds] *
                                    (-2*(x[number_of_dimension * body_num + p + number_of_dimension * i] -
                                         x[number_of_dimension * body_num + p + number_of_dimension *
                                           i_other_inds])**2 +
                                     (x[number_of_dimension * body_num + p_other0 + number_of_dimension * i] -
                                      x[number_of_dimension * body_num + p_other0 + number_of_dimension *
                                        i_other_inds])**2 +
                                     (x[number_of_dimension * body_num + p_other1 + number_of_dimension * i] -
                                      x[number_of_dimension * body_num + p_other1 + number_of_dimension *
                                        i_other_inds])**2)
                            )
                        else:
                            deriv[p + number_of_dimension * i,
                                  number_of_dimension * body_num + s + number_of_dimension * j] = \
                                gravitational_constant * m[i] * m[j] / distances_from_i[j] * \
                                (-2 * (x[number_of_dimension * body_num + p + number_of_dimension * i] -
                                       x[number_of_dimension * body_num + p + number_of_dimension * j]) ** 2 +
                                 (x[number_of_dimension * body_num + p_other0 + number_of_dimension * i] -
                                  x[number_of_dimension * body_num + p_other0 + number_of_dimension * j]) ** 2 +
                                 (x[number_of_dimension * body_num + p_other1 + number_of_dimension * i] -
                                  x[number_of_dimension * body_num + p_other1 + number_of_dimension * j]) ** 2)
                    else:

                        if i == j:
                            deriv[p + number_of_dimension * i,
                                  number_of_dimension * body_num + s + number_of_dimension * j] = \
                                3 * gravitational_constant * m[i] * np.sum(
                                    m[i_other_inds] / distances_from_i[i_other_inds] *
                                    (x[number_of_dimension * body_num + p + number_of_dimension * i] -
                                     x[number_of_dimension * body_num + p + number_of_dimension * i_other_inds]) *
                                    (x[number_of_dimension * body_num + s + number_of_dimension * i] -
                                     x[number_of_dimension * body_num + s + number_of_dimension * i_other_inds])
                                )
                        else:
                            deriv[p + number_of_dimension * i,
                                  number_of_dimension * body_num + s + number_of_dimension * j] = \
                                -3 * gravitational_constant * m[i] * m[j] / distances_from_i[j] * \
                                (x[number_of_dimension * body_num + p + number_of_dimension * i] -
                                 x[number_of_dimension * body_num + p + number_of_dimension * j]) * \
                                (x[number_of_dimension * body_num + s + number_of_dimension * i] -
                                 x[number_of_dimension * body_num + s + number_of_dimension * j])

    if print_iter:
        print('Done!')
    return deriv


def deriv_right_side_of_state_equation_control(x, m, t):

    body_num = x.size // (2 * number_of_dimension)
    deriv = np.zeros((x.size, m.size))

    for i in range(body_num):
        all_inds = np.arange(body_num)
        i_other_inds = other_inds(i, 0, body_num)
        distances_from_i = np.sqrt((x[number_of_dimension * body_num + number_of_dimension * i] -
                                    x[number_of_dimension * body_num + number_of_dimension * all_inds]) ** 2 +
                                   (x[number_of_dimension * body_num + number_of_dimension * i + 1] -
                                    x[number_of_dimension * body_num + number_of_dimension * all_inds + 1]) ** 2 +
                                   (x[number_of_dimension * body_num + number_of_dimension * i + 2] -
                                    x[number_of_dimension * body_num + number_of_dimension * all_inds + 2]) ** 2
                                   ) ** 3
        for j in range(body_num):
            for p in range(number_of_dimension):

                if i == j:
                    deriv[p + number_of_dimension * i, j] = \
                        -gravitational_constant * np.sum(
                            m[i_other_inds] / distances_from_i[i_other_inds] *
                            (x[number_of_dimension * body_num + p + number_of_dimension * i] -
                             x[number_of_dimension * body_num + p + number_of_dimension * i_other_inds])
                        )
                    deriv[number_of_dimension * body_num + p + number_of_dimension * i, j] = \
                        -x[p + number_of_dimension * i] / m[i] / m[i]

                else:
                    deriv[p + number_of_dimension * i, j] = \
                        -gravitational_constant * m[i] / distances_from_i[j] * \
                        (x[number_of_dimension * body_num + p + number_of_dimension * i] -
                         x[number_of_dimension * body_num + p + number_of_dimension * j])

    return deriv


def control_lower_bound(t, m):
    return np.zeros_like(m)


if __name__ == '__main__':

    body_number = 4
    grid_dot_num = 101
    control_dim, state_dim = body_number, 2 * number_of_dimension * body_number
    initial_control = np.zeros((control_dim, grid_dot_num)) + 10**3
    default_step = 0.01
    mode = 'projection'
    figsize = (15.0, 7.5)
    fignum = 4
    line_styles = ('k-',)
    figtitles = 'Оптимальное управление', 'Оптимальная фазовая траектория',\
                'Оптимальные сопряженные переменные', 'Значения целевого функционала'
    result_keys = ('control', 'state', 'costate', 'gradient', 'functional')
    iter_num = 0
    var_symbols = 'u', 'x', '\u03C8', 'J'

    masses = np.array([2*100**3, 15**3, 20**3, 8**3])
    momenta_initial, coordinates_initial = \
        np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 10.0],
            [10.0, 10.0, 0.0],
            [1.0, 1.0, -1.0]
        ]), \
        np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 4.0, 0.11]
        ])
    initial_state = momenta_and_coordinates_to_vector(momenta_initial, coordinates_initial)
    t0, T = 0.0, 60.0 * 60
    t = np.linspace(t0, T, grid_dot_num)
    calc_eps, h_initial = 0.0001, 1.0

    interpolation_kind = 'cubic'
    print_iter = True

    t_given, x_given = rk4.solve_ode(t0, momenta_and_coordinates_to_vector(momenta_initial, coordinates_initial), t0, T,
                                     lambda t, x: right_side_of_state_equation(x, masses, t), calc_eps=calc_eps,
                                     h_initial=h_initial)

    results = oc.solve_optimal_control_problem(lambda x, m, t:
                                               Lagrangian(x, m, t, t_given, x_given, interpolation_kind),
                                               endpoint_cost, t0, T,
                                               right_side_of_state_equation, initial_state, initial_control,
                                               lambda x, m, t:
                                               deriv_Lagrangian_state(x, m, t, t_given, x_given, interpolation_kind),
                                               deriv_Lagrangian_control, deriv_endpoint_cost_state,
                                               deriv_right_side_of_state_equation_state,
                                               deriv_right_side_of_state_equation_control,
                                               calc_error_arg=1e-2, calc_error_func=1e-2,
                                               calc_error_grad=1e-2, iter_lim=5,
                                               control_lower_bound=lambda t: control_lower_bound(t, masses),
                                               control_upper_bound=None, default_step=default_step,
                                               mode=mode, ode_error=calc_eps, result_keys=result_keys,
                                               print_iter=print_iter)
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
        if figi == 0 or figi == 1 or figi == 2:
            for i in range(control_dim):
                plt.plot(t, results[result_keys[figi]][-1][i], line_styles[i],
                         label='%s%s(t)' % (var_symbols[figi], '' if control_dim == 1 else str(i)))
            plt.legend()
            plt.xlabel('t')
            plt.ylabel(var_symbols[figi])
        elif figi == 3:
            plt.plot(np.arange(results[result_keys[figi+1]].size), results[result_keys[figi+1]], line_styles[0],
                     label='%s(u)' % (var_symbols[figi]))
            plt.xlabel('k, индекс итерации')
            plt.ylabel(var_symbols[figi])

    plt.show()
    plt.close()
