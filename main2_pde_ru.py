import warnings
warnings.filterwarnings("ignore")

import numpy as np
import optimal_control as oc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


if __name__ == '__main__':
    figsize = (15.0, 7.5)
    fignum = 5
    figtitles = 'Оптимальное управление', 'Оптимальная фазовая траектория',\
                'Оптимальная сопряженная переменная', 'Градиент оптимального управления',\
                'Значения целевого функционала'

    space_max, time_max = 1.0, 1.0
    grid_interval_num_space, grid_interval_num_time = 100, 100
    grid_dot_num_space, grid_dot_num_time = grid_interval_num_space+1, grid_interval_num_time+1
    s, t = np.linspace(0, space_max, grid_dot_num_space), np.linspace(0, time_max, grid_dot_num_time)
    tt, ss = np.meshgrid(t, s)
    initial_control = np.sin(t) #np.zeros_like(t) #+ np.random.random(t.shape) * 2 - 1.0
    terminal_state = np.sin(2 * s)
    nu = 2.0
    calc_error_arg, calc_error_func, calc_error_grad = 1e-8, 1e-8, 1e-8
    result_keys = ('control', 'state', 'costate', 'gradient', 'functional')
    mode = 'projection'
    iter_num = 0
    results = oc.solve_optimal_control_problem_parabolic_de(
        initial_control, nu, grid_interval_num_space, grid_interval_num_time, space_max, time_max, terminal_state,
        calc_error_arg, calc_error_func, calc_error_grad, mode=mode, result_keys=result_keys, default_step=1.0
    )

    for key in results.keys():
        print(key)
        if key == result_keys[4]:
            print(results[key])
            iter_num = len(results[key])
        else:
            print(results[key][-1])

    print('Количество итераций вычислительного процесса: %d' % iter_num)
    axes = [None, None]
    for figi in range(fignum):
        fig = plt.figure(figi+1, figsize)
        plt.title(figtitles[figi])
        if figi == 0:
            plt.plot(t, results[result_keys[figi]][-1], 'k-',
                     label='u(t)')
            plt.legend()
            plt.xlabel('t')
            plt.ylabel('u')
            plt.grid(True)
        elif figi == 1:
            ax1 = fig.add_subplot(111, projection='3d')

            ax1.plot_wireframe(tt, ss, results[result_keys[figi]][-1])
            ax1.set_xlim(0, time_max)
            ax1.set_ylim(0, space_max)
            ax1.set_xlabel('t')
            ax1.set_ylabel('s')
            ax1.set_zlabel('x')

        elif figi == 2:
            ax2 = fig.add_subplot(111, projection='3d')

            ax2.plot_wireframe(tt, ss, results[result_keys[figi]][-1])
            ax2.set_xlim(0, time_max)
            ax2.set_ylim(0, space_max)
            ax2.set_xlabel('t')
            ax2.set_ylabel('s')
            ax2.set_zlabel('\u03C8')

        elif figi == 3:
            plt.plot(t, results[result_keys[figi]][-1], 'k-', label='J\'(u)')
            plt.xlabel('t')
            plt.ylabel('J\'')
            plt.legend()
            plt.grid(True)
        else:
            plt.plot(np.arange(results[result_keys[figi]].size), results[result_keys[figi]], 'k-',
                     label='J(u)')
            plt.xlabel('k, индекс итерации')
            plt.ylabel('J')
            plt.legend()
            plt.grid(True)

    plt.figure(fignum+1, figsize=figsize)
    plt.plot(s, results['state'][0][:, -1], 'b-', label=r'$x(s,T,u^{(0)})$')
    plt.plot(s, results['state'][-1][:, -1], 'r-', label=r'$x(s,T,u^{*})$')
    plt.plot(s, terminal_state, 'k-', label='y(s)')
    plt.xlabel('s')
    plt.ylabel('x(s,T)')
    plt.legend()
    plt.grid(True)

    plt.show()
