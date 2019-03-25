import numpy as np
import RungeKutta4 as rk4
from scipy.integrate import quad as calc_integral
from scipy.interpolate import interp1d as interp
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as pool
from tqdm import tqdm


def norm_of_vector_function(func, left_bound, right_bound, integral_error=1.49e-08):
    return np.sqrt(calc_integral(lambda t: (np.array(func(t))**2).sum(), left_bound, right_bound,
                                 epsabs=integral_error))[0]


def solve_optimal_control_problem(Lagrangian, endpoint_cost, initial_time, terminal_time, right_side_of_state_equation,
                                  initial_state, initial_control, deriv_Lagrangian_state, deriv_Lagrangian_control,
                                  deriv_endpoint_cost_state, deriv_right_side_of_state_equation_state,
                                  deriv_right_side_of_state_equation_control, control_lower_bound=None,
                                  control_upper_bound=None, calc_error_arg=1e-4, calc_error_func=1e-4,
                                  calc_error_grad=1e-4, iter_lim=1000, mode='projection', interpolation_kind='cubic',
                                  integral_error=1.49e-08, ode_error=1e-4, default_step=0.01,
                                  result_keys=('control', 'state', 'costate', 'gradient', 'functional')):

    if mode != 'projection' and mode != 'conditional':
        raise ValueError('Invalid key for "mode"!')

    is_projection = True if mode == 'projection' else False

    target_functional = lambda state, control, time_t0, time_T:\
        calc_integral(lambda t: Lagrangian(state(t), control(t), t), time_t0, time_T, epsabs=integral_error)[0] +\
        endpoint_cost(state(time_T))

    control_results, state_results, costate_results, grads_results, functional_results = [], [], [], [], []

    control_current, control_next = np.random.random(initial_control.shape), initial_control.copy()
    control_results.append(control_next.copy())
    time_grid = np.linspace(initial_time, terminal_time, initial_control.shape[1])
    functional_current, functional_next = float(np.random.rand()), float(np.random.rand())
    grad_current, grad_next = np.zeros_like(initial_control), np.zeros_like(initial_control)

    if control_lower_bound is None:
        control_lower_bound_grid = -np.ones_like(initial_control) * np.infty
    else:
        bound_pool = pool(np.minimum(time_grid.size, cpu_count()))
        control_lower_bound_grid = bound_pool.map(control_lower_bound, time_grid)
        control_lower_bound_grid = np.array(control_lower_bound_grid).T
    if control_upper_bound is None:
        control_upper_bound_grid = np.ones_like(initial_control) * np.infty
    else:
        bound_pool = pool(np.minimum(time_grid.size, cpu_count()))
        control_upper_bound_grid = bound_pool.map(control_upper_bound, time_grid)
        control_upper_bound_grid = np.array(control_upper_bound_grid).T

    iterations = tqdm(range(iter_lim))

    for i in iterations:

        control_next_interpolated = interp(time_grid, control_next, kind=interpolation_kind)

        time_for_state_next, state_next = \
            rk4.solve_ode(initial_time, initial_state, initial_time, terminal_time,
                          lambda t, x: right_side_of_state_equation(x, control_next_interpolated(t), t), ode_error)
        state_next_interpolated = interp(time_for_state_next, state_next, kind=interpolation_kind) \
            if time_for_state_next.size > 2 else interp(time_for_state_next, state_next)
        state_results.append(state_next_interpolated(time_grid))

        time_for_costate_next, costate_next = \
            rk4.solve_ode(terminal_time, -deriv_endpoint_cost_state(state_next_interpolated(terminal_time)),
                          initial_time, terminal_time,
                          lambda t, psi: deriv_Lagrangian_state(
                              state_next_interpolated(t),
                              control_next_interpolated(t),
                              t
                          ) - np.dot(psi, deriv_right_side_of_state_equation_state(
                              state_next_interpolated(t),
                              control_next_interpolated(t),
                              t
                          )),
                          ode_error)
        costate_next_interpolated = interp(time_for_costate_next, costate_next, kind=interpolation_kind) \
            if time_for_costate_next.size > 2 else interp(time_for_costate_next, costate_next)
        costate_results.append(costate_next_interpolated(time_grid))

        grad_pool = pool(np.minimum(time_grid.size, cpu_count()))
        grad_current = grad_next.copy()
        grad_next = grad_pool.map(lambda t:
                                  deriv_Lagrangian_control(state_next_interpolated(t), control_next_interpolated(t), t)
                                  - np.dot(costate_next_interpolated(t),
                                           deriv_right_side_of_state_equation_control(state_next_interpolated(t),
                                                                                      control_next_interpolated(t),
                                                                                      t)), time_grid)
        grad_next = np.array(grad_next).T
        grad_pool.close()
        grad_pool.join()
        grads_results.append(grad_next)

        functional_current = functional_next
        functional_next = target_functional(state_next_interpolated, control_next_interpolated, initial_time,
                                            terminal_time)
        functional_results.append(functional_next)

        if np.linalg.norm(control_next - control_current) < calc_error_arg or \
                np.linalg.norm(grad_next) < calc_error_grad or \
                np.abs(functional_next - functional_current) < calc_error_func:
            iterations.close()
            break

        step = default_step \
            if i == 0 \
            else np.abs(np.dot((control_next - control_current).ravel(), (grad_next - grad_current).ravel())) / \
                 np.linalg.norm(grad_next - grad_current) ** 2

        if is_projection:
            control_next_intermediate = control_next - step * grad_next
            control_next_intermediate = np.where(control_next_intermediate > control_upper_bound_grid,
                                                 control_upper_bound_grid, control_next_intermediate)
            control_next_intermediate = np.where(control_next_intermediate < control_lower_bound_grid,
                                                 control_lower_bound_grid, control_next_intermediate)
            control_current = control_next.copy()
            control_next = control_next_intermediate.copy()
            control_results.append(control_next.copy())
        else:
            step = step if step >= 0.0 else 0.0
            step = step if step <= 1.0 else 1.0

            control_next_intermediate = np.where(grad_next >= 0, control_lower_bound_grid, control_upper_bound_grid)
            control_current = control_next.copy()
            control_next = control_next + step * (control_next_intermediate - control_next)
            control_results.append(control_next.copy())

    return {result_keys[0]: control_results, result_keys[1]: state_results, result_keys[2]: costate_results,
            result_keys[3]: grads_results, result_keys[4]: np.array(functional_results)}
