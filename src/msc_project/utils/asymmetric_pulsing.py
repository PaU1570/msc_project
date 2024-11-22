import matplotlib.pyplot as plt
import numpy as np
from aihwkit.utils.visualization import compute_pulse_response, get_tile_for_plotting
from aihwkit.simulator.configs import build_config

def plot_asymmetric(device, ax=None, n_steps=30, n_traces=1, noise_free=True, w_init=None, asym_up=1, asym_down=1, asym_only=False, **kwargs):
    """
    Plot the device response with an asymmetric pulsing scheme. Currently only works for UP direction asymmetry.

    Args:
        device: device configuration
        ax: matplotlib axis to plot on
        n_steps: number of pulses of each domain (positive, negative)
        n_traces: number of traces to plot
        noise_free: whether to plot noise-free traces
        w_init: initial weight value
        asym_up: number of up pulses for each up pulse
        asym_down: number of down pulses for each up pulse
        asym_only: whether to plot only the asymmetric pulses (else also plots the response with the normal scheme)

    Returns:
        ax
    """
    total_iters = 2 * n_steps

    rpu_config = build_config('mp', device=device)

    analog_tile = get_tile_for_plotting(
        rpu_config, n_traces, False, noise_free=noise_free, w_init=w_init
    )
    analog_tile_asymmetric = get_tile_for_plotting(
        rpu_config, n_traces, False, noise_free=noise_free, w_init=w_init
    )

    direction = np.sign(np.sin(np.pi * (np.arange(total_iters) + 1) / n_steps))
    if w_init == 0:
        direction = np.concatenate((direction[n_steps//2:], direction[:n_steps//2]))
    direction_w_asymmetric = []
    for d in direction:
        if d == -1:
            direction_w_asymmetric.append(-1)
        else:
            for _ in range(asym_up):
                direction_w_asymmetric.append(1)
            for _ in range(asym_down):
                direction_w_asymmetric.append(-1)
    direction_w_asymmetric = np.array(direction_w_asymmetric)

    w_trace = compute_pulse_response(analog_tile, direction, use_forward=True).reshape(-1, n_traces)
    w_trace_asymmetric = compute_pulse_response(
        analog_tile_asymmetric, direction_w_asymmetric, use_forward=True).reshape(-1, n_traces)

    pulse_numbers = np.arange(total_iters)
    asym_step = 1/(asym_up + asym_down)
    # asymmetric_pulse_numbers = np.concatenate(
    #     (np.arange(asym_step, total_iters/2 + asym_step, asym_step), np.arange(total_iters/2+1, total_iters+1)))
    asymmetric_pulse_numbers = [0] * (asym_up + asym_down)
    for i, d in enumerate(direction[1:]):
        if d == -1:
            asymmetric_pulse_numbers.append(i+1)
        else:
            for j in range(asym_up + asym_down):
                asymmetric_pulse_numbers.append(i+(j+1)*asym_step)
    asymmetric_pulse_numbers = np.array(asymmetric_pulse_numbers)

    tol = asym_step/2
    effective_idx = np.where(np.isclose(asymmetric_pulse_numbers % 1, 0, rtol=tol) |
                             np.isclose(asymmetric_pulse_numbers % 1, 1, rtol=tol))[0]
    w_trace_asymmetric_effective = w_trace_asymmetric[effective_idx]
    asymmetric_pulse_numbers_effective = asymmetric_pulse_numbers[effective_idx]

    if asym_only:
        if ax is None:
            raise ValueError("ax must be provided when asym_only is True")
        ax.plot(asymmetric_pulse_numbers, w_trace_asymmetric, color=kwargs.get('color', None), lw=0.8)
        ax.plot(asymmetric_pulse_numbers_effective, w_trace_asymmetric_effective, marker='o', **kwargs)
    else:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(pulse_numbers, w_trace, marker='o', color='r', label='Symmetric pulse scheme')
        ax.plot(asymmetric_pulse_numbers, w_trace_asymmetric, color='b', lw=0.8)

        ax.plot(asymmetric_pulse_numbers_effective, w_trace_asymmetric_effective,
                marker='o', color='b', label='Asymmetric pulse scheme')

        ax.legend()
        ax.set(title=f"Pulse response (symmetric vs asymmetric {asym_up} up {asym_down} down)",
               xlabel="Pulse number", ylabel="Weight [conductance]")

    return ax

def plot_symmetry_point(device, ax=None, n_steps=30, n_traces=1, noise_free=True, pre_alternating_pulses=0, alternating_pulses=None, sp_search_len=5, **kwargs):
    """
    Plot a sequence of positive pulses, then negative, then alternating. Also finds where the symetry point is.

    Args:
        device: device configuration
        ax: matplotlib axis to plot on
        n_steps: number of pulses of each domain (positive, negative, alternating)
        n_traces: number of traces to plot
        noise_free: whether to plot noise-free traces
        pre_alternating_pulses: number of positive pulses before alternating sequence (to escape low conductance region)
        alternating_pulses: number of alternating pulses to plot (if None, will be n_steps)
        sp_search_len: number of pulses to consider starting from the end to find the symmetry point (will be the average of these points)

    Returns:
        ax
    """
    alternating_pulses = n_steps if alternating_pulses is None else alternating_pulses
    total_iters = 2 * n_steps + pre_alternating_pulses + alternating_pulses

    rpu_config = build_config('mp', device=device)

    analog_tile = get_tile_for_plotting(rpu_config, n_traces, False, noise_free=noise_free)

    pre_alt = np.ones(pre_alternating_pulses)
    direction = np.concatenate((np.ones(n_steps), -np.ones(n_steps), pre_alt, np.sign(np.sin(np.pi * (np.arange(alternating_pulses) + 1/2)))))
    w_trace = compute_pulse_response(analog_tile, direction, use_forward=True).reshape(-1, n_traces)

    domain_walls = np.where(np.diff(direction) != 0)[0]
    pulse_numbers = np.arange(total_iters)

    # find symmetry point
    symmetry_point = np.mean(w_trace[-sp_search_len:])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(pulse_numbers, w_trace, **kwargs)
    for dw in domain_walls[:2 if pre_alternating_pulses == 0 else 3]:
        ax.axvline(dw, color='k', lw=0.5, ls='dotted')
    ax.axhline(symmetry_point, color='r', lw=0.5, ls='--', label=f'Symmetry point = {symmetry_point:.2f}')

    return ax