import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# aiwhkit imports
from aihwkit.utils.fitting import fit_measurements
from aihwkit.simulator.configs import PiecewiseStepDevice

# default degree for polynomial fit
DEFAULT_FIT_DEGREE = 5

# resolution of the piecewise update steps
N_ELEMENTS = 32

def read_conductance_data(filename):
    """
    Read conductance data from a Summary.dat file.

    Args:
        filename (str): path to the Summary.dat file. Must have the correct format.
    Returns:
        conductance (np.array): conductance values.
        pulses (np.array): pulse polarity sequence.
    """

    data = pd.read_csv(filename, skiprows=6)
    # take average conductance
    conductance_l = np.array(1 / data['R_high (ohm)'])
    conductance_h = np.array(1 / data['R_low (ohm)'])
    conductance = (conductance_l + conductance_h) / 2

    def normalize(c):
        return 2 * (c - c.mean()) / (c.max() - c.min())

    conductance = normalize(conductance)

    # generate pulse polarity sequence (positive for ltp, negative for ltd, otherwise fitting does not work)
    pulses = np.array([-1 if v > 0 else 1 for v in data['Pulse Amplitude (V)']])

    return conductance, pulses

def fit_conductance_change(conductance, pulses, degree=DEFAULT_FIT_DEGREE):
    """
    Fit the conductance changve vs conductance data with a polynomial function.
    
    Args:
        conductance (np.array): conductance values.
        pulses (np.array): pulse polarity sequence.
        degree (int): degree of the polynomial function to fit.

    Returns:
        fit_grad_ltd (np.polynomial.polynomial.Polynomial): polynomial fit for conductance decrease.
        fit_grad_ltp (np.polynomial.polynomial.Polynomial): polynomial fit for conductance increase.
    """

    # compute conductance change vs conductance
    pulses_ltp = np.where(pulses == 1)[0]
    pulses_ltd = np.where(pulses == -1)[0]
    conductance_ltp = conductance[pulses_ltp]
    conductance_ltd = conductance[pulses_ltd]

    # compute gradients
    grad_ltd = np.gradient(conductance_ltd)
    grad_ltp = np.gradient(conductance_ltp)

    # fit polynomial
    fit_grad_ltd = np.polynomial.polynomial.Polynomial.fit(conductance_ltd, grad_ltd, degree)
    fit_grad_ltp = np.polynomial.polynomial.Polynomial.fit(conductance_ltp, grad_ltp, degree)

    return fit_grad_ltd, fit_grad_ltp

def fit_piecewise_device(conductance, pulses, degree=DEFAULT_FIT_DEGREE):
    """
    Fit a piecewise device to the conductance data.
    
    Args:
        conductance (np.array): conductance values.
        pulses (np.array): pulse polarity sequence.
        degree (int): degree of the polynomial function to fit.
    Returns:
        result: result of the fit in lmfit format.
        device_config_fit: fitted device configuration.
        model_response: response curve of the fitted model.
    """

    # get polynomial fits
    fit_grad_ltd, fit_grad_ltp = fit_conductance_change(conductance, pulses, degree)

    # extract piecewise update vectors
    #cond_steps = np.linspace(np.min(conductance), np.max(conductance), N_ELEMENTS)
    cond_steps = np.linspace(-1, 1, N_ELEMENTS)
    piecewise_up = np.abs(fit_grad_ltp(cond_steps))
    piecewise_down = np.abs(fit_grad_ltd(cond_steps))

    dw_min = min(piecewise_up.min(), piecewise_down.min())
    piecewise_up /= dw_min
    piecewise_down /= dw_min

    # parameters to optimize (format: 'x': (x_init, x_min, x_max))
    params = {'dw_min': dw_min,
              'up_down': (0.0, -0.99, 0.99),
              'w_max': (1.0, 0.1, 2.0),
              'w_min': (-1.0, -2.0, -0.1)}

    # fit the piecewise device
    result, device_config_fit, model_response = fit_measurements(
        params,
        pulses,
        conductance,
        PiecewiseStepDevice(
            piecewise_up=list(piecewise_up),
            piecewise_down=list(piecewise_down)))
        
    return result, device_config_fit, model_response

def get_fit(filename):
    conductance, pulses = read_conductance_data(filename)
    return fit_piecewise_device(conductance, pulses)

if __name__ == '__main__':
    plt.style.use('ggplot')

    parser = argparse.ArgumentParser(description='Fit piecewise device to conductance data.')
    parser.add_argument('filename', type=str, help='path to the Summary.dat file')
    parser.add_argument('--degree', type=int, default=DEFAULT_FIT_DEGREE, help='degree of the polynomial fit')
    parser.add_argument('--savefig', type=str, default=None, help='path to save the figure')
    args = parser.parse_args()

    # plot 1: conductance vs pulses with fit
    conductance, pulses = read_conductance_data(args.filename)
    result, device_config_fit, model_response = fit_piecewise_device(conductance, pulses, args.degree)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(np.arange(len(pulses)), conductance, label='Measured', color='r')
    ax.plot(model_response, label='Fit', color='b')
    ax.set(xlabel="Pulse Number", ylabel="[Norm.] Conductance", title="Conductance vs Pulse Number")
    ax.legend()
    if args.savefig:
        plt.savefig(args.savefig)
    else:
        plt.show()
