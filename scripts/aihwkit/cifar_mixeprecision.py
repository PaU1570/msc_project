# This script takes a _Summary.dat file and uses aihwkit to fit a model and train it on the CIFAR-10 dataset.
# Output files:
#   logs: directory containing the logs from pytorch, includding loss and accuracy metrics
#   - fit.png: plot of the fit superimposed on the measurement data
#   - train.png: plot of the training loss
#   - validation.png: plot of the validation loss and accuracy
#   - Summary.dat: a copy of the input file
#   - RPU_config.txt: the RPU config used for the training

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pickle

from torch import nn
from torch.optim.lr_scheduler import StepLR

# aihwkit imports
from aihwkit.simulator.configs import build_config
from aihwkit.nn.conversion import convert_to_analog_mapped
from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.simulator.configs import IOParameters, UpdateParameters, PulseType, DigitalRankUpdateRPUConfig
from aihwkit.simulator.parameters.mapping import MappingParameter
from aihwkit.simulator.parameters.enums import AsymmetricPulseType
from aihwkit.simulator.configs.compounds import ReferenceUnitCell, MixedPrecisionCompound

from msc_project.utils.fit_piecewise import read_conductance_data, fit_piecewise_device
from msc_project.models.base import BaseCIFAR10
from msc_project.utils.asymmetric_pulsing import plot_symmetry_point

SEED = 2024
N_CLASSES = 10

class ResidualBlock(nn.Module):
    """Residual block of a residual network with option for the skip connection."""

    def __init__(self, in_ch, hidden_ch, use_conv=False, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_ch)

        if use_conv:
            self.convskip = nn.Conv2d(in_ch, hidden_ch, kernel_size=1, stride=stride)
        else:
            self.convskip = None

    def forward(self, x):
        """Forward pass"""
        y = nn.functional.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.convskip:
            x = self.convskip(x)
        y += x
        return nn.functional.relu(y)
    
def concatenate_layer_blocks(in_ch, hidden_ch, num_layer, first_layer=False):
    """Concatenate multiple residual block to form a layer.

    Returns:
       List: list of layer blocks
    """
    layers = []
    for i in range(num_layer):
        if i == 0 and not first_layer:
            layers.append(ResidualBlock(in_ch, hidden_ch, use_conv=True, stride=2))
        else:
            layers.append(ResidualBlock(hidden_ch, hidden_ch))
    return layers

def create_model():
    """ResNet34 inspired analog model.

    Returns:
       nn.Modules: created model
    """

    block_per_layers = (3, 4, 6, 3)
    base_channel = 16
    channel = (base_channel, 2 * base_channel, 4 * base_channel)

    l0 = nn.Sequential(
        nn.Conv2d(3, channel[0], kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
    )

    l1 = nn.Sequential(
        *concatenate_layer_blocks(channel[0], channel[0], block_per_layers[0], first_layer=True)
    )
    l2 = nn.Sequential(*concatenate_layer_blocks(channel[0], channel[1], block_per_layers[1]))
    l3 = nn.Sequential(*concatenate_layer_blocks(channel[1], channel[2], block_per_layers[2]))
    l4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(channel[2], N_CLASSES))

    return nn.Sequential(l0, l1, l2, l3, l4)

def add_noise(device_config, fit, data,
              dw_min_dtod=0.3,
              dw_min_std=0.3,
              w_min_dtod=0.3,
              w_max_dtod=0.3,
              up_down_dtod=0.01,
              write_noise_std_mult=1,
              enforce_consistency=True,
              dw_min_dtod_log_normal=False,
              construction_seed=SEED):
    """
    Add noise to the device configuration based on the fit and the data.
    
    Args:
        device_config (PulsedDevice): device configuration to add noise to
        fit (dict): model_response result from get_fit()
        data (np.array): normalized conductance data

    Returns:
        device_config (PulsedDevice): device configuration with added noise
    """
    std = (data - fit).std() / device_config.dw_min
    device_config.dw_min_dtod = dw_min_dtod
    device_config.dw_min_std = dw_min_std
    device_config.w_min_dtod = w_min_dtod
    device_config.w_max_dtod = w_max_dtod
    device_config.up_down_dtod = up_down_dtod
    device_config.write_noise_std = np.sqrt(std ** 2 - device_config.dw_min_std ** 2)/2 * write_noise_std_mult
    device_config.enforce_consistency=enforce_consistency  # do not allow dead devices
    device_config.dw_min_dtod_log_normal=dw_min_dtod_log_normal # more realistic to use log-normal
    device_config.construction_seed = construction_seed

    return device_config


def fig_fit(pulses, measured, fit, filename):
    pulse_num = np.arange(len(pulses))
    pulse_change_idx = np.where(np.diff(pulses) != 0)[0]

    fig, ax =  plt.subplots(figsize=(8, 6))
    ax.scatter(pulse_num, measured, label='Measured')
    ax.plot(fit, label='Fit', color='b')
    ax.vlines(pulse_change_idx, measured.min(), measured.max(), color='k', ls='--', lw=0.5)
    ax.set(xlabel='Pulse number', ylabel='Weight [conductance]', title='Conductance vs pulse number')
    ax.legend()

    plt.savefig(filename)

if __name__ == "__main__":
    plt.style.use('ggplot')

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train a model on the CIFAR-10 dataset using aihwkit.')
    # System arguments
    parser.add_argument('filename', type=str, help='Path to the _Summary.dat file')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory (a folder will be created in this directory)')
    parser.add_argument('--no_subdir', action='store_true', help="Don't create a subdirectory in the output directory")
    parser.add_argument('--save_fit', action='store_true', help='Save the fit plot')
    # Training arguments
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train the model')
    #parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    #parser.add_argument('--algorithm', type=str, choices=['sgd', 'mp', 'tt', 'ttv2', 'c-ttv2', 'agad'], default='mp', help='Training algorithm to use')
    # Fitting arguments
    parser.add_argument('--dw_min_dtod', type=float, default=0.3, help='Device-to-device variation of the minimum weight change')
    parser.add_argument('--dw_min_std', type=float, default=0.3, help='Pulse-to-pulse variation of the minimum weight change')
    parser.add_argument('--w_min_dtod', type=float, default=0.3, help='Device-to-device variation of the minimum weight')
    parser.add_argument('--w_max_dtod', type=float, default=0.3, help='Device-to-device variation of the maximum weight')
    parser.add_argument('--up_down_dtod', type=float, default=0.01, help='Device-to-device variation of the up/down asymmetry')
    parser.add_argument('--write_noise_std_mult', type=float, default=1, help='Multiplier for the write noise standard deviation')
    parser.add_argument('--pulse_type', type=str, choices=['none', 'noneWithDevice', 'stochastic'], default='stochastic', help='Pulse type to use')
    parser.add_argument('--save_weights', action='store_true', help='Save the weights at each epoch')
    parser.add_argument('--asymmetric_pulsing_dir', type=str, choices=['Up', 'Down', 'None'], default='None', help='Asymmetric pulsing direction')
    parser.add_argument('--asymmetric_pulsing_up', type=int, default=1, help='Asymmetric pulsing up number')
    parser.add_argument('--asymmetric_pulsing_down', type=int, default=1, help='Asymmetric pulsing down number')
    parser.add_argument('--use_reference_device', action='store_true', help='Use a ReferenceUnitCell to substract symmetry point from weight')
    parser.add_argument('--weight_scaling_omega', type=float, default=0, help='Weight scaling omega')
    parser.add_argument('--learn_out_scaling', action='store_true', help='Learn output scaling')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='sgd', help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.5, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.8, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer')
    args = parser.parse_args()

    filename = args.filename
    if args.no_subdir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.output_dir, os.path.basename(filename).replace('_Summary.dat', '')) if args.output_dir is not None else None
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the summary file
    if output_dir is not None:
        summary_filename = os.path.join(output_dir, 'Summary.dat')
        os.system(f'cp "{filename}" "{summary_filename}"')

    # Read the device data
    conductance, pulses = read_conductance_data(filename)

    # Fit the device data
    result, device_config, model_response = fit_piecewise_device(conductance, pulses)

    # Add noise to the device configuration
    device_config = add_noise(device_config, model_response, conductance,
                              dw_min_dtod=args.dw_min_dtod,
                              dw_min_std=args.dw_min_std,
                              w_min_dtod=args.w_min_dtod,
                              w_max_dtod=args.w_max_dtod,
                              up_down_dtod=args.up_down_dtod,
                              write_noise_std_mult=args.write_noise_std_mult)
    
    if args.use_reference_device:
        ax, sp = plot_symmetry_point(device_config, ax=None, n_steps=0, pre_alternating_pulses=0, alternating_pulses=50, w_init=0)
        print(f'Symmetry point: {sp}')
        device_config = ReferenceUnitCell([device_config], construction_seed=SEED)
        # set reference weights to symmetry point

    # Plot the fit
    if output_dir is not None and args.save_fit:
        fig_fit(pulses, conductance, model_response, os.path.join(output_dir, 'fit.png'))

    # Create the model
    model = create_model()

    #rpu_config = build_config('mp', device=device_config, construction_seed=SEED)
    up_params = UpdateParameters()
    if args.pulse_type == 'none':
        up_params.pulse_type = PulseType.NONE
    elif args.pulse_type == 'noneWithDevice':
        up_params.pulse_type = PulseType.NONE_WITH_DEVICE

    rpu_config = DigitalRankUpdateRPUConfig(
            device=MixedPrecisionCompound(
                device=device_config,
                asymmetric_pulsing_dir = AsymmetricPulseType(args.asymmetric_pulsing_dir),
                asymmetric_pulsing_up = args.asymmetric_pulsing_up,
                asymmetric_pulsing_down = args.asymmetric_pulsing_down,
                construction_seed=SEED),
            forward=IOParameters(),
            backward=IOParameters(),
            update=up_params,
            mapping=MappingParameter(weight_scaling_omega=args.weight_scaling_omega,
                                     weight_scaling_lr_compensation=(args.weight_scaling_omega != 0),
                                     learn_out_scaling=args.learn_out_scaling)
        )
        
    model = convert_to_analog_mapped(model, rpu_config=rpu_config)

    if args.use_reference_device:
        initial_weights = model.get_weights()
        model.apply_to_analog_tiles(lambda tile: tile.set_hidden_update_index(1))
        ref_weights = model.get_weights()
        for key, val in ref_weights.items():
            for t in val:
                if t is not None:
                    t.fill_(sp)
        model.set_weights(ref_weights)
        model.apply_to_analog_tiles(lambda tile: tile.set_hidden_update_index(0))
        for key, val in initial_weights.items():
            for t in val:
                if t is not None:
                    t = t - sp
        model.set_weights(initial_weights)

    # Create the CIFAR10 model
    cifar_model = BaseCIFAR10(model, seed=SEED)

    # Define the training dataset
    train_loader, valid_loader, test_loader = cifar_model.get_dataset()

    print(cifar_model.model)
    print('-' * 80)

    # Save the device configuration
    if output_dir is not None:
        with open(os.path.join(output_dir, 'RPU_Config.txt'), 'w') as f:
            f.write(str(rpu_config))

    # Train the model
    if args.optimizer == 'sgd':
        optimizer = AnalogSGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = AnalogAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-8)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    cifar_model.optimizer = optimizer
    cifar_model.scheduler = scheduler

    metrics, weights, analog_weights = cifar_model.train(train_loader, valid_loader, epochs=args.epochs, save_weights=args.save_weights)

    # Test the model
    cifar_model.test(test_loader)

    # Save training metrics
    df = pd.DataFrame(metrics, columns=['epoch', 'train_loss', 'val_loss', 'val_acc'])
    if output_dir is not None:
        df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

    if args.save_weights:
        with open(os.path.join(output_dir, 'weights.pkl'), 'wb') as f:
            pickle.dump(weights, f)
        with open(os.path.join(output_dir, 'analog_weights.pkl'), 'wb') as f:
            pickle.dump(analog_weights, f)
        with open(os.path.join(output_dir, 'out_scaling_alpha.pkl'), 'wb') as f:
            pickle.dump([tile.out_scaling_alpha for tile in model.analog_tiles()], f)


    if output_dir is not None:
        print(f'Output files saved in {output_dir}')