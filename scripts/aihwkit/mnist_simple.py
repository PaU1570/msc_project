# This script takes a _Summary.dat file and uses aihwkit to fit a model and train it on the MNIST dataset.
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

# pytorch/lightning imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import Generator, argmax
from torch import nn
from torch.nn.functional import nll_loss
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy

# aihwkit imports
from aihwkit.simulator.configs import SoftBoundsReferenceDevice, build_config
from aihwkit.utils.fitting import fit_measurements
from aihwkit.nn.conversion import convert_to_analog_mapped
from aihwkit.optim import AnalogSGD

# Define directory where dataset is stored
PATH_DATASET = os.path.join("/scratch/msc24h18/msc_project/data", "DATASET")

SEED = 2024

def read_device_data(filename):
    """
    Read the device data from a _Summary.dat file.

    Args:
        filename (str): path to the _Summary.dat file
    Returns:
        pulses (np.array): array of pulses (-1 for negative, 1 for positive)
        conductance (np.array): array of normalized conductance values centered around 0
    """
    data = pd.read_csv(filename, skiprows=6)
    conductance_l = np.array(1 / data['R_high (ohm)'])
    conductance_h = np.array(1 / data['R_low (ohm)'])
    conductance = (conductance_l + conductance_h) / 2

    def normalize(c):
        return 2 * (c - c.mean()) / (c.max() - c.min())

    conductance_l = normalize(conductance_l)
    conductance_h = normalize(conductance_h)
    conductance = normalize(conductance)

    pulses = np.array([-1 if v > 0 else 1 for v in data['Pulse Amplitude (V)']]) # ltp pulses are expected to be positive for fitting to work

    return pulses, conductance

def get_fit(pulses, conductance, device_config=SoftBoundsReferenceDevice()):
    """
    Use the fitting utility from aihwkit to fit the device data.

    Args:
        pulses (np.array): array of pulses (-1 for negative, 1 for positive)
        conductance (np.array): array of normalized conductance values centered around 0
        device_config (PulsedDevice): device configuration to use for fitting
    Returns:
        fit_results: output of the aihwkit.utils.fitting.fit_measurement function
    """
    params = {'dw_min': (0.1, 0.001, 1.0),
          'up_down': (0.0, -0.99, 0.99),
          'w_max': (1.0, 0.1, 2.0),
          'w_min': (-1.0, -2.0, -0.1),
          }

    result, device_config_fit, best_model_fit = fit_measurements(
        params,
        pulses,
        conductance,
        device_config)
    
    return result, device_config_fit, best_model_fit

def add_noise(device_config, fit, data,
              dw_min_dtod=0.3,
              dw_min_std=0.3,
              w_min_dtod=0.3,
              w_max_dtod=0.3,
              up_down_dtod=0.01,
              write_noise_std_mult=1,
              subtract_symmetry_point=True,
              reference_std=0.05,
              enforce_consistency=True,
              dw_min_dtod_log_normal=False,
              mult_noise=False,
              construction_seed=SEED):
    """
    Add noise to the device configuration based on the fit and the data.
    
    Args:
        device_config (PulsedDevice): device configuration to add noise to
        fit (dict): best_model_fit result from get_fit()
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
    device_config.subtract_symmetry_point = subtract_symmetry_point
    device_config.reference_std = reference_std  # assumed programming error of the reference device
    device_config.enforce_consistency=enforce_consistency  # do not allow dead devices
    device_config.dw_min_dtod_log_normal=dw_min_dtod_log_normal # more realistic to use log-normal
    device_config.mult_noise=mult_noise # additive noise
    device_config.construction_seed = construction_seed

    return device_config


def fig_fit(pulses, measured, fit, filename):
    pulse_change_idx = np.where(np.diff(pulses) != 0)[0]

    fig, ax =  plt.subplots(figsize=(8, 6))
    ax.plot(measured, label='Measured')
    ax.plot(fit, label='Fit')
    ax.vlines(pulse_change_idx, measured.min(), measured.max(), color='k', ls='--', lw=0.5)
    ax.set(xlabel='Pulse number', ylabel='Weight [conductance]', title='Conductance vs pulse number')
    ax.legend()

    plt.savefig(filename)

class LitAnalogModel(pl.LightningModule):
    def __init__(self, model, rpu_config, lr=0.5):
        super().__init__()
        # convert model to analog
        self.analog_model = convert_to_analog_mapped(model, rpu_config)
        self.lr = lr

    def forward(self, x):
        # flatten the input image
        x_reshaped = x.reshape(x.shape[0], -1)
        return self.analog_model(x_reshaped)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nll_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        y_hat = self(x)
        loss = nll_loss(y_hat, y)
        preds = argmax(y_hat, dim=1)
        acc = accuracy(preds, y, task='multiclass', num_classes=10)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')
    
    def configure_optimizers(self):
        optimizer = AnalogSGD(self.analog_model.parameters(), lr=self.lr)
        return optimizer
    
def get_dataset(batch_size=5, num_workers=23, split=[0.8, 0.2]):
    trainval_set = MNIST(PATH_DATASET, train=True, download=True, transform=ToTensor())
    test_set = MNIST(PATH_DATASET, train=False, download=True, transform=ToTensor())
    
    train_set, valid_set = random_split(trainval_set, split, generator=Generator().manual_seed(SEED))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    plt.style.use('ggplot')

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train a model on the MNIST dataset using aihwkit.')
    # System arguments
    parser.add_argument('filename', type=str, help='Path to the _Summary.dat file')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory (a folder will be created in this directory)')
    parser.add_argument('--no_subdir', action='store_true', help="Don't create a subdirectory in the output directory")
    # Training arguments
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train the model')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--algorithm', type=str, choices=['sgd', 'mp', 'tt', 'ttv2', 'c-ttv2', 'agad'], default='ttv2', help='Training algorithm to use')
    # Fitting arguments
    parser.add_argument('--dw_min_dtod', type=float, default=0.3, help='Device-to-device variation of the minimum weight change')
    parser.add_argument('--dw_min_std', type=float, default=5.0, help='Pulse-to-pulse variation of the minimum weight change')
    parser.add_argument('--w_min_dtod', type=float, default=0.1, help='Device-to-device variation of the minimum weight')
    parser.add_argument('--w_max_dtod', type=float, default=0.1, help='Device-to-device variation of the maximum weight')
    parser.add_argument('--up_down_dtod', type=float, default=0.05, help='Device-to-device variation of the up/down asymmetry')
    parser.add_argument('--write_noise_std_mult', type=float, default=1, help='Multiplier for the write noise standard deviation')
    parser.add_argument('--reference_std', type=float, default=0.05, help='Assumed programming error of the reference device')
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
    pulses, conductance = read_device_data(filename)

    # Fit the device data
    result, device_config_fit, best_model_fit = get_fit(pulses, conductance)

    # Add noise to the device configuration
    device_config = add_noise(device_config_fit, best_model_fit, conductance,
                              dw_min_dtod=args.dw_min_dtod,
                              dw_min_std=args.dw_min_std,
                              w_min_dtod=args.w_min_dtod,
                              w_max_dtod=args.w_max_dtod,
                              up_down_dtod=args.up_down_dtod,
                              write_noise_std_mult=args.write_noise_std_mult,
                              reference_std=args.reference_std)

    # Plot the fit
    if output_dir is not None:
        fig_fit(pulses, conductance, best_model_fit, os.path.join(output_dir, 'fit.png'))

    # Define the training dataset
    train_loader, valid_loader, test_loader = get_dataset()

    # Define the model
    input_size = 784
    hidden_sizes = [256, 128]
    output_size = 10

    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0], True),
        nn.Sigmoid(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1], True),
        nn.Sigmoid(),
        nn.Linear(hidden_sizes[1], output_size, True),
        nn.LogSoftmax(dim=1)
    )

    print(model)
    print('-' * 80)

    rpu_config = build_config(args.algorithm, device=device_config, construction_seed=SEED)
    if args.algorithm == 'ttv2':
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.mapping.weight_scaling_omega= 0.6
        rpu_config.device.fast_lr = 0.01
        rpu_config.device.auto_granularity = 15000

    # Save the device configuration
    if output_dir is not None:
        with open(os.path.join(output_dir, 'RPU_Config.txt'), 'w') as f:
            f.write(str(rpu_config))


    # Create lightning model    
    lit_model = LitAnalogModel(model, rpu_config)

    # Train the model
    logger = pl.loggers.CSVLogger(save_dir=output_dir, name=args.algorithm) if output_dir is not None else False
    callbacks = [EarlyStopping(monitor='val_loss', mode='min')] if args.early_stopping else None
    trainer = pl.Trainer(max_epochs=args.epochs,
                         enable_checkpointing=False,
                         logger=logger,
                         callbacks=callbacks)
    trainer.fit(lit_model, train_loader, valid_loader)

    # Test the model
    trainer.test(lit_model, test_loader)

    if output_dir is not None:
        print(f'Output files saved in {output_dir}')