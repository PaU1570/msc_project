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
import pickle

# pytorch/lightning imports
import torch
from torch import Generator, argmax
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import nll_loss
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split

# aihwkit imports
from aihwkit.simulator.configs import build_config
from aihwkit.utils.fitting import fit_measurements
from aihwkit.nn.conversion import convert_to_analog_mapped
from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.simulator.rpu_base import cuda
from aihwkit.simulator.configs import IOParameters, UpdateParameters, PulseType

from fit_piecewise import read_conductance_data, fit_piecewise_device

# Check device
USE_CUDA = 0
if cuda.is_compiled():
   USE_CUDA = 1
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
# DEVICE = torch.device("cpu")
print(f'Using device: {DEVICE}')

# Define directory where dataset is stored
PATH_DATASET = os.path.join("/scratch/msc24h18/msc_project/data", "DATASET")

SEED = 2024

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
    
def get_dataset(batch_size=64, num_workers=23, split=[0.8, 0.2]):
    trainval_set = MNIST(PATH_DATASET, train=True, download=True, transform=ToTensor())
    test_set = MNIST(PATH_DATASET, train=False, download=True, transform=ToTensor())
    
    train_set, valid_set = random_split(trainval_set, split, generator=Generator().manual_seed(SEED))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader

# train and test routines
def train(model, train_set, valid_set, optimizer, scheduler, epochs=3, save_weights=False):
    """Train the network.

    Args:
        model (nn.Module): model to be trained.
        train_set (DataLoader): dataset of elements to use as input for training.
        valid_set (DataLoader): dataset of elements to use as input for validation.
        epochs (int): number of epochs to train the model.
        save_weights (bool): if True, save the weights at each epoch.

    Returns:
        metrics (np.ndarray): array with the following values in the columns: epoch, train_loss, valid_loss, valid_accuracy.
        weights (list): list of weights at each epoch (if save_weigths=True), otherwise only the initial weights.
    """
    metrics = np.zeros((epochs, 4))
    weights = [model.get_weights()]

    classifier = nn.NLLLoss()

    for epoch_number in range(epochs):
        print(f"Epoch {epoch_number}:")
        total_loss = 0
        for i, (images, labels) in enumerate(train_set):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            # Flatten MNIST images into a 784 vector.
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            # Add training Tensor to the model (input).
            output = model(images)
            loss = classifier(output, labels)

            # Run training (backward propagation).
            loss.backward()

            # Optimize weights.
            optimizer.step()

            total_loss += loss.item()

        print("\t- Training loss: {:.16f}".format(total_loss / len(train_set)))

        # Save weights.
        if save_weights:
            weights.append(model.get_weights())

        # Evaluate the model.
        predicted_ok = 0
        total_images = 0
        val_loss = 0
        with torch.no_grad():
            for images, labels in valid_set:
                # Predict image.
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                images = images.view(images.shape[0], -1)
                pred = model(images)
                loss = classifier(pred, labels)
                val_loss += loss.item()

                _, predicted = torch.max(pred.data, 1)
                total_images += labels.size(0)
                predicted_ok += (predicted == labels).sum().item()

            print(f"\t- Validation loss: {val_loss / len(valid_set):.16f}")
            print(f"\t- Validation accuracy: {predicted_ok / total_images:.4f}")

        # Decay learning rate if needed.
        scheduler.step()

        # Update metrics.
        metrics[epoch_number, 0] = epoch_number
        metrics[epoch_number, 1] = total_loss / len(train_set) # train_loss
        metrics[epoch_number, 2] = val_loss / len(valid_set) # valid_loss
        metrics[epoch_number, 3] = predicted_ok / total_images # valid_accuracy

    return metrics, weights

def test(model, test_set):
    """Test trained network

    Args:
        model (nn.Model): Trained model to be evaluated
        test_set (DataLoader): Test set to perform the evaluation
    """
    # Setup counter of images predicted to 0.
    predicted_ok = 0
    total_images = 0

    model.eval()

    for images, labels in test_set:
        # Predict image.
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        images = images.view(images.shape[0], -1)
        pred = model(images)

        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()

    print("\nNumber Of Images Tested = {}".format(total_images))
    print("Model Accuracy = {}".format(predicted_ok / total_images))

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

    # Plot the fit
    if output_dir is not None:
        fig_fit(pulses, conductance, model_response, os.path.join(output_dir, 'fit.png'))

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

    rpu_config = build_config('mp', device=device_config, construction_seed=SEED)
    up_params = UpdateParameters()
    if args.pulse_type == 'none':
        up_params.pulse_type = PulseType.NONE
    elif args.pulse_type == 'noneWithDevice':
        up_params.pulse_type = PulseType.NONE_WITH_DEVICE

    rpu_config.update = up_params
    model = convert_to_analog_mapped(model, rpu_config=rpu_config)

    if USE_CUDA:
        model.cuda()

    print(model)
    print('-' * 80)

    # Save the device configuration
    if output_dir is not None:
        with open(os.path.join(output_dir, 'RPU_Config.txt'), 'w') as f:
            f.write(str(rpu_config))


    # Train the model
    optimizer = AnalogSGD(model.parameters(), lr=0.5)
    # optimizer = AnalogAdam(model.parameters(), lr=0.05, betas=(0.9, 0.999), eps=1e-8)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    metrics, weights = train(model, train_loader, valid_loader, optimizer, scheduler, epochs=args.epochs, save_weights=args.save_weights)

    # Test the model
    test(model, test_loader)

    # Save training metrics
    df = pd.DataFrame(metrics, columns=['epoch', 'train_loss', 'val_loss', 'val_acc'])
    if output_dir is not None:
        df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

    if args.save_weights:
        with open(os.path.join(output_dir, 'weights.pkl'), 'wb') as f:
            pickle.dump(weights, f)


    if output_dir is not None:
        print(f'Output files saved in {output_dir}')