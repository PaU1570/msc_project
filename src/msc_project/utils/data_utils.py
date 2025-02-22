import numpy as np
import pandas as pd
import os
import re

def get_files(folder_path, extension, contains=None, exclude=None):
    """
    Get a list of all the files in a directory and subdirectories with a specific extension. Order is guaranteed.

    Args:
        folder_path (str): Path to the directory.
        extension (str): File extension to search for.
        contains (str): String that the file name must contain (optional).
        exclude (str): String that the file name must not contain (optional).

    Returns:
        res (list): List of file paths
    """
    res = []
    for root, dirs, files in os.walk(folder_path):
        dirs.sort()
        files.sort()
        for file in files:
            if file.endswith(extension) and (contains is None or contains in file) and (exclude is None or exclude not in file):
                res.append(os.path.join(root, file))

    return res

def get_pas_csv_files(folder_path):
    return get_files(folder_path, '.csv', contains='pulsedAmplitudeSweep', exclude='(ALL)')

def get_metrics_csv_files(folder_path):
    return get_files(folder_path, '.csv', contains='metrics')

def get_summary_files(folder_path):
    return get_files(folder_path, 'Summary.dat')

def get_rpu_txt_files(folder_path):
    return get_files(folder_path, 'RPU_Config.txt')

def read_file(filename):
# read metadata and parameters from file
    with open(filename, 'r') as file:
        lines = file.readlines()
        # Extract metadata
        try:
            test_date = lines[1].split('\t')[1].strip()
            test_time = lines[2].split('\t')[1].strip()
            device = lines[3].split('\t')[1].strip()
            device_name, device_id = device.split('_')
            metadata = {'test_date': test_date, 'test_time': test_time, 'device_name': device_name, 'device_id': device_id}

            # Extract measurement parameters
            param_line = lines[8].strip().split(',')
            value_line = lines[9].strip().split(',')
            meas_params = dict()
            for param, value in zip(param_line, value_line):
                if len(param) > 0:
                    try:
                        meas_params[param] = float(value)
                    except ValueError:
                        meas_params[param] = value

        except:
            return None, None, None

    # read measurement data
    meas_data = np.loadtxt(filename, delimiter=',', skiprows=12)

    return metadata, meas_params, meas_data

def get_k_vals(meas_data, VStartPos, VEndPos, VStartNeg, VEndNeg):
    kpos = []
    kneg = []
    vprev = None
    for i, v in enumerate(meas_data[:,1]):
        if vprev is not None:
            if v > 0 and v > vprev and v > VStartPos and v < VEndPos:
                kpos.append(i)
            elif v < 0 and v < vprev and v < VStartNeg and v > VEndNeg:
                kneg.append(i)
        vprev = v
    kpos = np.array([kpos])
    kneg = np.array([kneg])

    return kpos, kneg

def get_df_1(meas_data, VStartPos, VEndPos, VStartNeg, VEndNeg):
    kpos, kneg = get_k_vals(meas_data, VStartPos, VEndPos, VStartNeg, VEndNeg)

    mask = np.ones(meas_data.shape[0], dtype=bool)
    if len(kpos[0]) > 0:
        mask[kpos] = False
    if len(kneg[0]) > 0:
        mask[kneg] = False

    data = {"Pulse Amplitude (V)": meas_data[:,1], "R_high (ohm)": meas_data[:,3], "Keep": ~mask}
    df = pd.DataFrame(data)
    return df
    
def get_df_2(meas_data, VStartPos, VEndPos, VStartNeg, VEndNeg):
    kpos, kneg = get_k_vals(meas_data, VStartPos, VEndPos, VStartNeg, VEndNeg)

    exp_LTD_raw = np.flip(1. / meas_data[kpos,3][0]) if len(kpos[0]) > 0 else np.array([0, 1])
    exp_LTP_raw = 1. / meas_data[kneg,3][0] if len(kneg[0]) > 0 else np.array([0, 1])
    pulse_num_LTD_raw = np.linspace(0, len(exp_LTD_raw) - 1, len(exp_LTD_raw))
    pulse_num_LTP_raw = np.linspace(0, len(exp_LTP_raw) - 1, len(exp_LTP_raw))
    # normalize
    exp_LTD_norm = (exp_LTD_raw - min(exp_LTD_raw))/(max(exp_LTD_raw) - min(exp_LTD_raw))
    exp_LTP_norm = (exp_LTP_raw - min(exp_LTP_raw))/(max(exp_LTP_raw) - min(exp_LTP_raw))
    pulse_num_LTD_norm = pulse_num_LTD_raw / max(pulse_num_LTD_raw)
    pulse_num_LTP_norm = pulse_num_LTP_raw / max(pulse_num_LTP_raw)

    data = {"Normalized Pulse Number": np.concatenate((pulse_num_LTD_norm, pulse_num_LTP_norm)), "Normalized Conductance": np.concatenate((exp_LTD_norm, exp_LTP_norm)), "Type": np.concatenate((np.repeat("LTD", len(exp_LTD_norm)), np.repeat("LTP", len(exp_LTP_norm))) )}
    df = pd.DataFrame(data)
    return df

def read_summary_file(filename):
    """
    Read summary file and return a dictionary and a pandas DataFrame.
    
    Args:
        filename (str): Path to the summary file.
    
    Returns:
        d (dict): Dictionary containing metadata and parameters.
        exp_data (pd.DataFrame): DataFrame containing the experimental data.
    """
    d = dict()
    with open(filename, 'r') as f:
        data = f.readlines()
        
        # dictionaries keep insertion order since python 3.7
        line = data[1].split(',')
        d['device_name'] = line[2]
        d['device_id'] = line[3].strip()
        d['test_date'] = line[0]
        d['test_time'] = line[1]

        keys = data[2].split(',')
        line = data[3].split(',')
        for key, val in zip(keys, line):
            d[key.strip()] = val.strip()

        keys = data[4].split(',')
        line = data[5].split(',')
        for key, val in zip(keys, line):
            d[key.strip()] = val.strip()

        # find all numeric values in the dirname 3 levels up and put them in the dictionary as val1, val2, ...
        parsename = filename.split('/')[-4]
        numeric_values = re.findall(r'\d+\.\d+|\d+', parsename)
        for i,val in enumerate(numeric_values):
            d[f'val{i+1}'] = float(val)

    exp_data = pd.read_csv(filename, skiprows=6)
    return d, exp_data

def read_rpu_txt(filename):
    d = { # default values from aihwkit
        'dw_min': 0.001,
        'dw_min_dtod': 0.3,
        'dw_min_std': 0.3,
        'w_min': -0.6,
        'w_min_dtod': 0.3,
        'w_max': 0.6,
        'w_max_dtod': 0.3,
        'up_down': 0.0,
        'up_down_dtod': 0.01,
        'reference_std': 0.05,
        'write_noise_std': 0.0
    }

    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return None

    with open(filename, 'r') as f:
        data = f.readlines()

        def get_float(line):
            return float(line.split('=')[1].strip().strip(','))
        
        for line in data:
            if "dw_min=" in line:
                d['dw_min'] = get_float(line)
            if "dw_min_std=" in line:
                d['dw_min_std'] = get_float(line)
            if "dw_min_dtod=" in line:
                d['dw_min_dtod'] = get_float(line)
            if "w_min=" in line:
                d['w_min'] = get_float(line)
            if "w_min_dtod=" in line:
                d['w_min_dtod'] = get_float(line)
            if "w_max=" in line:
                d['w_max'] = get_float(line)
            if "w_max_dtod=" in line:
                d['w_max_dtod'] = get_float(line)
            if "up_down=" in line:
                d['up_down'] = get_float(line)
            if "up_down_dtod=" in line:
                d['up_down_dtod'] = get_float(line)
            if "reference_std=" in line:
                d['reference_std'] = get_float(line)
            if "write_noise_std=" in line:
                d['write_noise_std'] = get_float(line)
            if "granularity=" in line and "asymmetric_granularity" not in line:
                d['granularity'] = get_float(line)
            if "granularity_up=" in line:
                d['granularity_up'] = get_float(line)
            if "granularity_down=" in line:
                d['granularity_down'] = get_float(line)
    
    return d

def _get_shared_keys(dict_list):
    """
    Get the keys that are shared by all dictionaries in a list.

    Args:
        dict_list (list): List of dictionaries.
    Returns:
        shared_keys (set): Set of keys shared by all dictionaries.
    """
    # use dict instead of set to preserve order (python 3.7+)
    shared_keys = dict.fromkeys(dict_list[0].keys())
    for d in dict_list[1:]:
        shared_keys = dict((k, None) for k in shared_keys if k in d)

    return shared_keys

def _get_final_metrics(metrics_files):
    """
    Get the final metrics from the metrics files.

    Args:
        metrics_files (list): List of metrics files.
    Returns:
        final_metrics (dict): Dictionary containing the final metrics (epochs, steps, test_acc, test_loss) in the same order as the files.
    """
    read_keys = ["epoch", "step", "test_acc", "test_loss", "val_acc", "val_loss", "train_loss", "pulses", "pulses_pos", "pulses_neg"]
    write_keys = ["epochs", "steps", "test_acc", "test_loss", "val_acc", "val_loss", "train_loss", "pulses", "pulses_pos", "pulses_neg"]
    final_metrics = dict() # dict.fromkeys(write_keys, None)
    for file in metrics_files:
        metrics = pd.read_csv(file)
        for rk, wk in zip(read_keys, write_keys):
            try:
                value = metrics[rk].dropna().iloc[-1] if not metrics[rk].dropna().empty else None
                if wk not in final_metrics.keys():
                    final_metrics[wk] = [value]
                else:
                    final_metrics[wk].append(value)
            except:
                pass

    return final_metrics

def get_pytorch_df(directory):
    metrics_files = get_metrics_csv_files(directory)
    summary_files = get_summary_files(directory)
    rpu_config_files = get_rpu_txt_files(directory)

    # remove the files that have summary and rpu_config but no metrics (incomplete simulations)
    summary_files = [s for s in summary_files if any(os.path.dirname(s) in m for m in metrics_files)]
    rpu_config_files = [r for r in rpu_config_files if any(os.path.dirname(r) in m for m in metrics_files)]

    final_metrics = _get_final_metrics(metrics_files)

    summaries = [read_summary_file(s)[0] for s in summary_files]
    summaries = {key: np.array([d[key] for d in summaries]) for key in _get_shared_keys(summaries)}

    rpu_configs = [read_rpu_txt(rpu) for rpu in rpu_config_files]
    rpu_configs = {key: np.array([d[key] for d in rpu_configs]) for key in _get_shared_keys(rpu_configs)}

    data = {**summaries, **rpu_configs, **final_metrics}

    df = pd.DataFrame(data)
    df = df.apply(pd.to_numeric, errors='ignore')

    return df

def _compute_area(shape, size):
    """ Compute are of device (circle, octagon, or star). """
    if shape == "Circ":
        return np.pi * size**2
    if shape == "Oct":
        return (34/9) * size**2 # Octagon 'ear' is 1/3 of radius
    if shape == "Star":
        return (2.938926261 * (size/2)**2) + 10 * (0.5245*size * 0.309*size)/2 # area of decagon + the 10 spikes
    
    raise ValueError(f"Unknown shape {shape}")

def _compute_j(v):
    """ Compute current density from voltage using modified Schottky emission model. """
    ALPHA = 3e-4 # [A * s * cm^-3 * K^(3/2)]
    MU = 8.9e-3 # mu(m*/m0)^(3/2) [cm^2 * s^-1 * V^-1]
    PHI_B = 0.19 # [eV]
    EPSILON_0 = 552634.9406 # [e^2 * eV^-1 * cm^-1]
    EPSILON_R = 18.2
    T = 300 # [K]
    K = 8.617e-5 # [eV * K^-1]
    THICKNESS = 5e-7 # [cm] (5nm)

    j = ALPHA * np.power(T, 3/2) * (v/THICKNESS) * MU * np.exp(-(PHI_B - np.sqrt((v/THICKNESS)/(4*np.pi*EPSILON_0*EPSILON_R))) / (K*T))
    return j

def add_energy_to_summary(data, shapes_n_sizes=None):
    """
    Add energy calculations to summary data.

    Args:
        data (pd.DataFrame): DataFrame containing the summary data.
        shapes_n_sizes (dict): Dictionary in the format {'device_id': {'size': size, 'shape': shape}}.
    """
    vmin = abs(data['VEndNeg (V)'])
    j = vmin.apply(_compute_j)
    area = 1 # [cm^2]
    if shapes_n_sizes is not None:
        data['shape'] = data['device_id'].map(lambda x: shapes_n_sizes[x]['shape'])
        data['size'] = data['device_id'].map(lambda x: shapes_n_sizes[x]['size'])
        area = data.apply(lambda x: _compute_area(x['shape'], x['size']), axis=1) * 1e-14 # convert to cm^2
    elif 'shape' in data.columns and 'size' in data.columns:
        area = data.apply(lambda x: _compute_area(x['shape'], x['size']), axis=1) * 1e-14
    
    data['energy'] = vmin * j * area * data['pulseWidth'] * data['pulses']

def add_energy_to_metrics(metrics, summaries, shapes_n_sizes=None):
    """
    Add energy calculations to metrics containing pulse count.
    
    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
        summaries (pd.DataFrame): DataFrame containing the summaries.
        shapes_n_sizes (dict): Dictionary in the format {'device_id': {'size': size, 'shape': shape}}.
    """
    for m, s in zip(metrics, summaries):
        # approximate upper bound of energy by using the most negative voltage and modified schottky emission model
        vmin = abs(float(s[1]["Pulse Amplitude (V)"].min()))
        j = _compute_j(vmin)
        area = 1 # [cm^2]
        if shapes_n_sizes is not None:
            shape = shapes_n_sizes[s[0]["device_id"]]['shape']
            size = shapes_n_sizes[s[0]["device_id"]]['size']
            area = _compute_area(shape, size) * 1e-14 # convert to cm^2
        pwidth = float(s[0]["pulseWidth"])
        m["energy"] = vmin * j * area * pwidth * m["pulses"]
