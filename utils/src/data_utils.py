import numpy as np
import pandas as pd
import os

def get_pas_csv_files(folder_path):
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv') and 'pulsedAmplitudeSweep' in file and '(ALL)' not in file:
                csv_files.append(os.path.join(root, file))

    return csv_files

def get_metrics_csv_files(folder_path):
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv') and 'metrics' in file:
                csv_files.append(os.path.join(root, file))

    return csv_files

def get_summary_files(folder_path):
    summary_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('_Summary.dat'):
                summary_files.append(os.path.join(root, file))

    return summary_files

def read_file(filename):
# read metadata and parameters from file
    with open(filename, 'r') as file:
        lines = file.readlines()
        # Extract metadata
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

    exp_data = pd.read_csv(filename, skiprows=6)
    return d, exp_data

def read_rpu_txt(filename):
    d = {
        'dw_min_dtod': 0.3,
        'dw_min_std': 0.3,
        'w_min_dtod': 0.3,
        'w_max_dtod': 0.3,
        'up_down_dtod': 0.1,
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
            if "dw_min_std=" in line:
                d['dw_min_std'] = get_float(line)
            if "dw_min_dtod=" in line:
                d['dw_min_dtod'] = get_float(line)
            if "w_min_dtod=" in line:
                d['w_min_dtod'] = get_float(line)
            if "w_max_dtod=" in line:
                d['w_max_dtod'] = get_float(line)
            if "up_down_dtod=" in line:
                d['up_down_dtod'] = get_float(line)
            if "reference_std=" in line:
                d['reference_std'] = get_float(line)
            if "write_noise_std=" in line:
                d['write_noise_std'] = get_float(line)
    
    return d