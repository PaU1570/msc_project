import nonlinear_fit as nlf
import data_utils as du
import argparse
import pandas as pd
import numpy as np
import os
import shutil

def analyze_pulseAmplitudeSweep(args):
    """
    Analyze pulsed amplitude sweep data from a directory.

    Args:
        args: argparse.Namespace. The arguments passed to the function. Contains:
            input_dir (str): Directory containing the data
            output_dir (str): Directory to save the output
            cutoffs_file (str): File containing the cutoffs for the data. Data not present in this file will be ignored.

    Returns:
        None. The analyzed data is saved in the output directory.
    """

    input_dir = args.input_dir
    output_dir = args.output_dir
    cutoffs_file = args.cutoffs

    # get all files in the directory
    csv_files = du.get_pas_csv_files(input_dir)

    # read cutoffs
    if cutoffs_file is not None:
        cutoffs = pd.read_csv(cutoffs_file)

    for file in csv_files:
        # read file
        metadata, meas_params, meas_data = du.read_file(file)

        if cutoffs is None:
            VStartPos = meas_params['startVolage1'] # (sic)
            VEndPos = meas_params['endVoltage1']
            VStartNeg = meas_params['startVolage2'] # (sic)
            VEndNeg = meas_params['endVoltage2']
        else:
            # check if the file is present in the cutoffs
            row = cutoffs[
                (cutoffs['device_name'] == metadata['device_name']) &
                (cutoffs['device_id'] == metadata['device_id']) &
                (cutoffs['test_date'] == metadata['test_date']) &
                (cutoffs['test_time'] == metadata['test_time'])
            ]

            if len(row) == 0:
                print(f"File {file} not present in the cutoffs file. Skipping...")
                continue
            elif len(row) > 1:
                print(f"Multiple entries found for file {file} in the cutoffs file. Skipping...")
                continue

            VStartPos = row['VStartPos'].values[0]
            VEndPos = row['VEndPos'].values[0]
            VStartNeg = row['VStartNeg'].values[0]
            VEndNeg = row['VEndNeg'].values[0]

        relative_path = os.path.splitext(os.path.relpath(file, input_dir))[0]
        output_folder = os.path.join(output_dir, relative_path)

        nlf.analyze(file,
                    output_folder,
                    cutoffs=(VStartPos, VEndPos, VStartNeg, VEndNeg),
                    plotmode=2,
                    savesummary=True)
        
def generate_configs(args):
    """
    Generate configuration files for NeuroSim.

    Args:
        args: argparse.Namespace. The arguments passed to the function. Contains:
            input_dir (str): Directory containing the data
            output_dir (str): Directory to save the output
            readPulseWidth (float): Pulse width for read operation
            readVoltage (float): Voltage for read operation
            numLevelsLTD (int): Number of levels for LTD
            numLevelsLTP (int): Number of levels for LTP
            copy_summary (bool): Copy summary files to output directory
            ref (str): Reference file for configuration

    Returns:
        None. The configuration files are saved in the output directory.
    """
    input_dir = args.input_dir
    output_dir = args.output_dir
    readPulseWidth = args.readPulseWidth
    readVoltage = args.readVoltage
    sigmaCtoC = args.sigmaCtoC
    sigmaDtoD = args.sigmaDtoD
    numLevelsLTD = args.numLevelsLTD if args.numLevelsLTD is not None else 'count'
    numLevelsLTP = args.numLevelsLTP if args.numLevelsLTP is not None else 'count'
    copy_summary = args.copy_summary
    ref = args.ref

    summary_files = du.get_summary_files(input_dir)
    
    for file in summary_files:
        d, exp = du.read_summary_file(file)
        exp = exp.drop(columns=['Pulse Number']).to_numpy()
        kpos = np.where(exp[:,1] > 0)
        kneg = np.where(exp[:,1] < 0)
        LTD_data, LTP_data, _, _ = nlf.get_raw_data(exp, kpos, kneg)

        params = {
            'readVoltage': float(readVoltage),
            'twidth': float(d['pulseWidth']),
            'VStartNeg': float(d['VStartNeg (V)']),
            'VEndNeg': float(d['VEndNeg (V)']),
            'VStartPos': float(d['VStartPos (V)']),
            'VEndPos': float(d['VEndPos (V)']),
            'stepSizeLTP': float(d['stepSize']),
            'stepSizeLTD': float(d['stepSize']),
            'NL': 40,
            'best_NL_LTP': float(d['NL_LTP']),
            'best_NL_LTD': float(d['NL_LTD']),
            'sigmaCtoC': sigmaCtoC,
            'sigmaDtoD': sigmaDtoD
        }

        relative_path = os.path.relpath(file, input_dir).replace('_Summary.dat', '')
        output_folder = os.path.join(output_dir, relative_path)
        os.makedirs(output_folder, exist_ok=True)
        config_filename = os.path.join(output_folder, os.path.basename(file.replace('_Summary.dat', '.json')))

        nlf.generate_config(LTD_data,
                            LTP_data,
                            params,
                            filename=config_filename,
                            readPulseWidth=readPulseWidth,
                            numLevelsLTD=numLevelsLTD,
                            numLevelsLTP=numLevelsLTP,
                            ref_config=ref)
        if copy_summary:
            shutil.copy(file, output_folder)
        
if __name__ == '__main__':
    common_parser = argparse.ArgumentParser(add_help=False)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Select sub-command', dest='subcommand')

    # common arguments
    common_parser.add_argument('input_dir', type=str, help='Directory containing the data')
    common_parser.add_argument('output_dir', type=str, help='Directory to save the output')

    # pulse amplitude sweep analysis
    parser_pas = subparsers.add_parser('pas', parents=[common_parser], help='Analyze pulsed amplitude sweep data')
    parser_pas.add_argument('--cutoffs', type=str, help='File containing the cutoffs for the data. Data not present in this file will be ignored')
    parser_pas.set_defaults(func=analyze_pulseAmplitudeSweep)

    # config generator
    parser_conf = subparsers.add_parser('conf', parents=[common_parser], help='Generate configuration files')
    parser_conf.add_argument('--readPulseWidth', type=float, help='Pulse width for read operation (default: 0.0005)', default=0.0005)
    parser_conf.add_argument('--readVoltage', type=float, help='Voltage for read operation (default: 0.1)', default=0.1)
    parser_conf.add_argument('--numLevelsLTD', type=int, help='Number of levels for LTD (default: number of points)')
    parser_conf.add_argument('--numLevelsLTP', type=int, help='Number of levels for LTP (default: number of points)')
    parser_conf.add_argument('--sigmaCtoC', type=float, help='Sigma for C to C (default: 0.05)', default=0.05)
    parser_conf.add_argument('--sigmaDtoD', type=float, help='Sigma for D to D (default: 0.05)', default=0.05)
    parser_conf.add_argument('--copy_summary', action='store_true', help='Copy summary files to output directory')
    parser_conf.add_argument('--ref', type=str, help='Reference file for configuration')
    parser_conf.set_defaults(func=generate_configs)
    
    args = parser.parse_args()
    args.func(args)


    

        
        



