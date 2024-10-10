import nonlinear_fit as nlf
import data_utils as du
import argparse
import pandas as pd
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze pulsed amplitude sweep data from a directory')
    parser.add_argument('directory', type=str, help='Directory containing the data')
    parser.add_argument('output_dir', type=str, help='Directory to save the output')
    parser.add_argument('--cutoffs', type=str, help='File containing the cutoffs for the data. Data not present in this file will be ignored')
    args = parser.parse_args()

    input_dir = args.directory
    output_dir = args.output_dir
    cutoffs_file = args.cutoffs

    # get all files in the directory
    csv_files = du.get_csv_files(input_dir)

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

        stepSize = meas_params['stepSize']
        twidth = meas_params['pulseWidth']

        relative_path = os.path.relpath(file, input_dir)
        output_file = os.path.join(output_dir, relative_path)

        nlf.analyze(file,
                    output_file,
                    cutoffs=(VStartPos, VEndPos, VStartNeg, VEndNeg),
                    plotmode=2,
                    savesummary=True)

        
        



