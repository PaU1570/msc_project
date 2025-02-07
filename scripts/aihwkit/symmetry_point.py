import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import msc_project.utils.asymmetric_pulsing as ap
import msc_project.utils.data_utils as du
import msc_project.utils.fit_piecewise as fp

if __name__ == '__main__':
    plt.style.use('ggplot')

    parser = argparse.ArgumentParser(description='Plot symmetry point')
    parser.add_argument('input', type=str, help='Input Summary.dat file')
    parser.add_argument('--output', type=str, help='Output directory')

    args = parser.parse_args()

    summary = du.read_summary_file(args.input)[0]
    _, device_config, model_response = fp.get_fit(args.input)

    ax, sp = ap.plot_symmetry_point(device_config, n_steps=51, pre_alternating_pulses=0, alternating_pulses=150, w_init=0, marker='o')
    ax.set(title=f"Symmetry point {summary['device_id']} - {summary['test_time']}", xlabel='Pulse number', ylabel='Weight [conductance]')
    ax.legend()

    if args.output:
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, f'symmetry_point_{sp:.2f}.png'))
    else:
        plt.show()