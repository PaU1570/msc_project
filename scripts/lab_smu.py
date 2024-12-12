import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

plt.style.use('ggplot')

def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    
def plot_device(filename):
    data = read_json(filename)
    device_id = data["device_id"]
    timestamp = data["timestamp"]
    voltages = data["data"]["sweep_voltages"]
    currents = data["data"]["sweep_currents"]

    fig, ax = plt.subplots()
    ax.plot(voltages, np.abs(currents))
    ax.set(xlabel="Voltage (V)", ylabel="Current (A)", title=f"Device ID {device_id}", yscale="log")
    plt.savefig(f"results/pauv02/{timestamp}_{device_id}.png")
    
if __name__ == '__main__':
    dir = sys.argv[1]

    for filename in os.listdir(dir):
        if filename.endswith(".json"):
            plot_device(f"{dir}/{filename}")