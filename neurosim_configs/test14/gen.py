import json
import os

with open('min1max10.json', 'r') as file:
    config = json.load(file)

    for i in range(2, 10):
        filename = f"min1max{i}.json"
        maxConductance = i * 1e-11
        config['device-params']['RealDevice']['maxConductance'] = maxConductance
        config['device-params']['RealDevice']['avgMaxConductance'] = maxConductance
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
