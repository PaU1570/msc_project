import json
import os
import itertools

with open('min9max8.json', 'r') as file:
    config = json.load(file)

    values = list(range(6, 14))
    for pair in itertools.product(values, repeat=2):
        # pair[0]: min, pair[1]: max
        if (pair[0] == 8 and pair[1] == 9) or (pair[0] <= pair[1]):
            continue
        filename = f"min{pair[0]}max{pair[1]}.json"
        minConductance = 10 ** (-pair[0])
        maxConductance = 10 ** (-pair[1])
        config['device-params']['RealDevice']['minConductance'] = minConductance
        config['device-params']['RealDevice']['avgMinConductance'] = minConductance
        config['device-params']['RealDevice']['maxConductance'] = maxConductance
        config['device-params']['RealDevice']['avgMaxConductance'] = maxConductance
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
