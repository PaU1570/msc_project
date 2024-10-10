import json
import os
import itertools

with open('ltd10ltp10.json', 'r') as file:
    config = json.load(file)

    values = [5, 10, 15, 20, 25, 30]
    for pair in itertools.product(values, repeat=2):
        if pair[0] == 10 and pair[1] == 10:
            continue
        filename = f"ltd{pair[0]}ltp{pair[1]}.json"
        config['device-params']['RealDevice']['maxNumLevelLTD'] = pair[0]
        config['device-params']['RealDevice']['maxNumLevelLTP'] = pair[1]
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)