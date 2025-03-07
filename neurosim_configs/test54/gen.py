import json
import os
import itertools

with open('ltd30ltp30.json', 'r') as file:
    config = json.load(file)

    values = [40, 50, 60, 70, 80]
    for val in values:
        filename = f"ltd30ltp{val}.json"
        config['device-params']['RealDevice']['maxNumLevelLTD'] = 30
        config['device-params']['RealDevice']['maxNumLevelLTP'] = val
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
        filename = f"ltd{val}ltp30.json"
        config['device-params']['RealDevice']['maxNumLevelLTP'] = 30
        config['device-params']['RealDevice']['maxNumLevelLTD'] = val
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)