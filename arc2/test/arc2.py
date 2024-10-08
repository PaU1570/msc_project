from pyarc2 import Instrument, find_ids

# Get the ID of the first available ArC TWO
ids = find_ids()
if (len(ids) == 0):
    print("Error: find_ids() is empty")
    exit(1)
    
arc2id = find_ids()[0]
print("Found arc2id ", arc2id)
# firmware; shipped with your board
fw = '../firmware/efm03_20240418.bin'

# connect to the board
arc = Instrument(arc2id, fw)
print(arc)
