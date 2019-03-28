'''Loads the community file for Youtube and outputs a json multilabel dictionary.
'''
import sys
import json
from collections import defaultdict

TARGET_PATH = sys.argv[1]

# load the Youtube mapping
node_mappings = defaultdict(list)
with open(TARGET_PATH + 'Youtube/Youtube.cmty', 'r') as f:
    for cmty, line in enumerate(f):
        for node in line.strip().split('\t'):
        	node_mappings[node].append(cmty)

with open(TARGET_PATH + 'Youtube/Youtube.json', 'w') as f:
    json.dump(node_mappings, f)
