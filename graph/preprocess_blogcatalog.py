'''Loads the community file for BlogCatalog and outputs a json multilabel dictionary.
'''
import sys
import json
from collections import defaultdict

TARGET_PATH = sys.argv[1]

# load the blog catalog mapping
node_mappings = defaultdict(list)
with open(TARGET_PATH + 'BlogCatalog/BlogCatalog.cmty', 'r') as f:
    for line in f:
        node, cmty = line.strip().split(',')
        node_mappings[node].append(int(cmty))

with open(TARGET_PATH + 'BlogCatalog/BlogCatalog.json', 'w') as f:
    json.dump(node_mappings, f)
