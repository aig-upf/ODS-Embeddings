import sys
import json

import igraph
import scipy.io

TARGET_PATH = sys.argv[1]
mat = scipy.io.loadmat(TARGET_PATH + '/CoCit.mat')
G = igraph.Graph(list(zip(*mat['network'].nonzero())))
l = {i: int(v) for i, v in enumerate(mat['group'].argmax(axis=-1).A.flatten())}
G.write_ncol(TARGET_PATH + '/CoCit.edgelist', None)
with open(TARGET_PATH + '/CoCit-labels.json', 'w') as f:
    json.dump(l, f)

