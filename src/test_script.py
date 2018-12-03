from fastText import FastText

import networkx as nx
from preprocess import preprocess_graph


# load a graph -- random for testing here
G = nx.fast_gnp_random_graph(2000, 0.005)
print('Total number of nodes: {}'.format(len(G)))
print('Total number of edges: {}'.format(len(G.edges())))

# preprocess the graph
d = 2
G, t = preprocess_graph(G, d, num_processes=8)
field_name = 'struct_label_{}'.format(d)

# check the cardinality of the structural label set
label_set = set([G.node[n][field_name] for n in G])
print('Total labels: {}'.format(len(label_set)))
print('Max. structural label length: {}'.format(max(map(len, label_set))))

# random walk
