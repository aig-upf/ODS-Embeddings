'''Loads a JSON-encoded NetworkX graph and returns an edgelist

Used to convert GraphSAGE experiments (PPI and Reddit) into edgelists.
'''

import json
import networkx as nx
from networkx.readwrite import json_graph


# load the PPI graph and prepare the split subgraph
with open('PPI/ppi-G.json', 'r') as f:
    G_json = json.load(f)
    G = json_graph.node_link_graph(G_json)
    
    N_train = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    N_valid = [n for n in G.nodes() if G.node[n]['val']]
    N_test  = [n for n in G.nodes() if G.node[n]['test']]

    nx.write_edgelist(G.subgraph(N_train), 'PPI/ppi-train.edgelist', delimiter=' ', data=False)
    nx.write_edgelist(G.subgraph(N_valid), 'PPI/ppi-valid.edgelist', delimiter=' ', data=False)
    nx.write_edgelist(G.subgraph(N_test), 'PPI/ppi-test.edgelist', delimiter=' ', data=False)


# load the PPI graph and prepare the split subgraph
with open('Reddit/reddit-G.json', 'r') as f:
    with open('Reddit/reddit-id_map.json', 'r') as m:
        mapping = json.load(m)
    G_json = json.load(f)
    G_json['nodes'] = [{'test': n['test'], 'id': mapping[n['id']], 'val': n['val']} for n in G_json['nodes']]
    G = json_graph.node_link_graph(G_json)
    
    N_train = [n['id'] for n in G_json['nodes'] if not n['val'] and not n['test']]
    N_valid = [n['id'] for n in G_json['nodes'] if n['val']]
    N_test  = [n['id'] for n in G_json['nodes'] if n['test']]

    nx.write_edgelist(G.subgraph(N_train), 'Reddit/reddit-train.edgelist', delimiter=' ', data=False)
    nx.write_edgelist(G.subgraph(N_valid), 'Reddit/reddit-valid.edgelist', delimiter=' ', data=False)
    nx.write_edgelist(G.subgraph(N_test), 'Reddit/reddit-test.edgelist', delimiter=' ', data=False)

