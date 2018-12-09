import json
import numpy as np
import networkx as nx
from fastText.FastText import load_model

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN


MODEL_PATH = 'emb/karate.emb' # 'emb/CA-AstroPh.emb'
LABELS_PATH = 'labels/karate.json' # 'labels/CA-AstroPh.json'
GRAPH_PATH = 'graph/karate.edgelist' # 'graph/CA-AstroPh.edgelist.txt'
DELIMITER =  ' ' # '\t'

# load the model and mapping
model = load_model(MODEL_PATH)
mapping = {int(k): v for k, v in json.load(open(LABELS_PATH)).items()}

# load graph
G = nx.read_edgelist(GRAPH_PATH, delimiter=DELIMITER, nodetype=int, create_using=nx.DiGraph())
for edge in G.edges():
    G[edge[0]][edge[1]]['weight'] = 1
G = G.to_undirected()

# load structural labels and embeddings on each node
vectors = []
indices = {}
for node in sorted(G):
    indices[len(indices)] = node
    struct_label = mapping[node]
    embedding_vector = model.get_word_vector(struct_label)

    G.node[node]['struct_label'] = struct_label
    G.node[node]['embedding'] = embedding_vector
    vectors.append(embedding_vector)

# build full representation computing normalized dot products
vectors = np.asarray(vectors)
v_norm = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
v_closest = v_norm.argsort(axis=-1)[:, -4:-1][:, ::-1]

# run clustering
km = DBSCAN(eps=0.05, min_samples=1, metric='cosine')
classes = km.fit_predict(vectors)

# check who is closest to who
for i, c in enumerate(zip(v_closest, classes)):
    dist, clust = c
    clust = str(clust)
    l = indices[i]
    print(u'{} ["{}"] is closest to {} in cluster {}'.format(l, mapping[l], u', '.join([str(indices[v]) for v in dist]), clust))

color_string = 'rbgmycw'
colors = [color_string[c] for c in classes]
nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color=colors)
plt.show()
