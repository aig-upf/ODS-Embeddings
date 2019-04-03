# -*- coding: utf-8 -*-
import json
import argparse
import numpy as np
import networkx as nx

from graph import read_graph
from fastText.FastText import load_model

from sklearn.cluster import DBSCAN


HTML_COLORS = ["#00ffff", "#7fffd4", "#f5f5dc", "#0000ff", "#8a2be2", "#a52a2a", "#deb887", "#5f9ea0", "#7fff00", "#d2691e", "#ff7f50", "#6495ed", "#fff8dc", "#dc143c", "#00ffff", "#00008b", "#008b8b", "#b8860b", "#a9a9a9", "#006400", "#a9a9a9", "#bdb76b", "#8b008b", "#556b2f", "#ff8c00", "#9932cc", "#8b0000", "#e9967a", "#8fbc8f", "#483d8b", "#2f4f4f", "#2f4f4f", "#00ced1", "#9400d3", "#ff1493", "#00bfff", "#696969", "#696969", "#1e90ff", "#b22222", "#fffaf0", "#228b22", "#ff00ff", "#dcdcdc", "#f8f8ff", "#ffd700", "#daa520", "#808080", "#008000", "#adff2f", "#808080", "#f0fff0", "#ff69b4", "#cd5c5c", "#4b0082", "#fffff0", "#f0e68c", "#e6e6fa", "#fff0f5", "#7cfc00", "#fffacd", "#add8e6", "#f08080", "#e0ffff", "#fafad2", "#d3d3d3", "#90ee90", "#d3d3d3", "#ffb6c1", "#ffa07a", "#20b2aa", "#87cefa", "#778899", "#778899", "#b0c4de", "#ffffe0", "#00ff00", "#32cd32", "#faf0e6", "#ff00ff", "#800000", "#66cdaa", "#0000cd", "#ba55d3", "#9370db", "#3cb371", "#7b68ee", "#00fa9a", "#48d1cc", "#c71585", "#191970", "#f5fffa", "#ffe4e1", "#ffe4b5", "#ffdead", "#000080", "#fdf5e6", "#808000", "#6b8e23", "#ffa500", "#ff4500", "#da70d6", "#eee8aa", "#98fb98", "#afeeee", "#db7093", "#ffefd5", "#ffdab9", "#cd853f", "#ffc0cb", "#dda0dd", "#b0e0e6", "#800080", "#663399", "#ff0000", "#bc8f8f", "#4169e1", "#8b4513", "#fa8072", "#f4a460", "#2e8b57", "#fff5ee", "#a0522d", "#c0c0c0", "#87ceeb", "#6a5acd", "#708090", "#708090", "#fffafa", "#00ff7f", "#4682b4", "#d2b48c", "#008080", "#d8bfd8", "#ff6347", "#40e0d0", "#ee82ee", "#f5deb3", "#ffffff", "#f5f5f5", "#ffff00", "#9acd32"]


def parse_commands():
    main_args = argparse.ArgumentParser()
    main_args.add_argument('-v', '--verbose', help='Verbosity factor. 0 for silent, 1 for verbose.', type=int, default=0)
    main_args.add_argument('-w', '--weighted', help='Flag to specify that the graph is weighted.', action='store_true')
    main_args.add_argument('-D', '--directed', help='Flag to specify that the graph is directed.', action='store_true')
    main_args.add_argument('-c', '--cluster-labels', help='Flag to specify that the clustering must be performed on the graph, otherwise it will be performed on the structural labels themselves.', action='store_true')
    main_args.add_argument('-M', '--model', help='Trained embedding model file.', type=str, required=True)
    main_args.add_argument('-m', '--mapping', help='Mapping file, aliasing every node to its corresponding structural label (or any other alias). Expects a JSON-encoded dictionary (node -> structural label).', type=str, required=True)
    main_args.add_argument('-g', '--graph', help='Input graph to be encoded.', type=str, required=True)
    main_args.add_argument('-t', '--task', help='Evaluation task to perform: {plot,print,closest,all}.', type=str, default='all')
    main_args.add_argument('-e', '--epsilon', help='DBSCAN epsilon.', type=float, default=0.05)
    main_args.add_argument('-s', '--samples', help='DBSCAN samples.', type=int, default=3)

    args = main_args.parse_args()
    return args


def plot_graph(G, classes):
    from igraph import plot
    colors = [HTML_COLORS[c] for c in classes]
    G.vs['color'] = colors
    plot(G, edge_curved=False, layout=G.layout_fruchterman_reingold())


def node_distances(indices, v_norm, classes):
    v_closest = v_norm.argsort(axis=-1)[:, -4:-1][:, ::-1]
    for i, c in enumerate(zip(v_closest, classes)):
        dist, clust = c
        clust = str(clust)
        l = indices[i]
        print(u'{} ["{}"] is closest to {} in cluster {}'.format(l, mapping[l], u', '.join([str(indices[v]) for v in dist]), clust))


if __name__ == "__main__":
    args = parse_commands()

    # load the model and mapping
    model = load_model(args.model)
    mapping = {k: v for k, v in json.load(open(args.mapping)).items()}
    struct_labels = sorted(set(mapping.values()))
    struct_indices = {s: i for i, s in enumerate(struct_labels)}

    # load graph
    G = read_graph(args.graph, args.weighted, args.directed, args.verbose)

    # load structural labels and embeddings on each node
    vectors = []
    indices = {}
    for node in sorted(G.vs):
        indices[len(indices)] = node.index
        struct_label = mapping[node['name']]
        embedding_vector = model.get_word_vector(struct_label)

        node['struct_label'] = struct_label
        node['embedding'] = embedding_vector
        vectors.append(embedding_vector)

    # build full representation computing normalized dot products
    vectors = np.asarray(vectors)
    v_norm = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)

    # run clustering
    km = DBSCAN(eps=args.epsilon, min_samples=args.samples, metric='cosine')
    if args.cluster_labels:
        struct_vectors = np.asarray([model.get_word_vector(l) for l in struct_labels])
        struct_classes = km.fit_predict(struct_vectors)
        struct_mapping = {s: c for s, c in zip(struct_labels, struct_classes)}
        classes = np.asarray([struct_mapping[mapping[n]] for n in sorted(G)])
    else:
        classes = km.fit_predict(vectors)

    # perform the actual task
    task = args.task.lower()
    if task in ['closest', 'all']:
        node_distances(indices, v_norm, classes)
    if task in ['print', 'all']:
        min_c = min(classes)
        for n, c in zip(sorted(G.vs), classes):
            label_index = struct_indices[n['struct_label']]
            print('{} {} {}'.format(n.index, c - (min_c if min_c < 0 else 0), label_index))
    if task in ['plot', 'all']:
        plot_graph(G, classes)

