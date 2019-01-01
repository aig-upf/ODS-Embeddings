# -*- coding: utf-8 -*-
import math
from collections import Counter
from networkx.classes.function import degree
from networkx.generators.ego import ego_graph

from multiprocessing import Pool


def compute_node_degree_counts(G, n, d, include_center):
    G_n = ego_graph(G, n, d)
    deg = G_n.degree()
    cnt = Counter(d for x, d in deg if include_center or n != x)
    cds = sorted(cnt.most_common())
    return cds


def _batch_compute_node_degree_counts(tuples_batch):
    return [(n, compute_node_degree_counts(G, n, d, c)) for (G, n, d, c) in tuples_batch]


def compute_degree_counts(G, d, include_center, num_workers):
    params = [(G, n, d, include_center) for n in G]
    chunks = [params[x::num_workers] for x in range(num_workers)]
    
    workers_pool = Pool(processes=num_workers if num_workers > 1 else 1)
    chunk_cds = workers_pool.map(_batch_compute_node_degree_counts, chunks)
    workers_pool.close()

    all_cds = reduce(lambda x, y: x + y, chunk_cds)
    for n, c in all_cds:
        G.node[n]['cds'] = c
    return G


def compute_global_degree_set(G):
    deg = set()
    for n in G:
        cds = G.node[n]['cds']
        for d, _ in cds:
            deg.add(d)
    return sorted(deg)


def build_degree_table(D, CHARACTER_OFFSET=0x2460):
    table = {}
    for i, d in enumerate(D):
        table[d] = unichr(CHARACTER_OFFSET + i)
    return table


def string_length_log(c):
    return int(round(math.log(c + 1)))


def compute_string_labels(G, d, t, length_func):
    if length_func == 'log':
        compress = string_length_log
    else:
        compress = lambda x: x

    for n in G:
        cds = G.node[n]['cds']
        label = u''.join(t[d] * compress(c) for (d, c) in cds)
        G.node[n]['struct_label'] = label
    return G


def preprocess_graph(G, d, include_center, length_func, num_workers, verbose=0):
    G = compute_degree_counts(G, d, include_center, num_workers)
    D = compute_global_degree_set(G)
    t = build_degree_table(D)
    G = compute_string_labels(G, d, t, length_func)
    m = {n: G.node[n]['struct_label'] for n in G}

    if verbose:
        # keep track of all the preprocessing!
        label_set = set([G.node[n]['struct_label'] for n in G])
        
        print('Total degrees: {}'.format(len(t)))
        print('Total labels: {}'.format(len(label_set)))
        print('Avg. structural label length: {0:.2f}'.format(sum(map(len, label_set)) / float(len(label_set))))
        print('Max. structural label length: {}'.format(max(map(len, label_set))))
    return G, t, m

