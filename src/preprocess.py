# -*- coding: utf-8 -*-
import math
import numpy as np
from collections import Counter, defaultdict
from networkx.classes.function import degree
from networkx.generators.ego import ego_graph

from multiprocessing import Pool


def compute_node_degree_counts(G, n, d, include_center):
    G_n = G.induced_subgraph(G.neighborhood(n, order=d))
    deg = G_n.degree()
    n_d = zip(G_n.vs, deg)
    cnt = Counter(d for x, d in n_d if include_center or n != x.index)
    cds = sorted(cnt.most_common())
    return cds


def _batch_compute_node_degree_counts(tuples_batch):
    return [(n, compute_node_degree_counts(G, n, d, c)) for (G, n, d, c) in tuples_batch]


def compute_degree_counts(G, d, include_center, num_workers):
    params = [(G, n.index, d, include_center) for n in G.vs]
    chunks = [params[x::num_workers] for x in range(num_workers)]
    
    workers_pool = Pool(processes=num_workers if num_workers > 1 else 1)
    chunk_cds = workers_pool.map(_batch_compute_node_degree_counts, chunks)
    workers_pool.close()

    all_cds = reduce(lambda x, y: x + y, chunk_cds)
    for n, c in all_cds:
        G.vs[n]['cds'] = c
    return G


def compute_global_degree_frequencies(G):
    total_counts = defaultdict(int)
    for n in G.vs:
        for (d, t) in n['cds']:
            total_counts[d] += t
    return sorted(total_counts.items())


def degree_bucketizer(D, num_buckets=50, bucket_fn='id'):
    degree_fn = lambda f: f if bucket_fn == 'id' else lambda f: int(round(math.log(f, 2) + 1))
    degree_logfreq = [i for i, f in D for _ in range(degree_fn(f))] 
    if num_buckets > 0:
        bucket_percentiles = [i * (100.0 / num_buckets) for i in range(1, num_buckets)]
        bucket_list = np.unique(np.percentile(degree_logfreq, bucket_percentiles).round())
    else:
        bucket_list = np.unique(degree_logfreq)
    bucket_list.sort()
    return bucket_list


def build_degree_table(D, B, CHARACTER_OFFSET=0x2460):
    return {d: unichr(CHARACTER_OFFSET + B.searchsorted(d)) for d, f in D}


def string_length_log(c):
    return int(round(math.log(c + 1)))


def compute_string_labels(G, d, t, length_fn):
    if length_fn == u'log':
        compress = string_length_log
    else:
        compress = lambda x: x

    for n in G.vs:
        cds = n['cds']
        label = u''.join(t[d] * compress(c) for (d, c) in cds)
        n['struct_label'] = label
    return G


def preprocess_graph(G, d, include_center=True, num_buckets=0, bucket_fn='id', length_fn='log', num_workers=8, verbose=0):
    G = compute_degree_counts(G, d, include_center, num_workers)
    D = compute_global_degree_frequencies(G)
    B = degree_bucketizer(D, num_buckets, bucket_fn)
    t = build_degree_table(D, B)
    G = compute_string_labels(G, d, t, length_fn)
    m = {n['name']: n['struct_label'] for n in G.vs}

    if verbose:
        # keep track of all the preprocessing!
        label_set = set([n['struct_label'] for n in G.vs])
        
        print('Total degrees: {}'.format(len(t)))
        print('Total labels: {}'.format(len(label_set)))
        print('Avg. structural label length: {0:.2f}'.format(sum(map(len, label_set)) / float(len(label_set))))
        print('Max. structural label length: {}'.format(max(map(len, label_set))))
    return G, B, t, m

