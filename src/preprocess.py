import math
from collections import Counter
from networkx.classes.function import degree
from networkx.generators.ego import ego_graph


def compute_node_degree_counts(G, n, d, include_center):
    G_n = ego_graph(G, n, d)
    deg = G_n.degree()
    cnt = Counter(d for x, d in deg if include_center or n != x)
    cds = sorted(cnt.most_common())
    return cds


def _batch_compute_node_degree_counts(tuples_batch):
    return [compute_node_degree_counts(*t) for t in tuples_batch]


def compute_degree_counts(G, d, include_center, workers_pool):
    workers = workers_pool._processes
    params = [(G, n, d, include_center) for n in G]
    chunks = [params[x::workers] for x in range(workers)]
    chunk_cds = workers_pool.map(_batch_compute_node_degree_counts, chunks)
    all_cds = reduce(lambda x, y: x + y, chunk_cds)
    for n, c in zip(G, all_cds):
        G.node[n]['cds'] = c
    return G


def compute_global_degree_set(G):
    deg = set()
    for n in G:
        cds = G.node[n]['cds']
        for d, _ in cds:
            deg.add(d)
    return sorted(deg)


def build_degree_table(D, CHARACTER_OFFSET=0xA000):
    table = {}
    for i, d in enumerate(D):
        table[d] = unichr(CHARACTER_OFFSET + i)
    return table


def string_length_log(c):
    return int(round(math.log(c + 1)))


def compute_string_labels(G, d, t, log_string_lengths):
    if log_string_lengths:
        compress = string_length_log
    else:
        compress = lambda x: x

    for n in G:
        cds = G.node[n]['cds']
        label = u''.join(t[d] * compress(c) for (d, c) in cds)
        G.node[n]['struct_label'] = label
    return G


def preprocess_graph(G, d, include_center=False, log_string_lengths=True, workers_pool=None):
    G = compute_degree_counts(G, d, include_center, workers_pool)
    D = compute_global_degree_set(G)
    t = build_degree_table(D)
    G = compute_string_labels(G, d, t, log_string_lengths)
    m = {n: G.node[n]['struct_label'] for n in G}

    # keep track of all the preprocessing!
    label_set = set([G.node[n]['struct_label'] for n in G])
    print('Total degrees: {}'.format(len(t)))
    print('Total labels: {}'.format(len(label_set)))
    print('Avg. structural label length: {0:.2f}'.format(sum(map(len, label_set)) / float(len(label_set))))
    print('Max. structural label length: {}'.format(max(map(len, label_set))))
    return G, t, m

