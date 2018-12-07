import numpy as np
import networkx as nx
from preprocess import preprocess_graph
import random


class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.t = {}
        self.alias_nodes = None
        self.alias_edges = None


    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        p = self.p
        q = self.q
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))

            if len(cur_nbrs) > 0:
                if p == 1 and q == 1:
                    index = random.randint(0, len(cur_nbrs) - 1)
                    walk.append(cur_nbrs[index])
                elif len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
                        alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk


    def simulate_walks(self, num_walks, walk_length, workers_pool):
        '''
        Repeatedly simulate random walks from each node.
        '''
        iters = [(self, walk_length) for x in range(num_walks)]
        iters_walks = workers_pool.map(walk_iteration, iters)
        return reduce(lambda x, y: x + y, iters_walks)


    def get_alias_nodes(self, workers_pool):
        '''
        Returns the node to alias mappings.
        '''
        G = self.G
        workers = workers_pool._processes
        iters = [node for node in G.nodes()]
        chunk = [(G, iters[x::workers]) for x in range(workers)]
        chunk_alias = workers_pool.map(get_chunk_alias_nodes, chunk)
        return {n: a for c in chunk_alias for (n, a) in c}


    def get_alias_edges(self, workers_pool):
        '''
        Returns the node to alias mappings.
        '''
        G = self.G
        p = self.p
        q = self.q
        is_directed = self.is_directed

        # find all the edges
        edge_pairs = [(edge[0], edge[1]) for edge in G.edges()]
        if not is_directed:
            edge_pairs.extend([(edge[1], edge[0]) for edge in G.edges()])

        # get the alias edges in parallel and then build the mapping table
        workers = workers_pool._processes
        chunk = [(G, p, q, edge_pairs[x::workers]) for x in range(workers)]
        chunk_alias = workers_pool.map(get_chunk_alias_edges, chunk)
        return {e: a for c in chunk_alias for (e, a) in c}


    def preprocess_transition_probs(self, workers_pool):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        self.alias_nodes = self.get_alias_nodes(workers_pool)
        self.alias_edges = self.get_alias_edges(workers_pool)
        return


    def compute_structural_labels(self, d, include_center, log_string_lengths, workers_pool):
        '''
        Computes structural labels at distance d for every node, caching as necessary
        '''
        G, t, m = preprocess_graph(self.G, d, include_center, log_string_lengths, workers_pool)
        return m


def walk_iteration(arg_tuple):
    '''
    Simulate random walks from each node.
    '''
    n2v_G, walk_length = arg_tuple
    nodes = list(n2v_G.G.nodes())
    random.shuffle(nodes)
    return [n2v_G.node2vec_walk(walk_length=walk_length, start_node=node)
            for node in nodes]


def get_chunk_alias_nodes(data):
    '''
    Get alias setups in batched chunks.
    '''
    result = []
    G, nodes = data
    for node in nodes:
        unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
        result.append((node, alias_setup(normalized_probs)))
    return result


def get_chunk_alias_edges(data):
    '''
    Get the alias edge setup lists for a given batch of edges.
    '''
    G, p, q, chunk = data
    results = []
    for (src, dst) in chunk:
        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)

        # normalize probabilities and set up the sampling alias tables
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
        as_alias = alias_setup(normalized_probs)

        edge = (src, dst)
        results.append((edge, as_alias))
    return results


def normalize_probs(unnormalized_probs):
    '''
    Normalize the probabilities produced by the alias edge procedure.
    '''
    norm_const = sum(unnormalized_probs)
    normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
    return alias_setup(normalized_probs)


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

