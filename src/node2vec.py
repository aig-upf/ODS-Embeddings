# -*- coding: utf-8 -*-
import random
import numpy as np
from functools import reduce
from multiprocessing import Pool


class Graph():
    def __init__(self, G, is_directed, p, q):
        self.G = G
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
                if alias_nodes is None or alias_edges is None:
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


    def simulate_walks(self, num_walks, walk_length, num_workers):
        '''
        Repeatedly simulate random walks from each node.
        '''
        iters = [(self, walk_length) for x in range(num_walks)]
        workers_pool = Pool(processes=num_workers if num_workers > 1 else 1)
        iters_walks = workers_pool.map(walk_iteration, iters)
        workers_pool.close()
        return reduce(lambda x, y: x + y, iters_walks)


    def get_alias_nodes(self, num_workers):
        '''
        Returns the node to alias mappings.
        '''
        G = self.G
        iters = [node.index for node in G.vs()]
        chunk = [(G, iters[x::num_workers]) for x in range(num_workers)]

        workers_pool = Pool(processes=num_workers if num_workers > 1 else 1)
        chunk_alias = workers_pool.map(get_chunk_alias_nodes, chunk)
        workers_pool.close()
        return {n: a for c in chunk_alias for (n, a) in c}


    def get_alias_edges(self, num_workers):
        '''
        Returns the node to alias mappings.
        '''
        G = self.G
        p = self.p
        q = self.q
        is_directed = self.is_directed

        # find all the edges
        edge_pairs = [edge.tuple for edge in G.es()]
        if not is_directed:
            edge_pairs.extend([(e_2, e_1) for e_1, e_2 in edge_pairs])

        # get the alias edges in parallel and then build the mapping table
        chunk = [(G, p, q, edge_pairs[x::num_workers]) 
                 for x in range(num_workers)]
        workers_pool = Pool(processes=num_workers if num_workers > 1 else 1)
        chunk_alias = workers_pool.map(get_chunk_alias_edges, chunk)
        workers_pool.close()
        return {e: a for c in chunk_alias for (e, a) in c}


    def preprocess_transition_probs(self, num_workers):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        self.alias_nodes = self.get_alias_nodes(num_workers)
        self.alias_edges = self.get_alias_edges(num_workers)
        return


def walk_iteration(arg_tuple):
    '''
    Simulate random walks from each node.
    '''
    n2v_G, walk_length = arg_tuple
    nodes =[n.index for n in n2v_G.G.vs]
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
        unnormalized_probs = [G.es.select(_source=node, _target=nbr)[0].attributes().get('weight', 1)
                              for nbr in sorted(G.neighbors(node))]
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
            weight = G.es.select(_source=dst, _target=dst_nbr)[0].attributes().get('weight', 1)
            if dst_nbr == src:
                unnormalized_probs.append(weight/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(weight)
            else:
                unnormalized_probs.append(weight/q)

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
    normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
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

