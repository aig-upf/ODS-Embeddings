import random
import igraph as ig


def read_graph(graph_path, weighted=False, directed=False, verbose=1):
    '''
    Reads the input network in networkx.
    '''
    G = ig.Graph.Read_Ncol(graph_path, names=True, weights=weighted, directed=directed).simplify()

    if verbose:
        print('Total nodes: {}'.format(len(G.vs)))
        print('Total edges: {}'.format(len(G.es)))
    return G


def sample_edges(G, percentage, connected=False, seed=None):
    '''
    Samples edges from a given graph to create a sub-graph, 
    ensuring connectivity if needed.
    '''
    new_n = int(round(len(G.es) * (1 - percentage)))
    if seed is not None:
        random.seed(seed)

    sub_G = G.copy()
    if not connected:
        edges = list(sub_G.es)
        random.shuffle(edges)
        sub_G.delete_edges([e for e in edges[:new_n]])
        return sub_G

    edges_count = 0
    seen_edges  = set()
    total_edges = len(G.es)
    while edges_count < new_n and len(seen_edges) < total_edges:
        e = random.choice(sub_G.es)
        attributes = e.attributes()
        e_1, e_2 = e.tuple
        if (e_1, e_2) in seen_edges or not sub_G.is_directed() and (e_2, e_1) in seen_edges:
            continue

        e.delete()
        reachable = False
        for x in sub_G.bfsiter(e_1):
            if x.index == e_2:
                reachable = True
                break

        seen_edges.add((e_1, e_2))
        if connected and not reachable and not sub_G.are_connected(e_1, e_2):
            sub_G.add_edge(e_1, e_2, **attributes)
        else:
            edges_count += 1
    return sub_G


def edge_complement(G, sub_G):
    '''
    Given a full graph and a sub-graph of it, 
    return graph with all nodes NOT in the sub-graph.
    '''
    compl_G = G.copy()
    compl_names = compl_G.vs['name']
    compl_indices = {n: i for i, n in enumerate(compl_names)}
    sub_names = {n.index: compl_indices[n['name']] for n in sub_G.vs}
    compl_G.delete_edges([tuple(sub_names[v] for v in e.tuple) for e in sub_G.es])
    return compl_G
