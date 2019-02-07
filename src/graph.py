import random
import networkx as nx


def read_graph(graph_path, separator, weighted, directed, verbose):
    '''
    Reads the input network in networkx.
    '''
    if weighted:
        G = nx.read_edgelist(graph_path, delimiter=separator, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(graph_path, delimiter=separator, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    if verbose:
        print('Total nodes: {}'.format(G.number_of_nodes()))
        print('Total edges: {}'.format(G.number_of_edges()))
    return G


def sample_edges(G, percentage, connected=False, seed=None):
    '''
    Samples edges from a given graph to create a sub-graph, 
    ensuring connectivity if needed.
    '''
    edges = list(G.edges())
    new_n = int(round(G.number_of_edges() * (1 - percentage)))
    
    if seed is not None:
        random.seed(seed)
    random.shuffle(edges)

    new_edges = []
    sub_G = G.copy()
    for e in edges:
        e_1, e_2 = e
        data = G[e_1][e_2]
        sub_G.remove_edge(e_1, e_2)

        reachable = nx.connected._plain_bfs(sub_G, e[0])
        if connected and e[1] not in reachable:
            sub_G.add_edge(e_1, e_2, **data)
        else:
            new_edges.append(e)

        if len(new_edges) == new_n:
            break
    return sub_G


def edge_complement(G, sub_G):
    '''
    Given a full graph and a sub-graph of it, 
    return graph with all nodes NOT in the sub-graph.
    '''
    sub_edges = set(sub_G.edges())
    new_edges = [e for e in G.edges() if e not in sub_edges]
    compl_G = G.edge_subgraph(new_edges)
    return compl_G
