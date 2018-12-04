'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import io
import os
import json
import argparse
import numpy as np
import networkx as nx
import node2vec
from fastText.FastText import train_unsupervised


def parse_args():
    '''
    Parses the structural embeddings arguments.
    '''
    parser = argparse.ArgumentParser(description="Run structural graph embeddings.")

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--delimiter', nargs='?', default=' ',
                        help='Input graph delimiter')

    parser.add_argument('--labelfile', nargs='?', default='labels/karate.json',
                        help='Label file to save/read from.')

    parser.add_argument('--walkfile', nargs='?', default='walk/karate.walk',
                        help='Random walk dump path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=3,
                        help='Context size for optimization. Default is 3.')

    parser.add_argument('--iter', default=10, type=int,
                      help='Number of epochs')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--minn', type=int, default=1,
                        help='Minimum ordered degree sequence ngram size. Default is 1.')

    parser.add_argument('--maxn', type=int, default=2,
                        help='Minimum ordered degree sequence ngram size. Default is 2.')

    parser.add_argument('--structdist', type=int, default=1,
                        help='Distance of the per-node ego networks. Default is 1.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')

    parser.add_argument('--center', dest='center', action='store_true',
                        help='Boolean specifying if the degree of the ego network source is used. Default is false.')

    parser.add_argument('--intlengths', dest='intlengths', action='store_true',
                        help='Boolean specifying if the lengths of the degrees in the ordered degree sequences should be int instead of logarithm. Default is false.')

    parser.add_argument('--nowalk', dest='nowalk', action='store_true',
                        help='Boolean specifying no need for walking. Default is walk.')

    parser.add_argument('--nostruct', dest='nostruct', action='store_true',
                        help='Boolean specifying no need for structural labelling. Default is structural distance computation.')

    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')

    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()

def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, delimiter=args.delimiter, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, delimiter=args.delimiter, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    print('Total nodes: {}'.format(len(G)))
    print('Total edges: {}'.format(len(G.edges())))
    return G


def learn_embeddings(args):
    '''
    Learn embeddings by optimizing the Skipgram objective calling FastText.
    '''
    model = train_unsupervised(args.walkfile, dim=args.dimensions, ws=args.window_size, thread=args.workers, epoch=args.iter, minn=args.minn, maxn=args.maxn, minCount=0)
    model.save_model(args.output)
    return model


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph()
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()

    if not args.nowalk:
        walks = G.simulate_walks(args.num_walks, args.walk_length, args.workers)

        # load structural labels if ever existing
        if args.labelfile and os.path.exists(args.labelfile):
            m = json.load(open(args.labelfile, 'r'))

        # generate structural labels
        if not args.nostruct:
            m = G.compute_structural_labels(args.structdist, args.center, args.workers, not args.intlengths)

            # dump file if necessary
            if args.labelfile:
                with open(args.labelfile, 'w') as f:
                    json.dump(m, f)

        # dump the walks
        with io.open(args.walkfile, 'w', encoding='utf8') as f:
            for walk in walks:
                f.write(u' '.join(m[n] for n in walk) + u'\n')
    model = learn_embeddings(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
