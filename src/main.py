# -*- coding: utf-8 -*-
'''
Reference implementation of YNSAN-E. 

Author: Nur Álvarez González

Based on the node2vec reference implementation
by Aditya Grover:

  https://github.com/aditya-grover/node2vec
'''

import io
import os
import sys
import json
import random
import argparse
import numpy as np
import networkx as nx
from fastText.FastText import train_unsupervised, load_model

import node2vec
from preprocess import preprocess_graph


def parse_commands():
    main_args = argparse.ArgumentParser()
    main_args.add_argument('-v', '--verbose', help='Verbosity factor. 0 for silent, 1 for verbose.', type=int, default=1)
    main_args.add_argument('-t', '--threads', help='Number of parallel workers. Default is 8.', type=int, default=8)
    main_subs = main_args.add_subparsers(title='Subcommands', description='Available commands to execute the different YNSAN-E steps.', dest='task')

    graph_args = argparse.ArgumentParser(add_help=False)
    graph_args.add_argument('-g', '--graph', help='Input graph to be encoded.', type=str, required=True)
    graph_args.add_argument('-S', '--separator', help='Separator for the fields in the edgelist file.', type=str, default=' ')
    graph_args.add_argument('-w', '--weighted', help='Flag to specify that the graph is weighted.', action='store_true')
    graph_args.add_argument('-D', '--directed', help='Flag to specify that the graph is directed.', action='store_true')

    enc_args = main_subs.add_parser('encode', description='Structural label generation for graph nodes.', parents=[graph_args])
    enc_args.add_argument('-c', '--center', help='Keep the degree of the ego network source in the ordered degree sequence.', action='store_true')
    enc_args.add_argument('-d', '--distance', help='Distance of the per-node induced ego-networks.', type=int, default=2)
    enc_args.add_argument('-f', '--function', help='Encoding function to use, from either "int" or "log".', default='log')
    enc_args.add_argument('-o', '--output', help='Output file to store the structural mapping associated of every node, in JSON.', type=str, required=True)

    walk_args = main_subs.add_parser('walk', description='Random walk generation from graphs.', parents=[graph_args])
    walk_args.add_argument('-n', '--num-walks', help='Number of random walks to start from each node.', type=int, default=50)
    walk_args.add_argument('-l', '--walk-length', help='Length of each random walk.', type=int, default=10)
    walk_args.add_argument('-p', '--p', help='Return hyperparameter for node2vec. Default is 1.', type=int, default=1)
    walk_args.add_argument('-q', '--q', help='Inout hyperparameter for node2vec. Default is 1.', type=int, default=1)
    walk_args.add_argument('-m', '--mapping', help='Mapping file, aliasing every node to its corresponding structural label (or any other alias).', type=str, default='')
    walk_args.add_argument('-o', '--output', help='Output file to store generated random walks.', type=str, required=True)

    embed_args = main_subs.add_parser('embed', description='Network Embedding training from Random Walks.')
    embed_args.add_argument('-c', '--context', help='Context size for optimization. Default is 3.', type=int, default=3)
    embed_args.add_argument('-e', '--epochs', help='Number of epochs. If set to 0, no training is performed. Default is 100.', type=int, default=100)
    embed_args.add_argument('-m', '--minn', help='Minimum ordered degree sequence ngram size. Default is 1.', type=int, default=1)
    embed_args.add_argument('-M', '--maxn', help='Maximum ordered degree sequence ngram size. Default is 1.', type=int, default=2)
    embed_args.add_argument('-d', '--dimensions', help='Dimensionality of the embedding vectors.', type=int, default=128)
    embed_args.add_argument('-l', '--learning-rate', help='Learning rate.', type=float, default=0.005)
    embed_args.add_argument('-w', '--walk', help='Input walk file to train the model on.', type=str, required=True)
    embed_args.add_argument('-o', '--output', help='Output file for the trained embedding model.', type=str, required=True)

    sample_args = main_subs.add_parser('sample', description='Graph edge sampling for link prediction.', parents=[graph_args])
    sample_args.add_argument('-p', '--percentage', help='Percentage of the graph to sample.', type=float, default=0.8)
    sample_args.add_argument('-s', '--seed', help='Seed used for sampling.', type=int, default=None)
    sample_args.add_argument('-o', '--output', help='Output file to store generated graph sample.', type=str, required=True)
    sample_args.add_argument('-c', '--complement', help='Complimentary output file to store the generated graph sample with the non-sampled edges, useful for link prediction.', type=str, default='')

    args = main_args.parse_args()
    return args


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


def encode_command(G, args):    
    if args.verbose:
        print('Generating structural labels.')

    mapping = preprocess_graph(G, 
                               args.distance, 
                               args.center, 
                               args.function, 
                               args.threads, 
                               args.verbose)[-1]

    with open(args.output, 'w') as f:
        json.dump(mapping, f)

    if args.verbose:
        print('Saved structural labels in "{}".'.format(args.output))

    sys.exit(0)


def sample_command(G, args):    
    if args.verbose:
        print('Sampling edges to create subgraph.')

    edges = list(G.edges())
    new_n = int(round(G.number_of_edges() * args.percentage))
    
    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(edges)

    new_edges = edges[:new_n]
    sub_G = G.edge_subgraph(new_edges)
    nx.write_edgelist(sub_G, args.output, data=args.weighted)

    if args.verbose:
        print('Saved edge-sampled graph in "{}" with {} edges out of {}.'.format(args.output, new_n, len(edges)))

    if args.complement:
        compl_n = G.number_of_edges() - new_n
        compl_edges = edges[new_n:]
        compl_G = G.edge_subgraph(compl_edges)
        nx.write_edgelist(compl_G, args.complement, data=args.weighted)

        if args.verbose:
            print('Saved complement edge-sampled graph in "{}" with {} edges out of {}.'.format(args.complement, compl_n, len(edges)))

    sys.exit(0)


def walk_command(G, args):
    G = node2vec.Graph(G, args.directed, args.p, args.q)
    if args.p != 1 or args.q != 1:
        if args.verbose:
            print('Preprocessing transition probabilities.')
        G.preprocess_transition_probs(args.threads)

    if args.verbose:
        print('Generating random walks.')
    walks = G.simulate_walks(args.num_walks, args.walk_length, args.threads)

    if args.mapping:
        mapping = {int(k): v for (k, v) in json.load(open(args.mapping, 'r')).items()}
    else:
        mapping = {n: str(n) for n in G}

    # dump the walks
    with io.open(args.output, 'w', encoding='utf8') as f:
        for walk in walks:
            f.write(u' '.join(mapping[n] for n in walk) + u'\n')

    if args.verbose:
        print('Saved random walks in "{}".'.format(args.output))

    sys.exit(0)


def embed_command(args):    
    if args.verbose:
        print('Training model from the random walks in "{}".'.format(args.walk))

    model = train_unsupervised(args.walk, 
                               dim=args.dimensions, 
                               ws=args.context, 
                               thread=args.threads, 
                               epoch=args.epochs, 
                               minn=args.minn, 
                               maxn=args.maxn, 
                               lr=args.learning_rate, 
                               verbose=args.verbose,
                               minCount=0)
    model.save_model(args.output)
    
    if args.verbose:
        print('Saved FastText model in "{}".'.format(args.output))

    sys.exit(0)


def main(args):
    if args.task == 'embed':
        embed_command(args)

    G = read_graph(args.graph, args.separator, args.weighted, args.directed, args.verbose)
    if args.task == 'encode':
        encode_command(G, args)

    if args.task == 'sample':
        sample_command(G, args)

    if args.task == 'walk':
        walk_command(G, args)


if __name__ == "__main__":
    args = parse_commands()
    main(args)
    
