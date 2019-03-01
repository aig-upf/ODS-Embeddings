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
import igraph as ig


import node2vec
from graph import read_graph, sample_edges, edge_complement
from preprocess import preprocess_graph


def parse_commands():
    main_args = argparse.ArgumentParser()
    main_args.add_argument('-v', '--verbose', help='Verbosity factor. 0 for silent, 1 for verbose.', type=int, default=1)
    main_args.add_argument('-t', '--threads', help='Number of parallel workers. Default is 8.', type=int, default=8)
    main_subs = main_args.add_subparsers(title='Subcommands', description='Available commands to execute the different YNSAN-E steps.', dest='task')

    graph_args = argparse.ArgumentParser(add_help=False)
    graph_args.add_argument('-g', '--graph', help='Input graph to be encoded.', type=str, required=True)
    graph_args.add_argument('-w', '--weighted', help='Flag to specify that the graph is weighted.', action='store_true')
    graph_args.add_argument('-D', '--directed', help='Flag to specify that the graph is directed.', action='store_true')

    enc_args = main_subs.add_parser('encode', description='Structural label generation for graph nodes.', parents=[graph_args])
    enc_args.add_argument('-c', '--center', help='Keep the degree of the ego network source in the ordered degree sequence.', action='store_true')
    enc_args.add_argument('-d', '--distance', help='Distance of the per-node induced ego-networks.', type=int, default=1)
    enc_args.add_argument('-b', '--bucket_function', help='Bucketing function to apply to the degree frequences, between "id" or "log".', default='id')
    enc_args.add_argument('-B', '--buckets', help='Number of buckets to use, or 0 to use all available values.', default=0)
    enc_args.add_argument('-p', '--buckets_path', help='Path to store the bucketing array, if needed.', default='')
    enc_args.add_argument('-f', '--comp_function', help='Encoding function to use, from either "int" or "log".', default='log')
    enc_args.add_argument('-o', '--output', help='Output file to store the structural mapping associated of every node, in JSON (node -> structural label).', type=str, required=True)

    walk_args = main_subs.add_parser('walk', description='Random walk generation from graphs.', parents=[graph_args])
    walk_args.add_argument('-n', '--num-walks', help='Number of random walks to start from each node.', type=int, default=8)
    walk_args.add_argument('-l', '--walk-length', help='Length of each random walk.', type=int, default=15)
    walk_args.add_argument('-p', '--p', help='Return hyperparameter for node2vec. Default is 1.', type=int, default=1)
    walk_args.add_argument('-q', '--q', help='Inout hyperparameter for node2vec. Default is 1.', type=int, default=1)
    walk_args.add_argument('-m', '--mapping', help='Mapping file, aliasing every node to its corresponding structural label (or any other alias). Expects a JSON-encoded dictionary (node -> structural label).', type=str, default='')
    walk_args.add_argument('-o', '--output', help='Output file to store generated random walks.', type=str, required=True)

    embed_args = main_subs.add_parser('embed', description='Network Embedding training from Random Walks.')
    embed_args.add_argument('-a', '--algorithm', help='Embedding algorithm to use out of \{fasttext,word2vec\}. Default is "fasttext".', type=str, default='fasttext')
    embed_args.add_argument('-c', '--context', help='Context size for optimization. Default is 2.', type=int, default=2)
    embed_args.add_argument('-e', '--epochs', help='Number of epochs. If set to 0, no training is performed. Default is 25.', type=int, default=25)
    embed_args.add_argument('-m', '--minn', help='Minimum ordered degree sequence ngram size. Default is 1.', type=int, default=1)
    embed_args.add_argument('-M', '--maxn', help='Maximum ordered degree sequence ngram size. Default is 1.', type=int, default=2)
    embed_args.add_argument('-d', '--dimensions', help='Dimensionality of the embedding vectors.', type=int, default=32)
    embed_args.add_argument('-l', '--learning-rate', help='Learning rate.', type=float, default=0.1)
    embed_args.add_argument('-w', '--walk', help='Input walk file to train the model on.', type=str, required=True)
    embed_args.add_argument('-o', '--output', help='Output file for the trained embedding model.', type=str, required=True)

    sample_args = main_subs.add_parser('sample', description='Graph edge sampling for link prediction.', parents=[graph_args])
    sample_args.add_argument('-C', '--connected', help='Whether or not to ensure that the graph remains connected after removing an edge.', action='store_true')
    sample_args.add_argument('-p', '--percentage', help='Percentage of the graph to sample.', type=float, default=0.5)
    sample_args.add_argument('-s', '--seed', help='Seed used for sampling.', type=int, default=None)
    sample_args.add_argument('-o', '--output', help='Output file to store generated graph sample.', type=str, required=True)
    sample_args.add_argument('-c', '--complement', help='Complimentary output file to store the generated graph sample with the non-sampled edges, useful for link prediction.', type=str, default='')

    args = main_args.parse_args()
    return args


def encode_command(G, args):    
    if args.verbose:
        print('Generating structural labels.')

    G, B, t, m = preprocess_graph(G, 
                               args.distance, 
                               args.center, 
                               args.buckets,
                               args.bucket_function,
                               args.comp_function, 
                               args.threads, 
                               args.verbose)

    if args.buckets_path:
        with open(args.buckets_path, 'w') as f:
            json.dump([int(v) for v in B], f)

    with open(args.output, 'w') as f:
        json.dump(m, f)

    if args.verbose:
        print('Saved structural labels in "{}".'.format(args.output))

    sys.exit(0)


def sample_command(G, args):    
    if args.verbose:
        print('Sampling edges to create subgraph.')

    sub_G = sample_edges(G, args.percentage, args.connected, args.seed)
    sub_G.write_ncol(args.output, weights='weight' if args.weighted else None)

    if args.verbose:
        sub_edges = len(sub_G.es)
        orig_edges = len(G.es)
        pct_edges = (100.0 * sub_edges) / orig_edges 
        print('Saved edge-sampled graph in "{}" with {} edges out of {} ({:.2f}%%).'.format(args.output, sub_edges, orig_edges, pct_edges))

    if args.complement:
        compl_G = edge_complement(G, sub_G)
        compl_G.write_ncol(args.complement, weights='weight' if args.weighted else None)

        if args.verbose:
            compl_edges = len(compl_G.es)
            orig_edges = len(G.es)
            pct_edges = (100.0 * compl_edges) / orig_edges 
            print('Saved complement edge-sampled graph in "{}" with {} edges out of {} ({:.2f}%%).'.format(args.complement, compl_edges, orig_edges, pct_edges))

    sys.exit(0)


def walk_command(G, args):
    n2v_G = node2vec.Graph(G, args.directed, args.p, args.q)
    if args.p != 1 or args.q != 1:
        if args.verbose:
            print('Preprocessing transition probabilities.')
        n2v_G.preprocess_transition_probs(args.threads)

    if args.verbose:
        print('Generating random walks.')
    walks = n2v_G.simulate_walks(args.num_walks, args.walk_length, args.threads)

    if args.mapping:
        mapping = {k: v for (k, v) in json.load(open(args.mapping, 'r')).items()}
    else:
        mapping = {n['name']: n['name'] for n in G.vs}

    # dump the walks
    names = G.vs['name']
    with io.open(args.output, 'w', encoding='utf8') as f:
        for walk in walks:
            f.write(u' '.join(mapping[names[n]] for n in walk) + u'\n')

    if args.verbose:
        print('Saved random walks in "{}".'.format(args.output))

    sys.exit(0)


def embed_command(args):    
    if args.verbose:
        print('Training model from the random walks in "{}".'.format(args.walk))

    if args.algorithm == 'fasttext':
        from fastText.FastText import train_unsupervised, load_model
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
    elif args.algorithm == 'word2vec':
        from gensim.models.word2vec import Word2Vec, LineSentence
        walks = LineSentence(io.open(args.walk, 'r', encoding='utf8'))
        model = Word2Vec(walks, 
                         size=args.dimensions, 
                         window=args.context, 
                         min_count=0, 
                         sg=1, 
                         workers=args.threads, 
                         iter=args.epochs)
        model.save_word2vec_format(args.output)
    else:
        print('Unknown embedding algorithm: "{}".'.format(args.algorithm)) 
        sys.exit(1)

    if args.verbose:
        print('Saved {} embedding model in "{}".'.format(args.algorithm, args.output))

    sys.exit(0)


def main(args):
    if args.task == 'embed':
        embed_command(args)

    G = read_graph(args.graph, args.weighted, args.directed, args.verbose)
    if args.task == 'encode':
        encode_command(G, args)

    if args.task == 'sample':
        sample_command(G, args)

    if args.task == 'walk':
        walk_command(G, args)


if __name__ == "__main__":
    args = parse_commands()
    main(args)
    
