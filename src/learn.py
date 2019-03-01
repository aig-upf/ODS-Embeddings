# -*- coding: utf-8 -*-
'''
Experiments designed to evaluated YNSAN-E. 

Author: Nur Álvarez González

'''

import io
import os
import sys
import json
import random
import argparse
import numpy as np
from joblib import dump


from graph import read_graph
import tasks.link_prediction as lp
import tasks.node_classification as nc
from tasks.ml_utils import merge_functions


def parse_commands():
    main_args = argparse.ArgumentParser()
    main_args.add_argument('-v', '--verbose', help='Verbosity factor. 0 for silent, 1 for verbose.', type=int, default=1)
    main_args.add_argument('-s', '--seed', help='Seed used for the random number generator.', type=int, default=None)
    main_args.add_argument('-S', '--split_size', help='Percentage used for the training split. The remaining data will be used for validation.', type=float, default=0.8)
    main_args.add_argument('-M', '--model', help='Trained embedding model, used to obtain the vector-space representations of the graph nodes.', type=str, required=True)
    main_args.add_argument('-a', '--algorithm', help='Algorithm used to train the model, chosen from \{fasttext,word2vec\}.', type=str, default='fasttext')
    main_subs = main_args.add_subparsers(title='Subcommands', description='Available commands to execute the different YNSAN-E experiments.', dest='task')

    graph_args = argparse.ArgumentParser(add_help=False)
    graph_args.add_argument('-g', '--graph', help='Input graph to be encoded.', type=str, required=True)
    graph_args.add_argument('-w', '--weighted', help='Flag to specify that the graph is weighted.', action='store_true')
    graph_args.add_argument('-D', '--directed', help='Flag to specify that the graph is directed.', action='store_true')

    link_args = main_subs.add_parser('link', description='Link prediction task.', parents=[graph_args])
    link_args.add_argument('-f', '--merge_function', help='Merge function to use when represented the edge, given the features of both nodes.', type=str, default='product')
    link_args.add_argument('-m', '--mapping', help='Mapping file, aliasing every node to its corresponding structural label (or any other alias). Expects a JSON-encoded dictionary (node -> structural label).', type=str, default='')
    link_args.add_argument('-o', '--output', help='Output file for the trained link prediction model.', type=str, default='')

    classify_args = main_subs.add_parser('classify', description='Node classification task.', parents=[graph_args])
    classify_args.add_argument('-m', '--mapping', help='Mapping file, aliasing every node to its corresponding structural label (or any other alias). Expects a JSON-encoded dictionary (node -> structural label).', type=str, default='')
    classify_args.add_argument('-f', '--features', help='Features file, matching every node to its specific features. Expects a JSON-encoded dictionary (node -> [features]).', type=str, default='')
    classify_args.add_argument('-l', '--labels', help='Labels file, matching every node with its corresponding label. Expects a JSON-encoded dictionary (node -> label).', type=str, required=True)
    classify_args.add_argument('-o', '--output', help='Output file for the trained classification model.', type=str, default='')

    args = main_args.parse_args()
    return args


def load_embedder(model_path, model_type):
    if model_type == 'fasttext':
        from fastText.FastText import load_model
        model = load_model(model_path)    
        return lambda node: model.get_sentence_vector(node)
    elif model_type == 'word2vec':
        from gensim.models import KeyedVectors
        model = KeyedVectors.load_word2vec_format(model_path)
        return lambda node: model[node]
    else:
        raise RuntimeError('Invalid/unknown model algorithm: "{}"'.format(model_type))


def link_command(G, args):    
    if args.verbose:
        print('Preparing link prediction experiment.')

    if args.mapping:
        mapping = {k: v for (k, v) in json.load(open(args.mapping, 'r')).items()}
    else:
        mapping = {n['name']: n['name'] for n in G.vs}

    model = load_embedder(args.model, args.algorithm)
    merge_fn = merge_functions[args.merge_function]
    lr, auc = lp.train(G, mapping, model, seed=args.seed, merge_fn=merge_fn, train_split=args.split_size)

    if args.verbose:
        print('Trained link prediction task with Area Under Curve (AUC) of "{}".'.format(auc))

    if args.output:
        dump(m, args.output)

    sys.exit(0)


def classify_command(G, args):    
    if args.verbose:
        print('Preparing classification experiment.')

    # structural labels
    if args.mapping:
        mapping = {k: v for (k, v) in json.load(open(args.mapping, 'r')).items()}
    else:
        mapping = {n['name']: n['name'] for n in G.vs}

    # node-specific features, if any
    if args.features:
        feat_json = {int(n): f for (n, f) in json.load(open(args.features, 'r')).items()}
        feat_dict = {n: np.asarray(f) for (n, f) in feat_json}
        feat_fn = lambda n: feat_dict[n]
    else:
        feat_fn = None

    # model and labels
    model = load_embedder(args.model, args.algorithm)
    labels = {n: l for (n, l) in json.load(open(args.labels, 'r')).items()}
    m, f1 = nc.train(G, mapping, model, labels, seed=args.seed, feat_fn=feat_fn, train_split=args.split_size)

    if args.verbose:
        print('Trained classification task with F1 score of "{}".'.format(f1))

    if args.output:
        dump(m, args.output)

    sys.exit(0)


def main(args):
    G = read_graph(args.graph, args.weighted, args.directed, args.verbose)
    if args.task == 'link':
        link_command(G, args)
    if args.task == 'classify':
        classify_command(G, args)


if __name__ == "__main__":
    args = parse_commands()
    main(args)
    
