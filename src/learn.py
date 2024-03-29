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
from sklearn.externals.joblib import dump, load


from graph import read_graph
import tasks.link_prediction as lp
import tasks.node_prediction as nd
from tasks.ml_utils import merge_functions, NetworkFactory


def parse_commands():
    main_args = argparse.ArgumentParser()
    main_args.add_argument('-v', '--verbose', help='Verbosity factor. 0 for silent, 1 for verbose.', type=int, default=1)
    main_args.add_argument('-s', '--seed', help='Seed used for the random number generator.', type=int, default=None)
    main_args.add_argument('-S', '--split_size', help='Percentage used for the training split. The remaining data will be used for validation.', type=float, default=0.8)
    main_args.add_argument('-M', '--model', help='Trained embedding model, used to obtain the vector-space representations of the graph nodes.', type=str, required=True)
    main_args.add_argument('-a', '--algorithm', help='Algorithm used to train the model, chosen from \{fasttext,word2vec,nil\}.', type=str, default='fasttext')
    main_subs = main_args.add_subparsers(title='Subcommands', description='Available commands to execute the different YNSAN-E experiments.', dest='task')

    graph_args = argparse.ArgumentParser(add_help=False)
    graph_args.add_argument('-g', '--graph', help='Input graph to be encoded.', type=str, required=True)
    graph_args.add_argument('-w', '--weighted', help='Flag to specify that the graph is weighted.', action='store_true')
    graph_args.add_argument('-D', '--directed', help='Flag to specify that the graph is directed.', action='store_true')

    network_args = argparse.ArgumentParser(add_help=False)
    network_args.add_argument('-H', '--hidden-size', help='Size of the hidden layer(s), in number of units.', type=int, default=32)
    network_args.add_argument('-N', '--hidden-number', help='Number of hidden layer(s).', type=int, default=0)
    network_args.add_argument('-a', '--hidden-activation', help='Activation function on the hidden layers, from the possible activations defined in https://keras.io/activations/.', type=str, default='tanh')
    network_args.add_argument('-A', '--output-activation', help='Activation function on the output layer, from the possible activations defined in https://keras.io/activations/.', type=str, default='sigmoid')
    network_args.add_argument('-L', '--loss', help='Loss function to compute error gradients, from the possible losses defined in https://keras.io/losses/.', type=str, default='binary_crossentropy')
    network_args.add_argument('-P', '--optimizer', help='Optimization algorithm to use, from the possible optimizers defined in https://keras.io/optimizers/.', type=str, default='adam')

    link_args = main_subs.add_parser('link', description='Link prediction task.', parents=[graph_args])
    link_args.add_argument('-f', '--merge_function', help='Merge function to use when represented the edge, given the features of both nodes.', type=str, default='product')
    link_args.add_argument('-m', '--mapping', help='Mapping file, aliasing every node to its corresponding structural label (or any other alias). Expects a JSON-encoded dictionary (node -> structural label).', type=str, default='')
    link_args.add_argument('-o', '--output', help='Output file for the trained link prediction model.', type=str, default='')

    predict_args = main_subs.add_parser('predict', description='Node prediction task.', parents=[graph_args, network_args])
    predict_args.add_argument('-V', '--vectorize', help='Whether or not to vectorize the values of the label so that it can be predicted in binary or categorical classification tasks.', action='store_true')
    predict_args.add_argument('-E', '--epochs', help='Number of training epochs to use for training the model.', type=int, default=50)
    predict_args.add_argument('-m', '--mapping', help='Mapping file, aliasing every node to its corresponding structural label (or any other alias). Expects a JSON-encoded dictionary (node -> structural label).', type=str, default='')
    predict_args.add_argument('-f', '--features', help='Features file, matching every node to its specific features. Expects a JSON-encoded dictionary (node -> [features]).', type=str, default='')
    predict_args.add_argument('-l', '--labels', help='Labels file, matching every node with its corresponding label/target. Expects a JSON-encoded dictionary (node -> label).', type=str, required=True)
    predict_args.add_argument('-o', '--output', help='Output file for the trained prediction model.', type=str, required=True)
    predict_args.add_argument('-z', '--scaler', help='File for the feature scaler used when preprocessing the model.', type=str, default='')

    evaluate_args = main_subs.add_parser('evaluate', description='Node evaluation task.', parents=[graph_args])
    evaluate_args.add_argument('-V', '--vectorize', help='Whether or not to vectorize the values of the label so that it can be predicted in binary or categorical classification tasks.', action='store_true')
    evaluate_args.add_argument('-m', '--mapping', help='Mapping file, aliasing every node to its corresponding structural label (or any other alias). Expects a JSON-encoded dictionary (node -> structural label).', type=str, default='')
    evaluate_args.add_argument('-f', '--features', help='Features file, matching every node to its specific features. Expects a JSON-encoded dictionary (node -> [features]).', type=str, default='')
    evaluate_args.add_argument('-l', '--labels', help='Labels file, matching every node with its corresponding label/target. Expects a JSON-encoded dictionary (node -> label).', type=str, required=True)
    evaluate_args.add_argument('-F', '--function', help='Evaluation function, to chose between {label,categorical,mse}. Label and categorical evaluation calls F1, using any averaging mode available included in the string, or macro (e.g. "categorical_micro")', type=str, required=True)
    evaluate_args.add_argument('-t', '--task-model', help='Model file, trained on the label/target task. Expects a path to a Keras trained model.', type=str, required=True)
    evaluate_args.add_argument('-z', '--scaler', help='File for the feature scaler used when preprocessing the model.', type=str, default='')

    args = main_args.parse_args()
    return args


def load_embedder(model_path, model_type):
    if model_type == 'fasttext':
        from fastText.FastText import load_model
        model = load_model(model_path)    
        return lambda node: model.get_sentence_vector(node), model.get_dimension()
    elif model_type == 'word2vec':
        from gensim.models import KeyedVectors
        model = KeyedVectors.load_word2vec_format(model_path)
        return lambda node: model[node], model.syn0.shape[-1]
    elif model_type == 'nil':
        return lambda node: np.zeros((0,)), 0
    else:
        raise RuntimeError('Invalid/unknown model algorithm: "{}"'.format(model_type))


def link_command(G, args):    
    if args.verbose:
        print('Preparing link prediction experiment.')

    if args.mapping:
        mapping = {k: v for (k, v) in json.load(open(args.mapping, 'r')).items()}
    else:
        mapping = {n['name']: n['name'] for n in G.vs}

    model, emb_size = load_embedder(args.model, args.algorithm)
    merge_fn = merge_functions[args.merge_function]
    lr, auc = lp.train(G, mapping, model, seed=args.seed, merge_fn=merge_fn, train_split=args.split_size)

    if args.verbose:
        print('Trained link prediction task with Area Under Curve (AUC) of "{}".'.format(auc))

    if args.output:
        dump(m, args.output)

    sys.exit(0)


def prediction_command(G, args):    
    if args.verbose:
        print('Preparing prediction experiment.')

    # structural labels
    if args.mapping:
        mapping = {k: v for (k, v) in json.load(open(args.mapping, 'r')).items()}
    else:
        mapping = {n['name']: n['name'] for n in G.vs}

    # node-specific features, if any
    if args.features:
        if args.features.endswith('npy'):
            feat_matrix = np.load(args.features)
            feat_fn = lambda n: feat_matrix[int(n)]
        else:
            feat_json = json.load(open(args.features, 'r'))
            feat_dict = {n: np.asarray(f) for (n, f) in feat_json.items()}
            feat_fn = lambda n: feat_dict[n]
    else:
        feat_fn = None

    # neural network parameters
    nf = NetworkFactory(args)

    # model and labels
    model, emb_size = load_embedder(args.model, args.algorithm)
    labels = {n: l for (n, l) in json.load(open(args.labels, 'r')).items()}

    # define input and output sizes for the neural model
    m, s = nd.train(G, mapping, model, labels, nf,
                    feat_fn=feat_fn, 
                    vectorize=args.vectorize,
                    epochs=args.epochs,
                    verbose=args.verbose,
                    use_scaler=len(args.scaler) != 0)

    m.save(args.output)
    if s is not None:
        dump(s, args.scaler, compress=True)
    if args.verbose:
        print('Trained prediction task model to "{}".'.format(args.output))

    sys.exit(0)


def evaluation_command(G, args):    
    if args.verbose:
        print('Preparing evaluation of a prediction model.')

    # structural labels
    if args.mapping:
        mapping = {k: v for (k, v) in json.load(open(args.mapping, 'r')).items()}
    else:
        mapping = {n['name']: n['name'] for n in G.vs}

    # node-specific features, if any
    if args.features:
        if args.features.endswith('npy'):
            feat_matrix = np.load(args.features)
            feat_fn = lambda n: feat_matrix[int(n)]
        else:
            feat_json = json.load(open(args.features, 'r'))
            feat_dict = {n: np.asarray(f) for (n, f) in feat_json.items()}
            feat_fn = lambda n: feat_dict[n]
    else:
        feat_fn = None

    # model and labels
    from keras.models import load_model
    model, emb_size = load_embedder(args.model, args.algorithm)
    labels = {n: l for (n, l) in json.load(open(args.labels, 'r')).items()}
    m = load_model(args.task_model)
    s = load(args.scaler) if args.scaler else None

    # define input and output sizes for the neural model
    metric = nd.evaluate(G, mapping, model, labels, m, args.function,
                         feat_fn=feat_fn, 
                         vectorize=args.vectorize,
                         scaler=s)
    print('Evaluated predictive model with {} score of "{}".'.format(args.function, metric))
    sys.exit(0)


def main(args):
    G = read_graph(args.graph, args.weighted, args.directed, args.verbose)
    if args.task == 'link':
        link_command(G, args)
    if args.task == 'predict':
        prediction_command(G, args)
    if args.task == 'evaluate':
        evaluation_command(G, args)


if __name__ == "__main__":
    args = parse_commands()
    main(args)
    
