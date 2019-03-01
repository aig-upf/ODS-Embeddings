import graph
import preprocess as pp

import math
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_commands():
    main_args = argparse.ArgumentParser()
    main_args.add_argument('-v', '--verbose', help='Verbosity factor. 0 for silent, 1 for verbose.', type=int, default=1)
    main_args.add_argument('-w', '--weighted', help='Flag to specify that the graph is weighted.', action='store_true')
    main_args.add_argument('-D', '--directed', help='Flag to specify that the graph is directed.', action='store_true')
    main_args.add_argument('-g', '--graph', help='Input graph whose degrees will be plotted.', type=str, required=True)
    main_args.add_argument('-n', '--name', help='Name of the graph whose degrees will be plotted.', default='')
    main_args.add_argument('-d', '--distance', help='Distance of the per-node induced ego-networks.', type=int, default=1)
    args = main_args.parse_args()
    return args

if __name__ == "__main__":
    args = parse_commands()

    # read the graph
    G = graph.read_graph(args.graph, args.weighted, args.directed, args.verbose)

    # compute degree counts and frequencies
    G = pp.compute_degree_counts(G, args.distance, True, 8)
    D = pp.compute_global_degree_frequencies(G)

    if args.verbose:
        print('Finished computing global degree frequencies.')

    # compute the degree freq. distribution
    name = args.name
    title = ('{} - '.format(name) if name else name) + 'Degree Probability Distribution'
    D_dict = {d: f for d, f in D}
    x, y = zip(*[(d, D_dict.get(d, 0)) for d in range(min(D_dict), max(D_dict) + 1)])
    y = 1 - (np.asarray(y, np.float32) / sum(y)).cumsum()

    # do the actual plotting
    plt.title(title)
    plt.subplot(1, 2, 1)
    plt.semilogx(x, y)
    plt.xlabel('Degree')
    plt.ylabel('Complementary Cumulative Probability (semilog)')

    plt.subplot(1, 2, 2)
    plt.loglog(x, y)
    plt.xlabel('Degree')
    plt.ylabel('Complementary Cumulative Probability (loglog)')

    plt.gcf().set_size_inches(20, 10)
    plt.tight_layout()
    plt.show()

