import json
import random
import argparse


def parse_commands():
    main_args = argparse.ArgumentParser()
    main_args.add_argument('-s', '--split-size', help='Percentage of the splits to generate.', type=float, default=0.5)
    main_args.add_argument('-S', '--seed', help='Seed to use in the split.', type=int, default=None)
    main_args.add_argument('-i', '--input', help='Path to the label dictionary to be split.', type=str, required=True)
    main_args.add_argument('-o', '--output1', help='Path to the label dictionary in the first split.', type=str, required=True)
    main_args.add_argument('-O', '--output2', help='Path to the label dictionary in the second split.', type=str, required=True)
    args = main_args.parse_args()
    return args

if __name__ == "__main__":
    args = parse_commands()
    if args.seed is not None:
        random.seed(args.seed)

    with open(args.input) as f:
        labels = json.load(f)

    # shuffle accordingly
    label_mappings = [t for t in labels.items()]
    random.shuffle(label_mappings)

    # compute split sizes
    N = len(label_mappings)
    split_N = int(round(N) * args.split_size)

    # generate the two splits
    with open(args.output1, 'w') as f:
        json.dump({k: v for (k, v) in label_mappings[:split_N]}, f)

    with open(args.output2, 'w') as f:
        json.dump({k: v for (k, v) in label_mappings[split_N:]}, f)
