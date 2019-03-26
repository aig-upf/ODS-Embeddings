# 1. Train a model on the training split graphs
./bin/train.sh graph/Facebook.edgelist labels/Facebook.json walk/Facebook.walk emb/Facebook.emb '-d 2 -c' '' '' '-v 2' FORCE

# 2. Compute per-node pagerank on the full graph
python src/main.py compute -g graph/Facebook.edgelist -o graph/Facebook.pagerank.json pagerank

# 3. Split the node to pagerank mapping into training and validation sets
python src/split_labels.py -i graph/Facebook.pagerank.json -o graph/Facebook.pagerank.train.json -O graph/Facebook.pagerank.valid.json

# 2. Train a model that learns pagerank
python src/learn.py -M emb/Facebook.emb predict -g graph/Facebook.edgelist -H 16 -N 0 -a 'tanh' -A 'linear' -L 'mse' -P 'adam' -m labels/Facebook.json -l graph/Facebook.pagerank.train.json -E 1000 -o models/Facebook.pagerank.h5py

# 3. Evaluate the pagerank model
python src/learn.py -M emb/Facebook.emb evaluate -g graph/Facebook.edgelist -m labels/Facebook.json -l graph/Facebook.pagerank.train.json -F "rev.kendalltau" -t models/Facebook.pagerank.h5py