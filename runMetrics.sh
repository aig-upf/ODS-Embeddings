# 1. Compute the graph-based metric
python src/main.py compute -g graph/Facebook.edgelist -o graph/Facebook.clust_coeff.json clust_coeff

# 2. Split the targets 50:50
python src/split_labels.py -s 0.5 -i graph/Facebook.clust_coeff.json -o graph/Facebook.clust_coeff.train.json -O graph/Facebook.clust_coeff.valid.json

# 3. Train a model
python src/learn.py -M emb/Facebook.emb predict -g graph/Facebook.edgelist -H 16 -N 1 -a 'tanh' -A 'linear' -L 'mse' -P 'sgd' -m labels/Facebook.json -l graph/Facebook.clust_coeff.train.json -E 1000 -o models/Facebook.clust_coeff.h5py

# 4. Evaluate the model
python src/learn.py -M emb/Facebook.emb evaluate -g graph/Facebook.edgelist -m labels/Facebook.json -l graph/Facebook.clust_coeff.valid.json -F mse -t models/Facebook.clust_coeff.h5py
