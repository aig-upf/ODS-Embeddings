# 1. Train a model on the training split graphs
./bin/train.sh graph/Reddit/reddit-train.edgelist labels/reddit.train.json walk/reddit.train.walk emb/reddit.train.emb '-d 2 -c' '' '' '-v 2' FORCE

# 2. Train predictive models with just features, features and graph, graph only
#    Evaluate them on the training set to get baseline measurements
python src/learn.py -a nil -M '' predict -g graph/Reddit/reddit-train.edgelist -N 0 -A 'softmax' -L 'categorical_crossentropy' -P 'sgd' -m labels/reddit.train.json -f graph/Reddit/reddit-feats.npy -l graph/Reddit/reddit-id_class_map.json -E 10 -o models/reddit.train.just-feats.h5py -z models/reddit.train.just-feats.scaler -V
python src/learn.py -a nil -M ''  evaluate -g graph/Reddit/reddit-train.edgelist -m labels/reddit.train.json -f graph/Reddit/reddit-feats.npy -l graph/Reddit/reddit-id_class_map.json -F categorical.micro -t models/reddit.train.just-feats.h5py -z models/reddit.train.just-feats.scaler -V

python src/learn.py -M emb/reddit.train.emb predict -g graph/Reddit/reddit-train.edgelist -N 0 -A 'softmax' -L 'categorical_crossentropy' -P 'sgd' -m labels/reddit.train.json -f graph/Reddit/reddit-feats.npy -l graph/Reddit/reddit-id_class_map.json -E 10 -o models/reddit.train.feats.h5py -z models/reddit.train.feats.scaler -V
python src/learn.py -M emb/reddit.train.emb evaluate -g graph/Reddit/reddit-train.edgelist -m labels/reddit.train.json -f graph/Reddit/reddit-feats.npy -l graph/Reddit/reddit-id_class_map.json -F categorical.micro -t models/reddit.train.feats.h5py -z models/reddit.train.feats.scaler -V

python src/learn.py -M emb/reddit.train.emb predict -g graph/Reddit/reddit-train.edgelist -N 0 -A 'softmax' -L 'categorical_crossentropy' -P 'sgd' -m labels/reddit.train.json -l graph/Reddit/reddit-id_class_map.json -E 10 -o models/reddit.train.no-feats.h5py -z models/reddit.train.no-feats.scaler -V
python src/learn.py -M emb/reddit.train.emb evaluate -g graph/Reddit/reddit-train.edgelist -m labels/reddit.train.json -l graph/Reddit/reddit-id_class_map.json -F categorical.micro -t models/reddit.train.no-feats.h5py -z models/reddit.train.no-feats.scaler -V

# 3. Encode the nodes in the validation graph
python src/main.py encode -g graph/Reddit/reddit-valid.edgelist -d 2 -c -o labels/reddit.valid.json
python src/main.py encode -g graph/Reddit/reddit-test.edgelist -d 2 -c -o labels/reddit.test.json

# 4. Evaluate the model on the validation graph
python src/learn.py -M emb/reddit.train.emb evaluate -g graph/Reddit/reddit-valid.edgelist -m labels/reddit.valid.json -f graph/Reddit/reddit-feats.npy -l graph/Reddit/reddit-id_class_map.json -F categorical.micro -t models/reddit.train.feats.h5py -z models/reddit.train.feats.scaler -V
python src/learn.py -M emb/reddit.train.emb evaluate -g graph/Reddit/reddit-test.edgelist -m labels/reddit.test.json -f graph/Reddit/reddit-feats.npy -l graph/Reddit/reddit-id_class_map.json -F categorical.micro -t models/reddit.train.feats.h5py -z models/reddit.train.feats.scaler -V

