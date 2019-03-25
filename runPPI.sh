# 1. Train a model on the training split graphs
./bin/train.sh graph/PPI/ppi-train.edgelist labels/ppi.train.json walk/ppi.train.walk emb/ppi.train.emb '-d 2 -c' '' '' '-v 2' FORCE

# 2. Train predictive models with just features, features and graph, graph only
#    Evaluate them on the training set to get baseline measurements
python src/learn.py -a nil -M '' predict -g graph/PPI/ppi-train.edgelist -N 0 -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -m labels/ppi.train.json -f graph/PPI/ppi-feats.npy -l graph/PPI/ppi-class_map.json -E 50 -o models/ppi.train.just-feats.h5py -z models/ppi.train.just-feats.scaler
python src/learn.py -a nil -M ''  evaluate -g graph/PPI/ppi-train.edgelist -m labels/ppi.train.json -f graph/PPI/ppi-feats.npy -l graph/PPI/ppi-class_map.json -F label.micro -t models/ppi.train.just-feats.h5py -z models/ppi.train.just-feats.scaler

python src/learn.py -M emb/ppi.train.emb predict -g graph/PPI/ppi-train.edgelist -N 0 -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -m labels/ppi.train.json -f graph/PPI/ppi-feats.npy -l graph/PPI/ppi-class_map.json -E 50 -o models/ppi.train.feats.h5py -z models/ppi.train.feats.scaler
python src/learn.py -M emb/ppi.train.emb evaluate -g graph/PPI/ppi-train.edgelist -m labels/ppi.train.json -f graph/PPI/ppi-feats.npy -l graph/PPI/ppi-class_map.json -F label.micro -t models/ppi.train.feats.h5py -z models/ppi.train.feats.scaler

python src/learn.py -M emb/ppi.train.emb predict -g graph/PPI/ppi-train.edgelist -N 0 -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -m labels/ppi.train.json -l graph/PPI/ppi-class_map.json -E 50 -o models/ppi.train.no-feats.h5py -z models/ppi.train.no-feats.scaler
python src/learn.py -M emb/ppi.train.emb evaluate -g graph/PPI/ppi-train.edgelist -m labels/ppi.train.json -l graph/PPI/ppi-class_map.json -F label.micro -t models/ppi.train.no-feats.h5py -z models/ppi.train.no-feats.scaler

# 3. Encode the nodes in the validation graph
python src/main.py encode -g graph/PPI/ppi-valid.edgelist -d 2 -c -o labels/ppi.valid.json
python src/main.py encode -g graph/PPI/ppi-test.edgelist -d 2 -c -o labels/ppi.test.json

# 4. Evaluate the model on the validation graph
python src/learn.py -M emb/ppi.train.emb evaluate -g graph/PPI/ppi-valid.edgelist -m labels/ppi.valid.json -f graph/PPI/ppi-feats.npy -l graph/PPI/ppi-class_map.json -F label.micro -t models/ppi.train.feats.h5py -z models/ppi.train.feats.scaler
python src/learn.py -M emb/ppi.train.emb evaluate -g graph/PPI/ppi-test.edgelist -m labels/ppi.test.json -f graph/PPI/ppi-feats.npy -l graph/PPI/ppi-class_map.json -F label.micro -t models/ppi.train.feats.h5py -z models/ppi.train.feats.scaler

