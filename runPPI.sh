#!/bin/bash
#SBATCH -J EmbedInductiveGraphs
#SBATCH -p high
#SBATCH -n 1 #number of tasks
#SBATCH -c 64
#SBATCH --mem=16384

TARGET_PATH=${1:-experiments/cls/}
GRAPH_K=${2:-2}
NUM_EPOCHS="${3:-100}"
NUM_EXPERIMENTS="${4:-25}"

NUM_THREADS=64
D=32; E=250; C=6; M=2; K=$GRAPH_K

module load python-igraph/0.7.1.post6-foss-2017a-Python-3.6.4
module load Keras/2.2.4-foss-2017a-Python-3.6.4 

# 1. Train a model on the training split graphs
./bin/train.sh graph/PPI/ppi-train.edgelist labels/ppi-$K.train.json walk/ppi-$K.train.walk emb/ppi-$K.train.emb "-d $K -c" '' "-d $D -c $C -e $E -M $M" "-t $NUM_THREADS -v 2" '-v 2'

# 2. Encode the nodes in the validation and test graphs
python src/main.py encode -g graph/PPI/ppi-valid.edgelist -d $K -c -o labels/ppi-$K.valid.json
python src/main.py encode -g graph/PPI/ppi-test.edgelist -d $K -c -o labels/ppi-$K.test.json

TRAIN_FEATS="[PPI] TRAIN FEATS $K $NUM_EXPERIMENTS"
TRAIN_FULL="[PPI] TRAIN FULL $K $NUM_EXPERIMENTS"
TRAIN_GRAPH="[PPI] TRAIN GRAPH $K $NUM_EXPERIMENTS"
VALID_FEATS="[PPI] VALID FEATS $K $NUM_EXPERIMENTS"
VALID_FULL="[PPI] VALID FULL $K $NUM_EXPERIMENTS"
VALID_GRAPH="[PPI] VALID GRAPH $K $NUM_EXPERIMENTS"
TEST_FEATS="[PPI] TEST FEATS $K $NUM_EXPERIMENTS"
TEST_FULL="[PPI] TEST FULL $K $NUM_EXPERIMENTS"
TEST_GRAPH="[PPI] TEST GRAPH $K $NUM_EXPERIMENTS"
for N in `seq $NUM_EXPERIMENTS`; do
  # 2. Train predictive models with just features, features and graph, graph only
  #    Evaluate them on the training set to get baseline measurements
  python src/learn.py -a nil -M '' predict -g graph/PPI/ppi-train.edgelist -N 0 -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -m labels/ppi-$K.train.json -f graph/PPI/ppi-feats.npy -l graph/PPI/ppi-class_map.json -E $NUM_EPOCHS -o models/ppi-$K.train.just-feats.h5py -z models/ppi-$K.train.just-feats.scaler
  TRAIN_FEATS="$TRAIN_FEATS $(python src/learn.py -a nil -M ''  evaluate -g graph/PPI/ppi-train.edgelist -m labels/ppi-$K.train.json -f graph/PPI/ppi-feats.npy -l graph/PPI/ppi-class_map.json -F label.micro -t models/ppi-$K.train.just-feats.h5py -z models/ppi-$K.train.just-feats.scaler | grep 'label.micro' | sed 's/\"/ /g' | rev | cut -d' ' -f2 | rev)"

  python src/learn.py -M emb/ppi-$K.train.emb predict -g graph/PPI/ppi-train.edgelist -N 0 -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -m labels/ppi-$K.train.json -f graph/PPI/ppi-feats.npy -l graph/PPI/ppi-class_map.json -E $NUM_EPOCHS -o models/ppi-$K.train.feats.h5py -z models/ppi-$K.train.feats.scaler
  TRAIN_FULL="$TRAIN_FULL $(python src/learn.py -M emb/ppi-$K.train.emb evaluate -g graph/PPI/ppi-train.edgelist -m labels/ppi-$K.train.json -f graph/PPI/ppi-feats.npy -l graph/PPI/ppi-class_map.json -F label.micro -t models/ppi-$K.train.feats.h5py -z models/ppi-$K.train.feats.scaler | grep 'label.micro' | sed 's/\"/ /g' | rev | cut -d' ' -f2 | rev)"

  python src/learn.py -M emb/ppi-$K.train.emb predict -g graph/PPI/ppi-train.edgelist -N 0 -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -m labels/ppi-$K.train.json -l graph/PPI/ppi-class_map.json -E $NUM_EPOCHS -o models/ppi-$K.train.no-feats.h5py -z models/ppi-$K.train.no-feats.scaler
  TRAIN_GRAPH="$TRAIN_GRAPH $(python src/learn.py -M emb/ppi-$K.train.emb evaluate -g graph/PPI/ppi-train.edgelist -m labels/ppi-$K.train.json -l graph/PPI/ppi-class_map.json -F label.micro -t models/ppi-$K.train.no-feats.h5py -z models/ppi-$K.train.no-feats.scaler | grep 'label.micro' | sed 's/\"/ /g' | rev | cut -d' ' -f2 | rev)"

  # 4. Evaluate the model on the validation and test graphs
  VALID_FEATS="$VALID_FEATS $(python src/learn.py -a nil -M '' evaluate -g graph/PPI/ppi-valid.edgelist -m labels/ppi-$K.valid.json -f graph/PPI/ppi-feats.npy -l graph/PPI/ppi-class_map.json -F label.micro -t models/ppi-$K.train.just-feats.h5py -z models/ppi-$K.train.just-feats.scaler -V | grep 'label.micro' | sed 's/\"/ /g' | rev | cut -d' ' -f2 | rev)"
  TEST_FEATS="$TEST_FEATS $(python src/learn.py -a nil -M '' evaluate -g graph/PPI/ppi-test.edgelist -m labels/ppi-$K.test.json -f graph/PPI/ppi-feats.npy -l graph/PPI/ppi-class_map.json -F label.micro -t models/ppi-$K.train.just-feats.h5py -z models/ppi-$K.train.just-feats.scaler -V | grep 'label.micro' | sed 's/\"/ /g' | rev | cut -d' ' -f2 | rev)"

  VALID_FULL="$VALID_FULL $(python src/learn.py -M emb/ppi-$K.train.emb evaluate -g graph/PPI/ppi-valid.edgelist -m labels/ppi-$K.valid.json -f graph/PPI/ppi-feats.npy -l graph/PPI/ppi-class_map.json -F label.micro -t models/ppi-$K.train.feats.h5py -z models/ppi-$K.train.feats.scaler | grep 'label.micro' | sed 's/\"/ /g' | rev | cut -d' ' -f2 | rev)"
  TEST_FULL="$TEST_FULL $(python src/learn.py -M emb/ppi-$K.train.emb evaluate -g graph/PPI/ppi-test.edgelist -m labels/ppi-$K.test.json -f graph/PPI/ppi-feats.npy -l graph/PPI/ppi-class_map.json -F label.micro -t models/ppi-$K.train.feats.h5py -z models/ppi-$K.train.feats.scaler | grep 'label.micro' | sed 's/\"/ /g' | rev | cut -d' ' -f2 | rev)"

  VALID_GRAPH="$VALID_GRAPH $(python src/learn.py -M emb/ppi-$K.train.emb evaluate -g graph/PPI/ppi-valid.edgelist -m labels/ppi-$K.valid.json -l graph/PPI/ppi-class_map.json -F label.micro -t models/ppi-$K.train.no-feats.h5py -z models/ppi-$K.train.no-feats.scaler | grep 'label.micro' | sed 's/\"/ /g' | rev | cut -d' ' -f2 | rev)"
  TEST_GRAPH="$TEST_GRAPH $(python src/learn.py -M emb/ppi-$K.train.emb evaluate -g graph/PPI/ppi-test.edgelist -m labels/ppi-$K.test.json -l graph/PPI/ppi-class_map.json -F label.micro -t models/ppi-$K.train.no-feats.h5py -z models/ppi-$K.train.no-feats.scaler | grep 'label.micro' | sed 's/\"/ /g' | rev | cut -d' ' -f2 | rev)"
done

echo "TRAIN_FEATS" >> $TARGET_PATH/PPI-$K-$NUM_EPOCHS.log
echo "TRAIN_FULL" >> $TARGET_PATH/PPI-$K-$NUM_EPOCHS.log
echo "TRAIN_GRAPH" >> $TARGET_PATH/PPI-$K-$NUM_EPOCHS.log
echo "VALID_FEATS" >> $TARGET_PATH/PPI-$K-$NUM_EPOCHS.log
echo "VALID_FULL" >> $TARGET_PATH/PPI-$K-$NUM_EPOCHS.log
echo "VALID_GRAPH" >> $TARGET_PATH/PPI-$K-$NUM_EPOCHS.log
echo "TEST_FEATS" >> $TARGET_PATH/PPI-$K-$NUM_EPOCHS.log
echo "TEST_FULL" >> $TARGET_PATH/PPI-$K-$NUM_EPOCHS.log
echo "TEST_GRAPH" >> $TARGET_PATH/PPI-$K-$NUM_EPOCHS.log
∫