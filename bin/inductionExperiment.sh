#!/bin/bash
MAIN="python src/main.py"
LEARN="python src/learn.py"
SPLIT="python src/split_labels.py"

module load python-igraph/0.7.1.post6-foss-2017a-Python-3.6.4

# input/output files
MODEL="${1:-emb/BlogCatalog.emb}"
TRAIN_GRAPH="${2:-graph/BlogCatalog/BlogCatalog.edgelist}"
VALID_GRAPH="${3:-graph/BlogCatalog/BlogCatalog.edgelist}"
TEST_GRAPH="${4:-graph/BlogCatalog/BlogCatalog.edgelist}"
LABELS="${5:-labels/BlogCatalog.json}"
DEST_PATH="${6:-models/BlogCatalog.h5py}"
TASK_LABELS="${7:-graph/BlogCatalog/BlogCatalog.json}"
METRIC="${8:-label.micro}"
NETWORK_PARAMS="${9:--H 0 -N 0 -a 'tanh' -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -E 100}"
NUM_EXPERIMENTS="${10:-25}" 

# execute necessary steps -- no target output means the step will be ignored
VALID_DATA=""
TEST_DATA=""
RESULT="[REGRESS] $MODEL $TRAIN_GRAPH $LABELS $DEST_PATH $TASK $METRIC $SPLIT_SIZE $NUM_EXPERIMENTS"
for N in `seq $NUM_EXPERIMENTS`; do
  TRAIN_OUT=$(eval "$LEARN -M '$MODEL' predict -g '$TRAIN_GRAPH' -m '$LABELS' -f '$TASK_LABELS' -l '$TRAIN_PATH' $NETWORK_PARAMS -o '$DEST_PATH-$N' -z '$DEST_PATH-$N.scaler'")
  VALID_OUTPUT=$(eval "$LEARN -M '$MODEL' evaluate -g '$VALID_GRAPH' -m '$LABELS' -f '$TASK_LABELS' -l '$TRAIN_PATH' -t '$DEST_PATH-$N'  -z '$DEST_PATH-$N.scaler' -F '$METRIC' | grep "$METRIC" | sed 's/\"/ /g' | rev | cut -d' ' -f2 | rev")
  TEST_OUTPUT=$(eval "$LEARN -M '$MODEL' evaluate -g '$TEST_GRAPH' -m '$LABELS' -f '$TASK_LABELS' -l '$TRAIN_PATH' -t '$DEST_PATH-$N'  -z '$DEST_PATH-$N.scaler' -F '$METRIC' | grep "$METRIC" | sed 's/\"/ /g' | rev | cut -d' ' -f2 | rev")
  VALID_DATA="$VALID_DATA $VALID_OUTPUT"
  TEST_DATA="$TEST_DATA $TEST_OUTPUT"
done
echo "$RESULT [VALID] $VALID_DATA [TEST] $TEST_DATA"
