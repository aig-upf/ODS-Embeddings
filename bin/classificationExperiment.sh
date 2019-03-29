#!/bin/bash
MAIN="python src/main.py"
LEARN="python src/learn.py"
SPLIT="python src/split_labels.py"

module load python-igraph/0.7.1.post6-foss-2017a-Python-3.6.4

# input/output files
MODEL="${1:-emb/BlogCatalog.emb}"
GRAPH="${2:-graph/BlogCatalog/BlogCatalog.edgelist}"
LABELS="${3:-labels/BlogCatalog.json}"
DEST_PATH="${4:-models/BlogCatalog.h5py}"
TASK_LABELS="${5:-graph/BlogCatalog/BlogCatalog.json}"
METRIC="${6:-label.micro}"
SPLIT_SIZE="${7:-0.5}"
NETWORK_PARAMS="${8:--H 0 -N 0 -a 'tanh' -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -E 100}"
NUM_EXPERIMENTS="${9:-25}" 

# execute necessary steps -- no target output means the step will be ignored
TRAIN_PATH="$TASK_LABELS.train"
VALID_PATH="$TASK_LABELS.valid"
eval "$SPLIT -i '$TASK_LABELS' -o '$TRAIN_PATH' -O '$VALID_PATH' -s $SPLIT_SIZE"
RESULT="[REGRESS] $MODEL $GRAPH $LABELS $DEST_PATH $TASK $METRIC $SPLIT_SIZE $NUM_EXPERIMENTS"
for N in `seq $NUM_EXPERIMENTS`; do
  TRAIN_OUT=$(eval "$LEARN -M '$MODEL' predict -g '$GRAPH' -m '$LABELS' -l '$TRAIN_PATH' $NETWORK_PARAMS -o '$DEST_PATH-$N' -z '$DEST_PATH-$N.scaler'")
  OUTPUT=$(eval "$LEARN -M '$MODEL' evaluate -g '$GRAPH' -m '$LABELS' -l '$TRAIN_PATH' -t '$DEST_PATH-$N'  -z '$DEST_PATH-$N.scaler' -F '$METRIC' | grep "$METRIC" | sed 's/\"/ /g' | rev | cut -d' ' -f2 | rev")
  RESULT="$RESULT $OUTPUT"
done
echo "$RESULT"
