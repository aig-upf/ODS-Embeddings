#!/bin/bash
MAIN="python src/main.py"
LEARN="python src/learn.py"
SPLIT="python src/split_labels.py"

module load Python/2.7.12-foss-2017a

# input/output files
MODEL="${1:-emb/Facebook.emb}"
GRAPH="${2:-graph/Facebook.edgelist}"
LABELS="${3:-labels/Facebook.json}"
DEST_PATH="${4:-models/Facebook.pagerank.h5py}"
TASK="${5:-pagerank_norm}"
METRIC="${6:-medianse}"
SPLIT_SIZE="${7:-0.5}"
NETWORK_PARAMS="${8:--H 16 -N 0 -a 'tanh' -A 'linear' -L 'mse' -P 'sgd' -E 1000}"
NUM_EXPERIMENTS="${9:-25}"

# execute necessary steps -- no target output means the step will be ignored
TASK_PATH="$GRAPH.$TASK.json"
TRAIN_PATH="$GRAPH.$TASK.train.json"
VALID_PATH="$GRAPH.$TASK.valid.json"
eval "$MAIN compute -g '$GRAPH' -o '$TASK_PATH' '$TASK'"
eval "$SPLIT -i '$TASK_PATH' -o '$TRAIN_PATH' -O '$VALID_PATH' -s $SPLIT_SIZE"
RESULT="[TASK] $MODEL $GRAPH $LABELS $DEST_PATH $TASK $METRIC $SPLIT_SIZE $NUM_EXPERIMENTS"
for N in `seq $NUM_EXPERIMENTS`; do
  TRAIN_OUT=$(eval "$LEARN -M '$MODEL' predict -g '$GRAPH' -m '$LABELS' -l '$TRAIN_PATH' $NETWORK_PARAMS -o '$DEST_PATH-$N' -z '$DEST_PATH-$N.scaler'")
  OUTPUT=$(eval "$LEARN -M '$MODEL' evaluate -g '$GRAPH' -m '$LABELS' -l '$TRAIN_PATH' -t '$DEST_PATH-$N'  -z '$DEST_PATH-$N.scaler' -F '$METRIC' | grep "$METRIC" | sed 's/\"/ /g' | rev | cut -d' ' -f2 | rev")
  RESULT="$RESULT $OUTPUT"
done
echo "$RESULT"
