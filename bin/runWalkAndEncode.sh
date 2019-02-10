#!/bin/bash
#
# Experiment preparation script
#
# vec. dim.   | d = 32, 64, 128
# epochs.     | e = 50, 250, 500,
# context     | c = 2, 3, 4, 5
# ngram sizes | m, M = (1, 1), (1, 2), (1, 3)
# distance    | k = 1, 2, 3
#

TARGET_DIR=${1:-.}
TRAIN_COMMAND=${2:-bin/train.sh}
GRAPH_PATH=${3:-graphs/}
GRAPH_NAME=${4:-Facebook}
GRAPH_FORMAT=${5:-.edgelist}
NUM_THREADS=${6:-8}
LINK_PREDICTION_COMMAND=${7:-bin/linkPredictionExperiment.sh}

# prepare the whole experiments array, so that we can easily distribute work
EXPERIMENTS_ARRAY=()
for K in 1; # 2 3;
do
  TARGET_ENC="$GRAPH_NAME-K$K.json"
  TARGET_WALK="$GRAPH_NAME-K$K.walk"
  CMD="$TRAIN_COMMAND \
      '$GRAPH_PATH$GRAPH_NAME$GRAPH_FORMAT' \
      '$TARGET_DIR/labels/$TARGET_ENC' \
      '$TARGET_DIR/walk/$TARGET_WALK' \
      '' \
      '-d $K' \
      '' \
      '' \
      '-t $NUM_THREADS -v 2'";
  EXPERIMENTS_ARRAY+=("$CMD")
done

