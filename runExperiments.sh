#!/bin/bash
#
# Experiment preparation script
#
# distance    | k = 1, 2, 3
# vec. dim.   | d = 32, 64, 128
# epochs.     | e = 50, 250, 500,
# context     | c = 2, 3, 4, 5
# ngram sizes | m, M = (1, 1), (1, 2), (1, 3)
#

TARGET_DIR=${1:-./}
TRAIN_COMMAND=${2:-train.sh}
TARGET_GRAPH=${3:-graphs/Facebook}
TARGET_OUTOUT=${3:-Facebook}
GRAPH_FORMAT=${4:-.edgelist}
NUM_THREADS=${5:-8}

EXPERIMENTS_ARRAY=()
for K in 1 2 3;
do
  for D in 32 64 128;
  do
    for E in 50 250 500;
    do
      for C in 2 3 4 5;
      do
        for M in 1 2 3;
        do
          CMD="$TARGET_DIR$TRAIN_COMMAND '$TARGET_GRAPH$GRAPH_FORMAT' 'labels/$TARGET_OUTOUT.json' 'walk/$TARGET_OUTOUT.walk' 'emb/$TARGET_OUTOUT.emb' '-d $K' '' '-d $D -c $C -e $E -M $M' '-t $NUM_THREADS'";
          EXPERIMENTS_ARRAY+=("$CMD")
        done
      done
    done
  done
done

