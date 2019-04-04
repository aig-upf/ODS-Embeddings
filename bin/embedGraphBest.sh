#!/bin/bash
#SBATCH -J EmbedBestConfigGraph
#SBATCH -p high
#SBATCH -n 1 #number of tasks
#SBATCH -c 32
#SBATCH --mem=16384

module load python-igraph/0.7.1.post6-foss-2017a-Python-3.6.4

OUTPUT_DIR=${1:-experiments/emb}
TRAIN_COMMAND=${2:-bin/train.sh}
GRAPH_PATH=${3:-graph}
GRAPH=${4:-Facebook}
NUM_THREADS=${5:-32}
GRAPH_K=${6:-2}
FORCE=${7:-}

D=32; E=250; C=6; M=2; K=$GRAPH_K
INPUT_PATH="$GRAPH_PATH/$GRAPH.edgelist"
ENCODING_PATH="labels/$GRAPH-K$K.json"
WALK_PATH="walk/$GRAPH-K$K.walk"
EMB_PATH="emb/$GRAPH-K$K-D$D-E$E-C$C-M$M.emb"
OUTPUT="$OUTPUT_DIR/$GRAPH-K$K-D$D-E$E-C$C-M$M.log"
if [[ ! -z "$FORCE" ]]; then
  rm -f "$EMB_PATH"
fi

$TRAIN_COMMAND "$INPUT_PATH" "$ENCODING_PATH" "$WALK_PATH" "$EMB_PATH" "-d $K" '' "-d $D -c $C -e $E -M $M" "-t $NUM_THREADS -v 2" > $OUTPUT
