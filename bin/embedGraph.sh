#!/bin/bash
#SBATCH -J EmbedGraphs
#SBATCH -p high
#SBATCH -n 1 #number of tasks
#SBATCH -c 8
#SBATCH --mem=16384
#SBATCH --array=1-15:1

module load python-igraph/0.7.1.post6-foss-2017a-Python-3.6.4

OUTPUT_DIR=${1:-experiments/emb}
TRAIN_COMMAND=${2:-bin/train.sh}
GRAPH_PATH=${3:-graph}
GRAPH=${4:-Facebook}
NUM_THREADS=${5:-8}
FORCE=${6:-}

D=32; E=250; C=6; M=2; K=2
KEYS=(D D D E E E C C C C C M M K K)
VALUES=(32 64 128 50 250 500 2 4 6 8 10 1 2 1 2)
COMMANDS_ARRAY=()
for i in `seq 0 $((${#KEYS[@]} - 1))`; do
  SETTING=${KEYS[$i]}
  VALUE=${VALUES[$i]}
  _D=$D; _E=$E; _C=$C; _M=$M; _K=$K;
  declare _$SETTING=$VALUE

  INPUT_PATH="$GRAPH_PATH/$GRAPH.edgelist"
  ENCODING_PATH="labels/$GRAPH-K$_K.json"
  WALK_PATH="walk/$GRAPH-K$_K.walk"
  EMB_PATH="emb/$GRAPH-K$_K-D$_D-E$_E-C$_C-M$_M.emb"
  DELETE_PATH="rm -f 'emb/$TARGET_EMB'"
  CMD="$TRAIN_COMMAND '$INPUT_PATH' '$ENCODING_PATH' '$WALK_PATH' '$EMB_PATH' '-d $_K' '' '-d $_D -c $_C -e $_E -M $_M' '-t $NUM_THREADS -v 2'";
  if [[ ! -z "$FORCE" ]]; then
    CMD="$DELETE_PATH; $CMD"
  fi

  OUTPUT="$OUTPUT_DIR/$GRAPH-K$_K-D$_D-E$_E-C$_C-M$_M.log"
  COMMANDS_ARRAY+=("$CMD")
  OUTPUTS_ARRAY+=("$OUTPUT")
done

if [[ ! -z "$SLURM_ARRAY_TASK_ID" ]]; then
  INDEX=$((SLURM_ARRAY_TASK_ID - 1))
  eval "${COMMANDS_ARRAY[$INDEX]} >> ${OUTPUTS_ARRAY[$INDEX]}" 
else
  for i in `seq 0 $((${#COMMANDS_ARRAY[@]} - 1))`; do
    eval "${COMMANDS_ARRAY[$i]} >> ${OUTPUTS_ARRAY[$i]}"
  done
fi
