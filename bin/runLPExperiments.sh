#!/bin/bash
#SBATCH -J LinkPredictionGraphs
#SBATCH -p high
#SBATCH -n 1 #number of tasks
#SBATCH -c 1
#SBATCH --mem=8192
#SBATCH --array=1-11:1

module load python-igraph/0.7.1.post6-foss-2017a-Python-3.6.4

OUTPUT_DIR=${1:-experiments/lp}
LINK_COMMAND=${2:-bin/linkPredictionExperiment.sh}
GRAPH_PATH=${3:-graph/sampled}
GRAPH=${4:-Facebook-1}

D=32; E=250; C=6; M=2; K=2
KEYS=(D D D E E C C C C M K)
VALUES=(32 64 128 50 500 2 4 8 10 1 1)
COMMANDS_ARRAY=()
for i in `seq 0 $((${#KEYS[@]} - 1))`; do
  SETTING=${KEYS[$i]}
  VALUE=${VALUES[$i]}
  _D=$D; _E=$E; _C=$C; _M=$M; _K=$K;
  declare _$SETTING=$VALUE

  INPUT_PATH="$GRAPH_PATH/$GRAPH-C.edgelist"
  ENCODING_PATH="labels/$GRAPH-K$_K.json"
  EMB_PATH="emb/$GRAPH-K$_K-D$_D-E$_E-C$_C-M$_M.emb"
  CMD="$TRAIN_COMMAND '$EMB_PATH' '$INPUT_PATH' '$ENCODING_PATH'";
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
