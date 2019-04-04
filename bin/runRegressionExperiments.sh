#!/bin/bash
#SBATCH -J RegressionExperiments
#SBATCH -p high
#SBATCH -n 1 #number of tasks

module load python-igraph/0.7.1.post6-foss-2017a-Python-3.6.4
module load Keras/2.2.4-foss-2017a-Python-3.6.4 

# Load the desired variables
OUTPUT_DIR=${1:-experiments/reg}
REGRESSION_COMMAND=${2:-bin/regressionExperiment.sh}
GRAPH_PATH=${3:-graph}
GRAPH=${4:-Facebook}
GRAPH_K=${5:-2}
FORCE=${6:-}
D=32; E=250; C=6; M=2; K=$GRAPH_K

# Tasks to compute
TASKS=("pagerank_norm" "log_degree" "individual_overlap" "clust_coeff")
MEASURES=("mse" "mse" "mse" "mse")
NETWORKS=("-H 0 -N 0 -a 'tanh' -A 'sigmoid' -L 'mse' -P 'sgd' -E 1000" "-H 16 -N 0 -a 'tanh' -A 'linear' -L 'mse' -P 'sgd' -E 1000" "-H 16 -N 0 -a 'tanh' -A 'sigmoid' -L 'mse' -P 'sgd' -E 1000" "-H 16 -N 0 -a 'tanh' -A 'sigmoid' -L 'mse' -P 'sgd' -E 1000")

# Target files
TARGET_ENC="$GRAPH-K$K.json"
TARGET_WALK="$GRAPH-K$K.walk"
TARGET_EMB="$GRAPH-K$K-D$D-E$E-C$C-M$M.emb"
REGRESSION_COMMANDS=""

# Whether or not to force the result
if [[ ! -z "$FORCE" ]]; then
  rm -f "emb/$TARGET_EMB"
fi

# Run all tasks
for T_INDEX in `seq 0 $((${#TASKS[@]} - 1))`;
do
  TASK=${TASKS[$T_INDEX]}
  MEASURE=${MEASURES[$T_INDEX]}
  NETWORK=${NETWORKS[$T_INDEX]}
  DEST_PATH="models/$GRAPH-K$K-D$D-E$E-C$C-M$M.$TASK.h5py"
  OUTPUT_FILE="$OUTPUT_DIR/$GRAPH-K$K-D$D-E$E-C$C-M$M.$TASK.log"
  $REGRESSION_COMMAND "emb/$TARGET_EMB" "$GRAPH_PATH$GRAPH.edgelist" "labels/$TARGET_ENC" "$DEST_PATH" "$TASK" "$MEASURE" "$NETWORK" >> $OUTPUT_FILE
done
