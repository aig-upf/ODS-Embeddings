#!/bin/bash
#SBATCH -J RegressionExperiments
#SBATCH -p high
#SBATCH -n 4 #number of tasks
#SBATCH -c 8
#SBATCH --array=1-4:1

module load python-igraph/0.7.1.post6-foss-2017a-Python-3.6.4

# Load the desired variables
TARGET_DIR=${1:-.}
OUTPUT_DIR=${2:-.}
TRAIN_COMMAND=${3:-bin/train.sh}
REGRESSION_COMMAND=${4:-bin/regressionExperiment.sh}
GRAPH_PATH=${5:-graphs/}
GRAPH_NAME=${6:-Facebook}
GRAPH_FORMAT=${7:-.edgelist}
NUM_THREADS=${8:-8}
MODEL_PATH=${9:-models}
FORCE=${10:-}

# All the available tasks -- each metric will be learned upon with 50:50 splits, using a single layer perceptron
# - If the output is meant to be in [0..1], we use a sigmoid activation
# - All the tasks are optimized for and evaluated by RMSE (ToDo: use KendallTau to evaluate pagerank?)
# - We do NOT do any hyperparameter tuning for this: just sgd for 1000 epochs.
# - The provided code can build arbitrary 'regular' DNNs (same hidden size and activation across all hiddens)
TASKS=("pagerank_norm" "log_betweenness" "log_degree" "global_overlap" "individual_overlap" "clust_coeff")
MEASURES=("mse" "mse" "mse" "mse" "mse" "mse")
NETWORKS=("-H 0 -N 0 -a 'tanh' -A 'sigmoid' -L 'mse' -P 'sgd' -E 1000" "-H 16 -N 0 -a 'tanh' -A 'linear' -L 'mse' -P 'sgd' -E 1000" "-H 16 -N 0 -a 'tanh' -A 'linear' -L 'mse' -P 'sgd' -E 1000" "-H 16 -N 0 -a 'tanh' -A 'sigmoid' -L 'mse' -P 'sgd' -E 1000" "-H 16 -N 0 -a 'tanh' -A 'sigmoid' -L 'mse' -P 'sgd' -E 1000" "-H 16 -N 0 -a 'tanh' -A 'sigmoid' -L 'mse' -P 'sgd' -E 1000")

# Prepare the whole experiments array, so that we can easily distribute work
# Note: we perform a very limited grid search for time & budget constraints
#       with the seemingly best hyperparams for link prediction on the FB graph
OUTPUTS_ARRAY=()
COMMANDS_ARRAY=()
for D in 32 64;
do
  for E in 500;
  do
    for C in 6 8; # 2 3 4 5;
    do
      for M in 2; # 1 2 3;
      do
        for K in 2; # 1 2 3;
        do
          TARGET_ENC="$GRAPH_NAME-K$K.json"
          TARGET_WALK="$GRAPH_NAME-K$K.walk"
          TARGET_EMB="$GRAPH_NAME-K$K-D$D-E$E-C$C-M$M.emb"
          OUTPUT_FILE="$OUTPUT_DIR/$GRAPH_NAME-K$K-D$D-E$E-C$C-M$M.log"
          REGRESSION_COMMANDS=""
          for T_INDEX in `seq ${#TASKS}`;
          do
            TASK=${TASKS[$T_INDEX]}
            MEASURE=${MEASURES[$T_INDEX]}
            NETWORK=${NETWORKS[$T_INDEX]}
            DEST_PATH="MODEL_PATH/$GRAPH-K$K-D$D-E$E-C$C-M$M.$TASK.h5py"
            REGRESSION_COMMANDS="$REGRESSION_COMMAND '$TARGET_DIR/emb/$TARGET_EMB' '$GRAPH_PATH$GRAPH_NAME-C$GRAPH_FORMAT' '$TARGET_DIR/labels/$TARGET_ENC' '$DEST_PATH' '$TASK' '$MEASURE' '$NETWORK'; $REGRESSION_COMMANDS"
          done
          DELETE_PATH="rm -f '$TARGET_DIR/emb/$TARGET_EMB'"
          CMD="$TRAIN_COMMAND '$GRAPH_PATH/$GRAPH_NAME$GRAPH_FORMAT' '$TARGET_DIR/labels/$TARGET_ENC' '$TARGET_DIR/walk/$TARGET_WALK' '$TARGET_DIR/emb/$TARGET_EMB' '-d $K' '' '-d $D -c $C -e $E -M $M' '-t $NUM_THREADS -v 2'; $REGRESSION_COMMANDS";
          if [[ ! -z "$FORCE" ]]; then
            CMD="$DELETE_PATH; $CMD"
          fi
          COMMANDS_ARRAY+=("$CMD")
          OUTPUTS_ARRAY+=("$OUTPUT_FILE")
        done
      done
    done
  done
done

# Run the array of jobs, dumping the output to file
# - If the Slurm environment variable is active, only one job will run
# - Otherwise, it will run all the tasks sequentially
if [[ ! -z "$SLURM_ARRAY_TASK_ID" ]]; then
  INDEX=$((SLURM_ARRAY_TASK_ID-1))
  eval "${COMMANDS_ARRAY[$INDEX]}" > ${OUTPUTS_ARRAY[$INDEX]}
else
  for i in `seq ${#COMMANDS_ARRAY[@]}`; do
    eval "${COMMANDS_ARRAY[$i]}" > ${OUTPUTS_ARRAY[$i]}
  done
fi
