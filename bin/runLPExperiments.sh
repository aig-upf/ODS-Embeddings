#!/bin/bash
#SBATCH -J LinkPredictionExperiments
#SBATCH -p high
#SBATCH -n 8 #number of tasks
#SBATCH -c 8
#SBATCH --array=1-324:1

module load python-igraph/0.7.1.post6-foss-2017a-Python-3.6.4

# Load the desired variables
TARGET_DIR=${1:-.}
OUTPUT_DIR=${2:-.}
TRAIN_COMMAND=${3:-bin/train.sh}
LINK_PREDICTION_COMMAND=${4:-bin/linkPredictionExperiment.sh}
GRAPH_PATH=${5:-graphs/}
GRAPH_NAME=${6:-Facebook}
GRAPH_FORMAT=${7:-.edgelist}
NUM_THREADS=${8:-8}

# Prepare the whole experiments array, so that we can easily distribute work
OUTPUTS_ARRAY=()
COMMANDS_ARRAY=()
for D in 32 64 128;
do
  for E in 50 250 500;
  do
    for C in 2 4 6 8; # 2 3 4 5;
    do
      for M in 1 2; # 1 2 3;
      do
        for K in 1 2; # 1 2 3;
        do
          TARGET_ENC="$GRAPH_NAME-K$K.json"
          TARGET_WALK="$GRAPH_NAME-K$K.walk"
          TARGET_EMB="$GRAPH_NAME-K$K-D$D-E$E-C$C-M$M.emb"
          OUTPUT_FILE="$OUTPUT_DIR/$GRAPH_NAME-K$K-D$D-E$E-C$C-M$M.log"
          CMD="rm -f '$TARGET_DIR/emb/$TARGET_EMB'; $TRAIN_COMMAND '$GRAPH_PATH/$GRAPH_NAME$GRAPH_FORMAT' '$TARGET_DIR/labels/$TARGET_ENC' '$TARGET_DIR/walk/$TARGET_WALK' '$TARGET_DIR/emb/$TARGET_EMB' '-d $K' '' '-d $D -c $C -e $E -M $M' '-t $NUM_THREADS -v 2'; $LINK_PREDICTION_COMMAND '$TARGET_DIR/emb/$TARGET_EMB' '$GRAPH_PATH$GRAPH_NAME-C$GRAPH_FORMAT' '$TARGET_DIR/labels/$TARGET_ENC'";
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
