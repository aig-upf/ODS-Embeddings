#!/bin/bash
#SBATCH -J EncodeAndWalkGraph
#SBATCH -p high
#SBATCH -n 3 #number of tasks
#SBATCH -c 8
#SBATCH --array=1-3:1

module load Python/2.7.12-foss-2017a

# Load the desired variables
TARGET_DIR=${1:-.}
OUTPUT_DIR=${2:-.}
TRAIN_COMMAND=${3:-bin/train.sh}
GRAPH_PATH=${4:-graph/}
GRAPH_NAME=${5:-Facebook}
GRAPH_FORMAT=${6:-.edgelist}
NUM_THREADS=${7:-8}

# Prepare the encoding and walk array, so that we can easily distribute work
OUTPUTS_ARRAY=()
COMMANDS_ARRAY=()
for K in 1 2 3;
do
  TARGET_ENC="$GRAPH_NAME-K$K.json"
  TARGET_WALK="$GRAPH_NAME-K$K.walk"
  OUTPUT_FILE="$OUTPUT_DIR/$GRAPH_NAME-K$K.enc-walk.log"
  CMD="$TRAIN_COMMAND '$GRAPH_PATH$GRAPH_NAME$GRAPH_FORMAT' '$TARGET_DIR/labels/$TARGET_ENC' '$TARGET_DIR/walk/$TARGET_WALK' '' '-d $K' '' '' '-t $NUM_THREADS -v 2'"
  COMMANDS_ARRAY+=("$CMD")
  OUTPUTS_ARRAY+=("$OUTPUT_FILE")
done

# Run the array of jobs, dumping the output to file
# - If the Slurm environment variable is active, only one job will run
# - Otherwise, it will run all the tasks sequentially
if [[ ! -z "$SLURM_ARRAY_TASK_ID" ]]; then
  INDEX=$((SLURM_ARRAY_TASK_ID-1))
  eval "${COMMANDS_ARRAY[$INDEX]} > ${OUTPUTS_ARRAY[$INDEX]}"
else
  for i in seq ${#COMMANDS_ARRAY[@]}; do
    ${COMMANDS_ARRAY[$i]} > ${OUTPUTS_ARRAY[$i]}
  done
fi
