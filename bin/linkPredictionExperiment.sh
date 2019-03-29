#!/bin/bash
MAIN="python src/learn.py"

module load python-igraph/0.7.1.post6-foss-2017a-Python-3.6.4

# input/output files
MODEL="${1:-emb/Facebook.emb}"
GRAPH="${2:-graph/Facebook.50P-C.edgelist}"
LABELS="${3:-labels/Facebook.50P.json}"
NUM_EXPERIMENTS="${4:-25}"

# execute necessary steps -- no target output means the step will be ignored
RESULT="[LINK] $MODEL $GRAPH $LABELS $NUM_EXPERIMENTS"
for N in `seq $NUM_EXPERIMENTS`; do
  OUTPUT=$(eval "$MAIN -M '$MODEL' link -g '$GRAPH' -m '$LABELS' | grep 'AUC' | sed 's/\"/ /g' | rev | cut -d' ' -f2 | rev")
  RESULT="$RESULT $OUTPUT"
done
echo "$RESULT"
