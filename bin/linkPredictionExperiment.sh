#!/bin/bash
MAIN="python src/main.py"

# input/output files
MODEL="${1:-emb/Facebook.emb}"
GRAPH="${2:-graph/Facebook.50P-C.edgelist}"
LABELS="${3:-labels/Facebook.50P.json}"

# execute necessary steps -- no target output means the step will be ignored
$MAIN "-M $MODEL" link "-g $GRAPH -m $LABELS" | grep "AUC"
