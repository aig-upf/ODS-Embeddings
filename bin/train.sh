#!/bin/bash
MAIN="python src/main.py"

# Input/output files
GRAPH="${1:-graph/Facebook.edgelist}"
LABELS="${2:-labels/Facebook.json}"
WALK="${3:-walk/Facebook.walk}"
MODEL="${4:-emb/Facebook.emb}"

# Additional args
ENCODE_ARGS="$5 -g $GRAPH -o $LABELS"
WALK_ARGS="$6 -g $GRAPH -m $LABELS -o $WALK"
EMBED_ARGS="$7 -w $WALK -o $MODEL"
GLOBAL_ARGS="$8"
FORCE_WRITE="$9"

# Execute necessary steps
# - No target output means the step will be ignored
# - If the file already exists, the step will run -- remove if needed!
if [[ ! -z "$GRAPH" && -f "$GRAPH" ]]; then
  # Encoding step -- Produce structural labels for every node
  if [[ ! -z "$LABELS" && ( ! -z "$FORCE_WRITE" || ! -f "$LABELS" ) ]]; then
    START_TIME=$(date +%s)
    $MAIN $GLOBAL_ARGS encode $ENCODE_ARGS
    END_TIME=$(date +%s)
    echo "[TIME] ENCODE $(( $END_TIME - $START_TIME )) s"
  fi

  # Random Walk step -- Sample graph interactions
  if [[ ! -z "$WALK" && ( ! -z "$FORCE_WRITE" || ! -f "$WALK" ) ]]; then
    START_TIME=$(date +%s)
    $MAIN $GLOBAL_ARGS walk $WALK_ARGS
    END_TIME=$(date +%s)
    echo "[TIME] WALK $(( $END_TIME - $START_TIME )) s"
  fi
fi

# Embedding step -- Learn structure representations
if [[ ! -z "$MODEL" && ( ! -z "$FORCE_WRITE" || ! -f "$MODEL" ) ]]; then
  START_TIME=$(date +%s)
  $MAIN $GLOBAL_ARGS embed $EMBED_ARGS
  END_TIME=$(date +%s)
  echo "[TIME] EMBED $(( $END_TIME - $START_TIME )) s"
fi
