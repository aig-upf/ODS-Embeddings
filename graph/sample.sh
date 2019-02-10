SAMPLE_COMMAND=${1:-python src/main.py sample}
GRAPH_PATH=${2:-graph/}
GRAPH_FORMAT=${3:-.edgelist}

for GRAPH in "Facebook" "CA-AstroPh";
do
  for PERCENTAGE in 5 6 7 8 9;
  do
    $SAMPLE_COMMAND \
      -C \
      -g "$GRAPH_PATH/$GRAPH$GRAPH_FORMAT" \
      -p 0.$PERCENTAGE \
      -o "$GRAPH_PATH/sampled/$GRAPH-P${PERCENTAGE}0$GRAPH_FORMAT" \
      -c "$GRAPH_PATH/sampled/$GRAPH-P${PERCENTAGE}0-C$GRAPH_FORMAT";
  done
done
