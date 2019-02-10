SAMPLE_COMMAND=${1:-python src/main.py sample}
GRAPH_PATH=${2:-graph/}
GRAPH_FORMAT=${3:-.edgelist}

for GRAPH in "Facebook" "CA-AstroPh" "Youtube/Youtube";
do
  $SAMPLE_COMMAND \
    -g "$GRAPH_PATH/$GRAPH$GRAPH_FORMAT" \
    -p 0.5 \
    -o "$GRAPH_PATH/$GRAPH-P50$GRAPH_FORMAT" \
    -c "$GRAPH_PATH/$GRAPH-P50-C$GRAPH_FORMAT" &;
done
