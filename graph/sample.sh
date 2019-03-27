SAMPLE_COMMAND=${1:-python src/main.py sample}
GRAPH_PATH=${2:-graph}
GRAPH_FORMAT=${3:-.edgelist}
PROBABILITY=${4:-0.5}
TRIES=${5:-3}

for GRAPH in "Facebook" "BlogCatalog/BlogCatalog" "CA-AstroPh";
do
  for N in `seq $TRIES`;
  do
  	TARGET_NAME=$(echo $GRAPH | cut -d'/' -f2)
    $SAMPLE_COMMAND \
      -C \
      -g "$GRAPH_PATH/$GRAPH$GRAPH_FORMAT" \
      -p $PROBABILITY \
      -o "$GRAPH_PATH/sampled/$TARGET_NAME-$N$GRAPH_FORMAT" \
      -c "$GRAPH_PATH/sampled/$TARGET_NAME-$N-C$GRAPH_FORMAT";
  done
done
