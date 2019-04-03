# Do the full embedding for one of the samples of link prediction
sbatch bin/embedGraph.sh 'experiments/enc' 'bin/train.sh' 'graph/sampled' "Facebook-1" '32'
sbatch bin/embedGraph.sh 'experiments/enc' 'bin/train.sh' 'graph/sampled' "BlogCatalog-1" '32'
sbatch bin/embedGraph.sh 'experiments/enc' 'bin/train.sh' 'graph/sampled' "CA-AstroPh-1" '32'

# Embed every other graph with the best configuration found
NUM_THREADS=32
GRAPH_NAMES=("Facebook" "CA-AstroPh" "BlogCatalog"  "CoCit" "Youtube"  "Facebook-2" "Facebook-3" "BlogCatalog-2" "BlogCatalog-3" "CA-AstroPh-2" "CA-AstroPh-3")
GRAPH_PATHS=(""         ""           "BlogCatalog/" ""      "Youtube/" "sampled/"   "sampled/"   "sampled/"      "sampled/"      "sampled/"     "sampled/")
for i in `seq 0 $((${#GRAPH_NAMES[@]} - 1))`; 
do
  GRAPH="${GRAPH_NAMES[$i]}"
  GRAPH_PATH="${GRAPH_PATHS[$i]}$GRAPH"
  sbatch bin/embedGraphBest.sh 'experiments/enc' 'bin/train.sh' "$GRAPH_PATH" "$GRAPH" "$NUM_THREADS"
done
