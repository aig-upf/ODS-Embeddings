# Embed every other graph with the best configuration found
GRAPH_NAMES=("Facebook" "CA-AstroPh" "BlogCatalog"  "CoCit")
GRAPH_PATHS=(""         ""           "BlogCatalog/" ""     )
for i in `seq 0 $((${#GRAPH_NAMES[@]} - 1))`; 
do
  for K in `seq 1 2`; 
  do
    GRAPH="${GRAPH_NAMES[$i]}"
    GRAPH_PATH="${GRAPH_PATHS[$i]}$GRAPH"
    sbatch bin/runRegressionExperiments.sh 'experiments/reg' 'bin/regressionExperiment.sh' "graph/$GRAPH_PATH" "$GRAPH" "$K"
  done
done
