# Do the full embedding for one of the samples of link prediction
sbatch bin/embedGraph.sh 'experiments/enc' 'bin/train.sh' 'graph/sampled' "Facebook-1" '8'
sbatch bin/embedGraph.sh 'experiments/enc' 'bin/train.sh' 'graph/sampled' "BlogCatalog-1" '8'
sbatch bin/embedGraph.sh 'experiments/enc' 'bin/train.sh' 'graph/sampled' "CA-AstroPh-1" '8'

# Embed every other graph with the best configuration found
NUM_THREADS=8
D=32; E=250; C=6; M=2; K=2;
GRAPH_NAMES=("Facebook" "BlogCatalog"  "CA-AstroPh" "Youtube"  "CoCit")
GRAPH_PATHS=(""         "BlogCatalog/" ""           "Youtube/" "")
for i in `seq 0 $((${#GRAPH_NAMES[@]} - 1))`; do
do
  GRAPH="${GRAPH_NAMES[$i]}"
  GRAPH_PATH="${GRAPH_PATHS[$i]}$GRAPH"
  srun -c $NUM_THREADS -p high -n 1 --mem=16384 bin/train.sh "graph/$GRAPH_PATH.edgelist" "labels/$GRAPH.json" "walk/$GRAPH.walk" "emb/$GRAPH.emb" "-d $K" '' '-d $D -c $C -e $E -M $M' '-t $NUM_THREADS -v 2' > "experiments/enc/$GRAPH-K$K-D$D-E$E-C$C-M$M.log"
done

