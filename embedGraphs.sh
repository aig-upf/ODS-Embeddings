# Do the full embedding for one of the samples of link prediction
sbatch bin/embedGraph.sh 'experiments/enc' 'bin/train.sh' 'graph/sampled/' "Facebook-1" '8'
sbatch bin/embedGraph.sh 'experiments/enc' 'bin/train.sh' 'graph/sampled/' "BlogCatalog-1" '8'
sbatch bin/embedGraph.sh 'experiments/enc' 'bin/train.sh' 'graph/sampled/' "CA-AstroPh-1" '8'

# Embed every other graph with the best configuration
