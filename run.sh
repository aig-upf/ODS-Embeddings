# Prepare sampled directory
mkdir graph/sampled
./graph/sample.sh

# Create experiments log directory
mkdir experiments
sbatch bin/encodeAndWalkGraph.sh '.' 'experiments' 'bin/train.sh' 'graph/sampled' 'Facebook-P50' '.edgelist' '8'
sbatch bin/encodeAndWalkGraph.sh '.' 'experiments' 'bin/train.sh' 'graph/sampled' 'CA-AstroPh-P50' '.edgelist' '8'

# Run actual experiments 
