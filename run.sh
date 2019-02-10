# Prepare sampled directory
if [[ ! -d "graph/sampled" ]]; then
    mkdir graph/sampled
fi
./graph/sample.sh

# Create experiments log directory
if [[ ! -d "experiments" ]]; then
    mkdir experiments
fi

# Encode the graphs
sbatch bin/encodeAndWalkGraph.sh '.' 'experiments' 'bin/train.sh' 'graph/sampled' 'Facebook-P50' '.edgelist' '8'
sbatch bin/encodeAndWalkGraph.sh '.' 'experiments' 'bin/train.sh' 'graph/sampled' 'CA-AstroPh-P50' '.edgelist' '8'

# Run actual experiments 
