# Download all graphs
./graph/fetch.sh "graph/" "graph/download.sh" "python graph/preprocess_graphsage.py"

# Prepare the packages and libraries
./bin/preparePackages.sh

# Prepare sampled directory
if [[ ! -d "graph/sampled" ]]; then
  mkdir graph/sampled
  ./graph/sample.sh
fi

# Create experiments log directory
if [[ ! -d "experiments" ]]; then
  mkdir experiments
fi

# Create labels directory
if [[ ! -d "labels" ]]; then
  mkdir labels
fi

# Create walks directory
if [[ ! -d "walk" ]]; then
  mkdir walk
fi

# Encode the graphs
sbatch bin/encodeAndWalkGraph.sh '.' 'experiments' 'bin/train.sh' 'graph/sampled' 'Facebook-P50' '.edgelist' '8'
sbatch bin/encodeAndWalkGraph.sh '.' 'experiments' 'bin/train.sh' 'graph/sampled' 'CA-AstroPh-P50' '.edgelist' '8'

# Run actual experiments 
