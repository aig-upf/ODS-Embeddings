# Download all graphs
./graph/fetch.sh "graph/" "graph/download.sh" "python graph/preprocess_graphsage.py" "python graph/preprocess_cocit.py"

# Prepare the packages and libraries
./bin/preparePackages.sh

# Prepare sampled directory
NUM_SAMPLES="3"
if [[ ! -d "graph/sampled" ]]; then
  mkdir graph/sampled
  ./graph/sample.sh "python src/main.py sample" "graph" ".edgelist" "0.5" "$NUM_SAMPLES"
fi

# Create experiments log directory
if [[ ! -d "experiments" ]]; then
  mkdir experiments
  mkdir experiments/lp/
  mkdir experiments/cmty/
  mkdir experiments/reg/
  mkdir experiments/cls/
fi

# Create labels directory
if [[ ! -d "labels" ]]; then
  mkdir labels
fi

# Create walks directory
if [[ ! -d "walk" ]]; then
  mkdir walk
fi

# Create walks directory
if [[ ! -d "emb" ]]; then
  mkdir emb
fi

# Encode the graphs and produce their random walks
./encodeGraphs.sh "$NUM_SAMPLES"

# Run actual experiments 
./runExperiments.sh
