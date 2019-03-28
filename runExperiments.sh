# Prepare all the output directories
mkdir experiments
mkdir experiments/lp/
mkdir experiments/cmty/
mkdir experiments/reg/
mkdir experiments/cls/

# Run link prediction experiments
NUM_SAMPLES=${1:-3}
for N in `seq $NUM_SAMPLES`;
do
  bin/runLPExperiments.sh '.' 'experiments/lp/' 'bin/train.sh' 'bin/linkPredictionExperiment.sh' 'graphs/' "Facebook-$N" '.edgelist' '8'
  bin/runLPExperiments.sh '.' 'experiments/lp/' 'bin/train.sh' 'bin/linkPredictionExperiment.sh' 'graphs/' "BlogCatalog-$N" '.edgelist' '8'
  bin/runLPExperiments.sh '.' 'experiments/lp/' 'bin/train.sh' 'bin/linkPredictionExperiment.sh' 'graphs/' "CA-AstroPh-$N" '.edgelist' '8'
done

# Run community detection experiments
# bin/runCmtyDetectionExperiments.sh '.' 'experiments/cmty/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/BlogCatalog' "BlogCatalog" '.edgelist' '8' 'models/'
# bin/runCmtyDetectionExperiments.sh '.' 'experiments/cmty/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/Youtube'     "Youtube" '.edgelist' '8' 'models/'

# Run regression experiments
bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/'            "Facebook" '.edgelist' '8' 'models/'
bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/BlogCatalog' "BlogCatalog" '.edgelist' '8' 'models/'
bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/'            "CA-AstroPh" '.edgelist' '8' 'models/'
bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/Youtube'     "Youtube" '.edgelist' '8' 'models/'
bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/'            "CoCit" '.edgelist' '8' 'models/'
bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/PPI'         "ppi-train" '.edgelist' '8' 'models/'
bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/Reddit'      "reddit-train" '.edgelist' '8' 'models/'

# Run classification experiments
# bin/runRegressionExperiments.sh '.' 'experiments/cls/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/'       "CoCit" '.edgelist' '8' 'models/'
# bin/runRegressionExperiments.sh '.' 'experiments/cls/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/PPI'    "ppi-train" '.edgelist' '8' 'models/'
# bin/runRegressionExperiments.sh '.' 'experiments/cls/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/Reddit' "reddit-train" '.edgelist' '8' 'models/'
