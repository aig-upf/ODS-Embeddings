# Run link prediction experiments -- the main experiments with just one graph
for N in `seq 1`;
do
  bin/runLPExperiments.sh '.' 'experiments/lp/' 'bin/train.sh' 'bin/linkPredictionExperiment.sh' 'graph/sampled/' "Facebook-$N" '.edgelist' '8'
  bin/runLPExperiments.sh '.' 'experiments/lp/' 'bin/train.sh' 'bin/linkPredictionExperiment.sh' 'graph/sampled/' "BlogCatalog-$N" '.edgelist' '8'
  bin/runLPExperiments.sh '.' 'experiments/lp/' 'bin/train.sh' 'bin/linkPredictionExperiment.sh' 'graph/sampled/' "CA-AstroPh-$N" '.edgelist' '8'
done

# Run community detection experiments
bin/runClassificationExperiments.sh '.' 'experiments/cmty/' 'bin/train.sh' 'bin/classificationExperiment.sh' 'graph/BlogCatalog' "BlogCatalog" '.edgelist' '8' 'models/' '.json' 'label.micro' "-H 0 -N 0 -a 'tanh' -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -E 30"
bin/runClassificationExperiments.sh '.' 'experiments/cmty/' 'bin/train.sh' 'bin/classificationExperiment.sh' 'graph/Youtube'     "Youtube"     '.edgelist' '8' 'models/' '.json' 'label.micro' "-H 0 -N 0 -a 'tanh' -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -E 50"

# Run regression experiments
bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/'            "Facebook" '.edgelist' '8' 'models/'
bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/BlogCatalog' "BlogCatalog" '.edgelist' '8' 'models/'
bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/'            "CA-AstroPh" '.edgelist' '8' 'models/'
bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/Youtube'     "Youtube" '.edgelist' '8' 'models/'
bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/'            "CoCit" '.edgelist' '8' 'models/'
bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/PPI'         "ppi-train" '.edgelist' '8' 'models/'
bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/Reddit'      "reddit-train" '.edgelist' '8' 'models/'

# Run classification experiments
bin/runClassificationExperiments.sh '.' 'experiments/cls/' 'bin/train.sh' 'bin/classificationExperiment.sh' 'graph/'       "CoCit"        '.edgelist' '8' 'models/' '.json' 'category.micro' "-H 0 -N 0 -a 'tanh' -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -E 30"
bin/runClassificationExperiments.sh '.' 'experiments/cls/' 'bin/train.sh' 'bin/classificationExperiment.sh' 'graph/PPI'    "ppi-train"    '.edgelist' '8' 'models/' '.json' 'label.micro'    "-H 0 -N 0 -a 'tanh' -A 'sigmoid' -L 'categorical_crossentropy' -P 'sgd' -E 25"
bin/runClassificationExperiments.sh '.' 'experiments/cls/' 'bin/train.sh' 'bin/classificationExperiment.sh' 'graph/Reddit' "reddit-train" '.edgelist' '8' 'models/' '.json' 'category.micro' "-H 0 -N 0 -a 'tanh' -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -E 30"
