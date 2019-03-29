# Run link prediction experiments
if [[ -z "$1" ]] || [[ $1 = "link" ]];
then
  for N in `seq 1`;
  do
    sbatch bin/runLPExperiments.sh '.' 'experiments/lp/' 'bin/train.sh' 'bin/linkPredictionExperiment.sh' 'graph/sampled/' "Facebook-$N" '.edgelist' '8'
    sbatch bin/runLPExperiments.sh '.' 'experiments/lp/' 'bin/train.sh' 'bin/linkPredictionExperiment.sh' 'graph/sampled/' "BlogCatalog-$N" '.edgelist' '8'
    sbatch bin/runLPExperiments.sh '.' 'experiments/lp/' 'bin/train.sh' 'bin/linkPredictionExperiment.sh' 'graph/sampled/' "CA-AstroPh-$N" '.edgelist' '8'
  done
fi

# Run classification experiments
if  [[ -z "$1" ]] || [[ $1 = "classify" ]];
then
  sbatch bin/runClassificationExperiments.sh '.' 'experiments/cmty/' 'bin/train.sh' 'bin/classificationExperiment.sh' 'graph/BlogCatalog' "BlogCatalog" '.edgelist' '8' 'models/' '.json' 'label.micro' "-H 0 -N 0 -a 'tanh' -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -E 30"
  sbatch bin/runClassificationExperiments.sh '.' 'experiments/cmty/' 'bin/train.sh' 'bin/classificationExperiment.sh' 'graph/Youtube'     "Youtube"     '.edgelist' '8' 'models/' '.json' 'label.micro' "-H 0 -N 0 -a 'tanh' -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -E 50"
  sbatch bin/runClassificationExperiments.sh '.' 'experiments/cls/' 'bin/train.sh' 'bin/classificationExperiment.sh' 'graph/'       "CoCit"        '.edgelist' '8' 'models/' '.json' 'category.micro' "-H 0 -N 0 -a 'tanh' -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -E 30"
fi

# Run regression experiments
if  [[ -z "$1" ]] || [[ $1 = "regress" ]];
then
  sbatch bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/'            "Facebook" '.edgelist' '8' 'models/'
  sbatch bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/BlogCatalog' "BlogCatalog" '.edgelist' '8' 'models/'
  sbatch bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/'            "CA-AstroPh" '.edgelist' '8' 'models/'
  sbatch bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/Youtube'     "Youtube" '.edgelist' '8' 'models/'
  sbatch bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/'            "CoCit" '.edgelist' '8' 'models/'
  sbatch bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/PPI'         "ppi-train" '.edgelist' '8' 'models/'
  sbatch bin/runRegressionExperiments.sh '.' 'experiments/reg/' 'bin/train.sh' 'bin/regressionExperiment.sh' 'graph/Reddit'      "reddit-train" '.edgelist' '8' 'models/'
fi

# Run classification experiments
if  [[ -z "$1" ]] || [[ $1 = "inductive" ]];
then
  sbatch bin/runInductiveExperiments.sh '.' 'experiments/cls/' 'bin/train.sh' 'bin/classificationExperiment.sh' 'graph/PPI'    "ppi-train"    '.edgelist' '8' 'models/' '.json' 'label.micro'    "-H 0 -N 0 -a 'tanh' -A 'sigmoid' -L 'categorical_crossentropy' -P 'sgd' -E 25"
  sbatch bin/runInductiveExperiments.sh '.' 'experiments/cls/' 'bin/train.sh' 'bin/classificationExperiment.sh' 'graph/Reddit' "reddit-train" '.edgelist' '8' 'models/' '.json' 'category.micro' "-H 0 -N 0 -a 'tanh' -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -E 30"
fi
