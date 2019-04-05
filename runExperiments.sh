# Run link prediction experiments
if [[ -z "$1" ]] || [[ $1 = "link" ]];
then
  for N in `seq 1`;
  do
    sbatch bin/runLPExperiments.sh 'experiments/lp/' 'bin/linkPredictionExperiment.sh' 'graph/sampled/' "Facebook-$N"
    sbatch bin/runLPExperiments.sh 'experiments/lp/' 'bin/linkPredictionExperiment.sh' 'graph/sampled/' "BlogCatalog-$N"
    sbatch bin/runLPExperiments.sh 'experiments/lp/' 'bin/linkPredictionExperiment.sh' 'graph/sampled/' "CA-AstroPh-$N"
  done
fi

# Run link prediction experiments on unseen graphs
if [[ -z "$1" ]] || [[ $1 = "unseen" ]];
then
  for N in `seq 1`;
  do
    sbatch bin/runUnseenLPExperiments.sh 'experiments/lp/' 'bin/linkPredictionExperiment.sh' 'graph/sampled/' "Facebook-$N" "Facebook-$(($N + 1))"
    sbatch bin/runUnseenLPExperiments.sh 'experiments/lp/' 'bin/linkPredictionExperiment.sh' 'graph/sampled/' "BlogCatalog-$N" "BlogCatalog-$(($N + 1))"
    sbatch bin/runUnseenLPExperiments.sh 'experiments/lp/' 'bin/linkPredictionExperiment.sh' 'graph/sampled/' "CA-AstroPh-$N" "CA-AstroPh-$(($N + 1))"
  done
fi

# Run classification experiments
if  [[ -z "$1" ]] || [[ $1 = "classify" ]];
then
  sbatch bin/runClassificationExperiments.sh '.' 'experiments/cmty/' 'bin/train.sh' 'bin/classificationExperiment.sh' 'graph/BlogCatalog' "BlogCatalog" '.edgelist' '8' 'models/' '.json' 'label.micro'    "-H 0 -N 0 -a 'tanh' -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -E 30"
  sbatch bin/runClassificationExperiments.sh '.' 'experiments/cmty/' 'bin/train.sh' 'bin/classificationExperiment.sh' 'graph/Youtube'     "Youtube"     '.edgelist' '8' 'models/' '.json' 'label.micro'    "-H 0 -N 0 -a 'tanh' -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -E 50"
  sbatch bin/runClassificationExperiments.sh '.' 'experiments/cls/'  'bin/train.sh' 'bin/classificationExperiment.sh' 'graph/'            "CoCit"       '.edgelist' '8' 'models/' '.json' 'category.micro' "-H 0 -N 0 -a 'tanh' -A 'sigmoid' -L 'binary_crossentropy' -P 'sgd' -E 30"
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
  for DIST in 1 2;
  do
    sbatch runPPI.sh "experiments/cls/" $DIST '' '' "encode"
    for EPOCHS in 50 60 70 80 90 100;
    do
      sbatch runPPI.sh "experiments/cls/" $DIST $EPOCHS 25 "evaluate"
    done
  done
fi
