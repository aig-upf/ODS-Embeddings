# Full graphs for classification and regression
sbatch bin/encodeAndWalkGraph.sh '.' 'experiments' 'bin/train.sh' 'graph/' "Facebook" '.edgelist' '8'
sbatch bin/encodeAndWalkGraph.sh '.' 'experiments' 'bin/train.sh' 'graph/BlogCatalog' "BlogCatalog" '.edgelist' '8'
sbatch bin/encodeAndWalkGraph.sh '.' 'experiments' 'bin/train.sh' 'graph/' "CA-AstroPh" '.edgelist' '8'
sbatch bin/encodeAndWalkGraph.sh '.' 'experiments' 'bin/train.sh' 'graph/Youtube' "Youtube" '.edgelist' '8'
sbatch bin/encodeAndWalkGraph.sh '.' 'experiments' 'bin/train.sh' 'graph/' "CoCit" '.edgelist' '8'
sbatch bin/encodeAndWalkGraph.sh '.' 'experiments' 'bin/train.sh' 'graph/PPI' "ppi-train" '.edgelist' '8'
sbatch bin/encodeAndWalkGraph.sh '.' 'experiments' 'bin/train.sh' 'graph/Reddit' "reddit-train" '.edgelist' '8'

# Sampled graphs for link prediction
for N in `seq 3`;
do
  sbatch bin/encodeAndWalkGraph.sh '.' 'experiments' 'bin/train.sh' 'graph/' "Facebook-$N" '.edgelist' '8'
  sbatch bin/encodeAndWalkGraph.sh '.' 'experiments' 'bin/train.sh' 'graph/' "BlogCatalog-$N" '.edgelist' '8'
  sbatch bin/encodeAndWalkGraph.sh '.' 'experiments' 'bin/train.sh' 'graph/' "CA-AstroPh-$N" '.edgelist' '8'
done
