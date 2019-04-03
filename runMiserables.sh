for D in 1 2 3; do
  ./bin/train.sh graph/Miserables.edgelist labels/Miserables_D$D.json walk/Miserables_D$D.walk emb/Miserables_D$D.emb "-d $D -c" '-n 10 -l 800' '-d 5 -e 5000 -m 1 -M 1 -c 2' '-v 2' '1'
  python src/eval_model.py -g graph/Miserables.edgelist -M emb/Miserables_D$D.emb -m labels/Miserables_D$D.json -t print > results/Miserables_D$D.cmty
done
