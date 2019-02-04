TARGET_DIR=${1:-.}
DOWNLOAD_CMD=${2:-./download.sh}

# comparison with node2vec -- link prediction only graphs
$DOWNLOAD_CMD "https://snap.stanford.edu/data/facebook_combined.txt.gz" "gz" "$TARGET_DIR/Facebook.edgelist"
$DOWNLOAD_CMD "https://snap.stanford.edu/data/ca-AstroPh.txt.gz" "gz" "$TARGET_DIR/CA-AstroPh.edgelist"
$DOWNLOAD_CMD "https://snap.stanford.edu/biodata/datasets/10008/files/PP-Decagon_ppi.csv.gz" "gz" "$TARGET_DIR/Human-PPI.edgelist"

# comparison with node2vec -- community detection (classification)
$DOWNLOAD_CMD "http://socialcomputing.asu.edu/uploads/1283153973/BlogCatalog-dataset.zip" \
              "zip" \
              "$TARGET_DIR/BlogCatalog" \
              "cp BlogCatalog-dataset/data/edges.csv BlogCatalog.edgelist && \
               cp BlogCatalog-dataset/data/groups.csv BlogCatalog.cmty && \
               rm -rf BlogCatalog-dataset"

# additional community detection on a larger graph (classification)
rm -rf Youtube && mkdir Youtube
$DOWNLOAD_CMD "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz" "gz" "$TARGET_DIR/Youtube/Youtube.edgelist"
$DOWNLOAD_CMD "https://snap.stanford.edu/data/bigdata/communities/com-youtube.top5000.cmty.txt.gz" "gz" "$TARGET_DIR/Youtube/Youtube.cmty"

# large scale training on a bipartite network (classification)
$DOWNLOAD_CMD "http://files.grouplens.org/datasets/movielens/ml-20m.zip" \
              "zip" \
              "$TARGET_DIR/MovieLens" \
              "cut -d',' -f1-3 ml-20m/ratings.csv | tail -n +2 > MovieLens.ratings.csv && \
               cut -d',' -f1,3 ml-20m/movies.csv | tail -n +2 > MovieLens.genres.csv && \
               rm -rf ml-20m"

