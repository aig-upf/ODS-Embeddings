#!/bin/bash
TARGET_DIR=${1:-.}
DOWNLOAD_CMD=${2:-./download.sh}
GRAPHSAGE_PREPROCESSOR=${3:-python preprocess_graphsage.py}
COCIT_PREPROCESSOR=${4:-python preprocess_cocit.py}
BLOGCATALOG_PREPROCESSOR=${5:-python preprocess_blogcatalog.py}
YOUTUBE_PREPROCESSOR=${6:-python preprocess_youtube.py}

module load python-igraph/0.7.1.post6-foss-2017a-Python-3.6.4

# comparison with node2vec -- link prediction only graphs
$DOWNLOAD_CMD "https://snap.stanford.edu/data/facebook_combined.txt.gz" "gz" "$TARGET_DIR/Facebook.edgelist"
$DOWNLOAD_CMD "https://snap.stanford.edu/data/ca-AstroPh.txt.gz" "gz" "$TARGET_DIR/CA-AstroPh.edgelist"

# comparison with node2vec -- community detection (classification)
$DOWNLOAD_CMD "http://socialcomputing.asu.edu/uploads/1283153973/BlogCatalog-dataset.zip" \
              "zip" \
              "$TARGET_DIR/BlogCatalog" \
              "cp BlogCatalog-dataset/data/edges.csv BlogCatalog.edgelist && \
               cp BlogCatalog-dataset/data/group-edges.csv BlogCatalog.cmty && \
               rm -rf BlogCatalog-dataset"

# comparison with VERSE -- multiclass prediction
$DOWNLOAD_CMD "http://tsitsul.in/pub/academic_confs.mat" "txt" "$TARGET_DIR/CoCit.mat"

# comparison with GraphSAGE -- community and role detection
$DOWNLOAD_CMD "http://snap.stanford.edu/graphsage/ppi.zip" \
              "zip" \
              "$TARGET_DIR/PPI" \
              "mv ppi/* . && rmdir ppi"

$DOWNLOAD_CMD "http://snap.stanford.edu/graphsage/reddit.zip" \
              "zip" \
              "$TARGET_DIR/Reddit" \
              "mv reddit/* . && rmdir reddit"

# additional community detection on a larger graph (classification)
rm -rf "$TARGET_DIR/Youtube" && mkdir "$TARGET_DIR/Youtube"
$DOWNLOAD_CMD "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz" "gz" "$TARGET_DIR/Youtube/Youtube.edgelist"
$DOWNLOAD_CMD "https://snap.stanford.edu/data/bigdata/communities/com-youtube.top5000.cmty.txt.gz" "gz" "$TARGET_DIR/Youtube/Youtube.cmty"

# large scale training on a bipartite network (classification)
#$DOWNLOAD_CMD "http://files.grouplens.org/datasets/movielens/ml-20m.zip" \
#              "zip" \
#              "$TARGET_DIR/MovieLens" \
#              "cut -d',' -f1-3 ml-20m/ratings.csv | tr ',' ' ' | grep -v '^\s*\#' | tail -n +2 | sed 's/^/u_/g' > MovieLens.ratings.csv && \
#               paste -d' ' <(cut -d',' -f1 ml-20m/movies.csv) <(rev ml-20m/movies.csv | cut -d',' -f1 | rev) | tail -n +2 > MovieLens.genres.csv && \
#               rm -rf ml-20m && \
#               python ../prepare_movielens.py"

# Run postprocessing step
echo "Running additional postprocessing step to simplify experiments..."

# Youtube -- Tabs to spaces
tr '	' ' ' < "$TARGET_DIR/Youtube/Youtube.edgelist" | grep -v '^\s*\#' > "$TARGET_DIR/Youtube/Youtube.spaces.edgelist"
mv "$TARGET_DIR/Youtube/Youtube.spaces.edgelist" "$TARGET_DIR/Youtube/Youtube.edgelist"
echo "Youtube: change separator -- Tabs to spaces done!"

# CA-AstroPH -- Tabs to spaces
tr '	' ' ' < "$TARGET_DIR/CA-AstroPh.edgelist" | grep -v '^\s*\#' > "$TARGET_DIR/CA-AstroPh.spaces.edgelist"
mv "$TARGET_DIR/CA-AstroPh.spaces.edgelist" "$TARGET_DIR/CA-AstroPh.edgelist"
echo "CA-AstroPH: change separator -- Tabs to spaces done!"

# BlogCatalog -- Commas to spaces
tr ',' ' ' < "$TARGET_DIR/BlogCatalog/BlogCatalog.edgelist" | grep -v '^\s*\#' > "$TARGET_DIR/BlogCatalog/BlogCatalog.spaces.edgelist"
mv "$TARGET_DIR/BlogCatalog/BlogCatalog.spaces.edgelist" "$TARGET_DIR/BlogCatalog/BlogCatalog.edgelist"
echo "BlogCatalog: change separator -- Commas to spaces done!"

# Prepare GraphSAGE edgelist graphs
$GRAPHSAGE_PREPROCESSOR "$TARGET_DIR/"
echo "GraphSAGE datasets: PPI & Reddit -- JSON to edgelists done!"

# Prepare the Microsoft CoCitation dataset
$COCIT_PREPROCESSOR "$TARGET_DIR/"
echo "CoCit dataset: edgelist and labels done!"

# Prepare the community detection datasets
$BLOGCATALOG_PREPROCESSOR "$TARGET_DIR/"
$YOUTUBE_PREPROCESSOR "$TARGET_DIR/"
echo "BlogCatalog & Youtube datasets: Community mappings done!"
