module load python-igraph/0.7.1.post6-foss-2017a-Python-3.6.4

# if igraph gives you trouble, install it from: 
#    https://igraph.org/c/#downloads
# and then try using pip
pip install python-igraph --user 
pip install networkx --user # needed to preprocess ppi + reddit
pip install sklearn --user

# Install FastText
pip install pybind11 --user
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install . --user
