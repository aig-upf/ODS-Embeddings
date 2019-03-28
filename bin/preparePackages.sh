module load Python/2.7.12-foss-2017a

# if igraph gives you trouble, install it from: 
#    https://igraph.org/c/#downloads
# and then try using pip
pip install python-igraph --user 
pip install sklearn --user

# Install FastText
pip install pybind11 --user
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install . --user
