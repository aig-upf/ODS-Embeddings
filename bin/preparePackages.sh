module load Python/2.7.12-foss-2017a
pip install networkx --user

# Install FastText
pip install pybind11 --user
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install . --user
