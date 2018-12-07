# Ordered Degree Sequence Embeddings

This repository contains a Python implementation of Ordered Degree Sequence Embeddings (ODSE). 
The idea with ODSE is to represent a given graph by labelling nodes with a similar structure in 
the same way. In practical terms, this means:

* We compute a structural representation for each node. Currently, this is the Ordered Degree
Sequence for a *k-distance ego network* centered at the represented node.
* The structural representation is turned into a structural label that represents the node. The ODS
is currently transformed into a string, with each degree being represented by a unique unicode 
character. Those *degree characters* are repeated as many times as defined by a function of their 
occurrences in the ODS.
* The labels are embedded in a high dimensional space by training a language model that captures 
both in-node and neighbouring relationships. Currently, this is done training unsupervised FastText
embeddings on a random-walk through the graph. FastText captures node features by computing char 
ngram vectors, which represent degree-centric structural properties, and word vectors, which map 
to the overall structural label.

# Repository Overview

The code provided is based on the reference 
[node2vec implementation](https://github.com/aditya-grover/node2vec). To use FastText, follow the 
steps defined in [the FastText python documentation on 
Github](https://github.com/facebookresearch/fastText/tree/master/python). When up and running, you 
can run ODSE by simply passing:

1. An input graph, as an edgelist file. 
2. The label file, onto which to dump/load the per-node structural labels.
3. The walk file, where we will save a generated set of random walks. FastText requires a file path!
4. The output embedding file, containing the learned FastText embeddings.

# Ordered Degree Sequences: A Primer

An ordered degree sequence is simply the ordered sequence of the node degrees in a given sub-graph.
In our case, we generate ordered degree sequences for the k-distance ego networks centered at each 
of the nodes in the graph. For instance, consider the following (beautiful!) ASCII graph:

```
  0 ----- 1   7
 /      / |   |
/      /  |   |
2 ---- 3--4---6
       | /
       |/
       5
```

The `k=1` ego network for node 3 is:

```
          1
        / |
       /  |
2 ---- 3--4
       | /
       |/
       5
```

If we replace each node by their degrees, we get:

```
          2
        / |
       /  |
1 ---- 4--3
       | /
       |/
       2
```

So the resulting ordered degree sequence is: `{1, 2, 2, 3, 4}`. A possible extension, already 
covered in the provided implementation, is to ignore the degree of the source node. In the case 
where `k=1`, the degree of the source node is that of the length of the sequence, and might not 
add significant structural information. In our previous example, the resulting ordered degree 
sequence would then be `{1, 2, 2, 3}`.

# Representing an ODS as a String

To build a language model, each structural label must be turned into a string. A string 
representation is mostly needed so that we can leverage existing language modelling algorithms
such as word2vec, GloVe or FastText. Our structural labels encode some degree of information about
the neighbourhood in which a given node lives. We want to maintain the notion that a given degree
shows up with some total frequency among our peers.

Strings are made up of letters: we must first select an appropriate alphabet for our strings. In
our case, we define a mapping `M(deg): N —> U` that assigns an Unicode codepoint to every possible
integer degree seen in any of our neighbourhoods. As long as the assigned codepoints are not
considered spacing by the target algorithm, any arbitrary mapping will work.

To maintain the notion of degree frequency, we define a repetition function `R(freq): N -> N`. 
`R` takes the frequency of a given degree in an ordered degree sequence and returns how much its
letter representation should be repeated in the target string. Using `M` and `R`, we can 
define the structural string `S` given the sorted list of degrees `D` and their corresponding
frequencies `F`:

![S = M(D_1) * R(F_1) + M(D_2) * R(F_2) + ... + M(D_n) * R(F_n)](https://latex.codecogs.com/gif.latex?S%20%3D%20M%28D_1%29%20*%20R%28F_1%29%20&plus;%20M%28D_2%29%20*%20R%28F_2%29%20&plus;%20...%20&plus;%20M%28D_n%29%20*%20R%28F_n%29)

With `A * R` denoting repeat `A` exactly `R` times and `+` standing for string concatenation. A
more general approach merges both `M` and `R` into a function `Z(deg, freq): N -> N -> U^*` that 
is in charge of generating the representational string for a `<degree, frequency>` pair. This 
may allow for frequent pairs to be mapped into a single token, creating a different language 
model altogether.

# Learning Structural Representations

Every node is given a structural label that we now use when training a language model. Currently, 
this means learning a [FastText language model](https://fasttext.cc). To generate a text corpus, 
we run a node2vec style random-walk with controlling parameters `p` and `q`. Those parameters 
are further discussed in the node2vec paper, but they can be understood as a mechanism to control
how close or far away from the initial node we should go. The random-walk is a way of capturing 
neighbourhood dependencies on the graph: for every node we traverse, we will display its 
structural label. A full random walk will be a _sentence_ in our text corpus. Our corpus will 
be formed by many such walks, starting from every possible node. 

The last step in our algorithm is to actually learn a language model on the random-walk corpus.
Following that of node2vec, we optimize a negative sampling skipgram objective. This means that
the model is meant to correctly predict whether or not a word —a structural label, in our case— 
can appear in the vicinity of a target word, a source node. The maximum distance at which a node 
is considered to be in the vicinity  of a node is a hyperparameter, directly translated into the 
language model terminology as the size of the context window. Our task, formally, is to minimize 
the binary cross-entropy objective:

![-(y_t * log(y_p) + (1 - y_t) * log(1 - y_p))](https://latex.codecogs.com/gif.latex?-%28y_t%20*%20log%28y_p%29%20&plus;%20%281%20-%20y_t%29%20*%20log%281%20-%20y_p%29%29)

Given a pair of nodes `<n_s, n_t>` where `n_s` is the source node, `n_t` is the target node that 
may or may not be in the neighbourhood of `n_s`, and `y_t` is the binary label of such 
relationship. To do so, we represent each structural label as a `R^d` vector, its corresponding 
structural embedding. With `r_i` standing for the representation of `n_i`, our skipgram model must
learn the parameters so that for a pair of embeddings `<r_i, r_j>`, the following holds:

![\sigma(w * (r_i \cdot  r_j) + b) = y_t](https://latex.codecogs.com/gif.latex?%5Csigma%28w%20*%20%28r_i%20%5Ccdot%20r_j%29%20&plus;%20b%29%20%3D%20y_t)

The only missing piece of the puzzle is the underlying process to learn such embeddings. In 
node2vec, each `r_i` would be a learned vector. However, our structural labels include additional 
information, information that FastText is capable of capturing when represented as ordered degree
sequence strings. For a given embedding `r_i`, we will average the sum of both the global 
structural embedding `G_i` that corresponds to the label itself and the embeddings of the `M` 
co-occurring degree N-grams `c_{ij} | 1 <= j <= M` in the label string, as follows:

![r_i = \frac{G_i + \sum_{j=1}^{M} c_{ij}}{1 + M}](https://latex.codecogs.com/gif.latex?r_i%20%3D%20%5Cfrac%7BG_i%20&plus;%20%5Csum_%7Bj%3D1%7D%5E%7BM%7D%20c_%7Bij%7D%7D%7B1%20&plus;%20M%7D)

The learning procedure must learn both the global structural embedding and the shared, 
degree-gram components. 

# Example: Embedding a Structural Label

The best way to grasp what the equations represent is by example. Consider the structural label for 
node in our ASCII graph. For convenience, we map degrees to their numeric value and repeat them as 
often as they appear in the structural label. In our model, this means that `R` is the identity 
function, `R(freq) = freq`. The structural label then becomes `1223`, and we proceed to embed it. 

For simplicity, we restrict the degree N-grams to a minimum and a maximum length of 1. Furthermore, 
we limit our vectors to only have 3 components. Recall that the representation for label `1223` is 
the average of both the global vector corresponding to 1223 and the sum of 4 N-gram vectors, 
corresponding to `1`, `2`, `2` and `3` respectively. If we have the following embedding table:

```
Word embeddings (v):

1123:  1  0  1
1223:  2 -1 -1
....

N-gram embeddings (c):
   1: -1  0  1
   2:  1 -1  0
   3:  1  0 -2
   4: -1  0  1
...
```

Our complete representation is:

![r_{1223} = \frac{v_{1223} + c_1 + c_2 + c_2 + c_3}{5}](https://latex.codecogs.com/gif.latex?r_%7B1223%7D%20%3D%20%5Cfrac%7Bv_%7B1223%7D%20&plus;%20c_1%20&plus;%20c_2%20&plus;%20c_2%20&plus;%20c_3%7D%7B5%7D)

That is, the resulting vector is: 

![\frac{\langle2 - 1 + 1 + 1 + 1, -1 + 0 -1 + -1 + 0, -1 + 1 + 0 + 0 - 2\rangle}{5} =  \langle\frac{4}{5}, \frac{-3}{5}, \frac{-2}{5}\rangle](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Clangle2%20-%201%20&plus;%201%20&plus;%201%20&plus;%201%2C%20-1%20&plus;%200%20-1%20&plus;%20-1%20&plus;%200%2C%20-1%20&plus;%201%20&plus;%200%20&plus;%200%20-%202%5Crangle%7D%7B5%7D%20%3D%20%5Clangle%5Cfrac%7B4%7D%7B5%7D%2C%20%5Cfrac%7B-3%7D%7B5%7D%2C%20%5Cfrac%7B-2%7D%7B5%7D%5Crangle)

# Contact

Please contact [nalvarez@ntent.com](mailto:nalvarez@ntent.com) if you have any questions 
regarding ODSE.
