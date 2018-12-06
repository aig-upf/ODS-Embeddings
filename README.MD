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

```
S = M(D_1) * R(F_1) + M(D_2) * R(F_2) + ... + M(D_n) * R(F_n)
```

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

```
−(y^t * log(y^p) + (1 − y^t) * log(1 − y^p))
```

Given a pair of nodes `<n_s, n_t>` where `n_s` is the source node, `n_t` is the target node that 
may or may not be in the neighbourhood of `n_s`, and `y^t` is the true label of such relationship.
To do so, we represent each structural label as a `R^d` vector, its corresponding structural 
embedding. With `e_i` standing for the representation of `n_i`, our skipgram model must learn the
parameters so that for a pair of embeddings `<e_i, e_j>`, the following holds:

```
σ(w * (e_i · e_j) + b) = y_t
```

The only missing piece of the puzzle is the underlying process to learn such embeddings. In 
node2vec, each `e_i` would be a learned vector. However, our structural labels include additional 
information, information that FastText is capable of capturing when represented as ordered degree
sequence strings. For a given embedding `e_i`, we will average the sum of both the global 
structural embedding `G_i` that corresponds to the label itself and the embeddings of the `M` 
co-occurring degree N-grams `c_{ij} | 1 <= j <= M` in the label string, as follows:

```
e_i = \frac{G_i + \sum_{j=1}^{M} c_{ij}}{1 + M}
```

The learning procedure must learn both the global structural embedding and the shared, 
degree-gram components. 

# Contact

Please contact [nalvarez@ntent.com](mailto:nalvarez@ntent.com) if you have any questions 
regarding ODSE.
