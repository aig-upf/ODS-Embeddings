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

So the resulting ordered degree sequence is: `{1, 2, 2, 3, 4}`.

# Contact

Please contact [nalvarez@ntent.com](mailto:nalvarez@ntent.com) if you have any questions 
regarding ODSE.
