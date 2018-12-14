

We invite authors to submit a brief expression of interest containing a short
outline or extended abstract (approx. 1000 words), Including the topic, key
concepts, methods, expected results, and conclusions

(YNSAN-E) Your Neighbors Sure Are Nice: Scalable Graph Embeddings based on Neighborhood Similarity


**Topic:** Network embedding methods

**Key concepts:** network embeddings, representation learning, graph modeling, graph structures, machine learning

**Outline:**

**Introduction:**

Representation learning has been successfully applied to graphs capturing useful
interactions in compact ways, which can be used for downstream machine learning
tasks. The derived representations have seen applications as varied as molecule
generation for drug design, product recommendation in e-commerce, and friendship
suggestion in social networks.

A wide array of approaches have focused on local node features and interactions.
Such approaches, dubbed _transductive_, often require that all nodes in the
graph are known at training to learn a representation of each of them. These
algorithms, exemplified by DeepWalk, LINE, node2vec, and more recently VERSE,
are capable of capturing dependencies across node neighborhoods. However, their
representations depend on the node labels, thus making the outcome susceptible
to label permutations.

Taking these limitations into account, some models have been proposed to capture
arbitrary relationships while constructing order-independent representations. 
These approaches, dubbed _inductive_, manage to represent the information 
contained in the graph in such a way that individual node labels are irrelevant
to the final representation. However, algorithms that produce inductive 
representations tend to be computationally expensive, as they rely on graph 
convolutional networks, like graphSAGE, or pairwise notions of similarity, like
struct2vec. This means that they have difficulty scaling to large graphs where 
the number of nodes is in the order of millions.

To overcome these limitations, we introduce a novel inductive representation
learning framework with the aim to be (a) scalable (b) trivially parallelizable
and (c) competitive with respect to recent results in the state of the art.

**Methods:**

Our approach is based on a process of encoding nodes that introduces
neighborhood information into a model similar to the ones learned by node2vec or
DeepWalk. For every node of a given graph, the encoding process aggregates
information about the surroundings of the node. We derive a discrete structural
description from the ego network formed with the node at its center and every
node at a maximum distance given by a parameter `k`. Particularly, we take
degrees and their corresponding frequencies among nodes in the `k-induced` ego
network as descriptors of the structure neighboring the node. The structural
descriptions are converted into structural labels. To preserve the invariance to
permutations, the components of the structural labels are ordered, forming
ordered degree sequences. The ordering ensures that the strings in the induced
node language match exactly for nodes in the same context, and match
approximately for nodes in similar environments.

Having represented node neighborhood structures, the remaining problem of
learning network embeddings is the unsupervised task of training a language
model over the structural label language. Following node2vec, DeepWalk and
struct2vec, we train a language model over random walks on the graph --or the
denser structural label meta-graph--. Since our structural labels encode
information within themselves, i.e., structure has been morphologically encoded,
the model must capture both morphological and semantic information. In
particular, we train an unsupervised FastText model, whose N-gram sub-word
modeling can be understood as a means of capturing morphological information.
Additionally, we leverage the high scalability of FastText, which has been used
for very large datasets, such as Common Crawl, scaling to corpora with billions
of tokens. In summary, we produce a structural label for every node, which is
then embedded in a linear space by FastText. The output is a network embedding
representing our node on the basis of its neighborhood.

The resulting network embeddings are evaluated on a series of different tasks,
with the objective to test its scalability and suitability for several use
cases. Particularly, we evaluate the structural representations in the task of
link prediction, community detection (& role classification?) in social networks,
and protein/molecule identification.

For link prediction, we evaluate a series of social graphs of different sizes
to evaluate both accuracy and scalability: 
  1.) Arxiv AstroPhysics (V = 18772, E = 198110), 
  2.) DBLP (V = 317080, E = 1049866), 
  3.) YouTube (V = 1134890, E =2987624). 

On the task of community detection, we perform classification on embeddings from
the DBLP and YouTube datasets. Because of our structural approach, community 
detection should be a challenging task to our algorithm. (Also BlogCatalog: 
http://socialcomputing.asu.edu/datasets/BlogCatalog3 ?)

For role classification, we identify roles of senders in the Enron Dataset 
(V = 36692, E = 183831)?

For protein/molecule identification ???

In addition to the comparison across algorithms, performed in terms of both
scalability and usefulness in each specific task, we considering space and time
limitations, for which we provide additional theoretical bounds.

**Expected Result & Conclusions:**

This work will present a trivially scalable, inductive embedding method for
graphs. Our method is composed of an encoding process that represents node
neighborhoods as strings, and a language modeling task to derive network
embeddings from the encoded representation. Because the encoding process is
local and independent across nodes, it can be batched and trivially
parallelized. Likewise, the language modeling task reduces to generating random
walks, which can be easily distributed, and training a scalable FastText model.

To our knowledge, the proposed method is a simple yet novel contribution for
capturing structural information in the representation of a node. We aim to 
show the appropriateness of a single representation to be applied to multiple
tasks. Furthermore we expect this work to serve as a first step towards 
generalizing the method for other node features and multiple edge types.

