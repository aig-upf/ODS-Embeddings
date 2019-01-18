Full paper outline.

1. Method

Modelling inspired by LM:
	- Capture node structural contexts.
	- Build a compact representation that captures (un-)related terms.
	- We do NOT care about conditional sentence probabilities for a generative model as in standard LM.

1.1. Structural Encoding.

Local sampling.

1.1.1. Neighbourhood Representation: Ego Networks.
	- Capture structure by representing the neighbourhood.
	- A direct approach: ego network at a maximum distance `k` surrounding a node.
	- Induced subgraph is still not something that can directly be trained on.
1.1.2. Ego-based Ordered Degree Sequences.
	- For inductive learning, our subgraph should be turned into a single representation that is permutation invariant.
	- This is dependent on the nodes in the graph and their local relations. 
	- To capture structure, we use node-level structural features: degrees in the induced subgraph (optionally including information about the node at the center).
	- The representation can be trivially made permutation invariant by represented the degrees as a sorted list, known as an ordered degree sequence.
		• Ordered degree sequences are used in struct2vec, subgraph2vec.
		• The ordered degree sequence can be compactly represented as a sorted list of pairs: <degree, occurrences>.
		• Equivalent to a 'bag of words' of degree features!
1.1.3. Encoding Neighbourhoods as Strings.
	- The previous representation is already usable as an input to a model that may accept weighted bag of words inputs.
	- Given literature and existing work, we can reduce the modelling task to a well known algorithm in other domain.
	- This reduction means (1) training a language model, for which we need (2) an additional encoding step to represent ordered degree sequences as strings.
1.1.4. Representation Functions.
	- A direct mapping assigns a character to each of the possible degrees in the subgraphs and forms the string in order, repeating by number of occurrences.
	- Those strings are too long, which may unnecessarily degrade performance. [Insert Graphs]
	- The degrees do not need to be linearly mapped to a string: higher degrees might be bucketed together!
	- The number of occurrences may be corrected --and discretized-- to reduce overall string length.

1.2. Random Walk Generation.

Global sampling.

NOTE: Mixing time to stationary distribution -- empirical and theoretical bounds check?

1.2.1. Motivation on Using Random Walks.
	- We reduce our problem to a task language modelling, and the structural encoding represents words with morphological aspects.
	- Random walk sampling captures co-occurrences across nodes -- in our case, about their structural labels.
	- To build a sentence for a language modelling algorithm, print each node label that is traversed.
	- Each random walk is independent, so sampling them is easy to parallelize.
1.2.1. DeepWalk: Uniformly Chosen Random Walks.
	- Generate walks by uniformly following edges of the graph, without jumping.
	— If the graph is weighted, edge selection might consider the weights.
	- Very efficient & low memory footprint (alias table to sample edges, per node, ~ \sigma_n=1^|V| |E_n| / |V|).
1.2.2. node2vec: 2-step Random Walk with Return and In-out parameters.
	- Generate walks keeping track of the previous node.
	- Hyper-parameters control whether the new node should move away, stay at the same distance or remain close to the previous node.
	- More expressive than DeepWalk since it can be tuned to control close or distant aspects.
	- Still efficient & memory footprint on a per-edge basis.
1.2.3. Structural Label Walks.
	- In either DeepWalk or node2vec, the original node ids are used.
	- To train structural labels, follow the same random walk process but use structural labels instead.
	- The model should pick up information about structural surroundings.

1.3. Network Embeddings.

1.3.1. Node Embeddings as Language Modelling.
	- At this point, we can train network embeddings on the structural random walks.
	— Previous models rely on word2vec, which captures a single vector on a per-word basis.
	- Our input has structure encoded in the words --as ordered degree sequences-- so this approach would not be sufficient.
	- We can capture structural features using FastText.
1.3.2. Morphological Language Modelling with FastText.
	- Word vector, with 1 vector per structural label.
	- Sub-word N-Grams, corresponding to co-occurring structural features. 
	- Because degrees are sorted, N > 1 will capture common co-occurring degrees.
1.3.3. Learning Embeddings over Random Walks.
	- Skipgram as our objective.
	- Context sizes equal maximum distance from our node. 
	- Negative sampling as a form of NCE (negative sampling with binary cross entropy to approx. soft-max).

