This repository contains scripts that implement some examples of machine learning algorithms.

BayesianNetwork.py: This script implements variable elimination for binary Bayesian networks and allows for the computation of any (joint or conditional) probability once the network is defined.

DecisionTree.py: This script implements a greedy max information gain algorithm to fit a decision tree to binary data.

HopfieldNetwork.py: This script implements a deterministic Hopfield network with Hebbian learning (i.e., an Ising model at zero temperature).

MultiLayerPerceptron.py: This script implements a multi-layer perceptron with back propagation and gradient descent. An example perceptron is used as an autoencoder to compress as well as to classify images from the mnist handwritten digits data set.