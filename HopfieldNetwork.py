# Hopfield Network
# Devanshu Agrawal

from __future__ import division
from pylab import randint
import numpy as np

class Hopfield():

# __init__(N) creates a Hopfield network with N nodes.

	def __init__(self, N):
		self.N = N
		self.W = np.asmatrix(np.zeros((N, N)))

# Let T be a list of lists:
# T = [T_1, T_2, . . . ],
# where every T_i is a list of length N,
# and every entry of T_i is either -1 or 1.
# T is the set of states that self will remember.

	def Train(self, T):
		for i in range(self.N):
			for j in range(i):
				self.W[i,j] = np.mean([T[k][i]*T[k][j] for k in range(len(T))])
				self.W[j,i] = self.W[i,j]

# Let X be a list of length N,
# where every entry of X is either -1 or 1.
# Let Steps be a positive integer.
# Run(X, Steps) returns the remembered state most similar to X.
# Steps is the number of iterations performed.
# If MC=True, then a stochastic method is used.

	def Run(self, X, Steps, MC=False):
		X = np.matrix(X).T
		if not MC:
			for i in range(Steps):
				X = np.sign(self.W*X)
		elif MC:
			for i in range(Steps):
				n = randint(self.N)
				X[n,0] = np.sign(self.W[n,:]*X)
		return [int(X[i,0]) for i in range(self.N)]
		

# Example:

T = [ \
[1] + [-1]*9, \
[-1]*7 + [1]*3 \
]

H = Hopfield(10)
H.Train(T)
X = [1]*2 + [-1]*8
print H.Run(X, 10)