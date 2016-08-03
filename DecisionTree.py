# Decision Tree
# Devanshu Agrawal

from __future__ import division
import numpy as np
import networkx as nx

# Let A be a 2D array of data.
# NumObs(A) returns the number of rows minus 1.
# NumFeat(A) returns the number of columns minus 1.

def NumObs(A):
	return A.shape[0]-1

def NumFeat(A):
	return A.shape[1]-1

# Let X and Y be lists or 1D arrays of 0's and 1's.
# Think of X and Y as samples of two binary random variables.
# CondEntropy(X, Y) returns the empirical conditional entropy of the sample X given Y.

def CondEntropy(X, Y):
	N = len(X)
	sum = 0
	for x in range(1):
		for y in range(1):
			PY = len([i for i in range(N) if Y[i] == y])
			PXY = len([i for i in range(N) if X[i] == x and Y[i] == y])
			if PY > 0 and PXY > 0:
				sum += PXY*np.log2(PXY/PY)/N
	return -sum


class DecisionTree(nx.DiGraph):

# self.__init__ defines an instance of a decision tree.
# The initial tree contains no nodes.

	def __init__(self):
		nx.DiGraph.__init__(self)
		self.Count = 0

# Let event and branch be strings.
# Let cp in [0, 1].
# Example: Suppose self already contains a node labeled event D.
# Then the line
# self.AddNode('B', 'D=1', 0.7)
# adds a node labeled event B that is a child of event D along the "yes" (D=1) branch.
# Moreover, P(B=1 | D=1) = cp.
# If self contains no nodes, then the above line adds a root node labeled event B,
# and the second argument 'D=1' is ignored.
# If nodes labeled event D exist,
# then the new node B connects to the first D that was defined and does not have two child nodes.

	def AddNode(self, event, branch, cp):
		if self.Count == 0:
			self.add_node(0, Event=event, CP=np.array([1-cp, cp]), Joint=np.array([1-cp, cp]), Children=[0,0])
		else:
			YN = int(branch[-1])
			ParentEvent = branch[:-2]
			Parent = min([i for i in range(self.Count) if self.node[i]['Event']==ParentEvent and self.out_degree(i)<2])
			self.node[Parent]['Children'][YN] = self.Count
			self.add_node(self.Count, Event=event, YesOrNo=YN, CP=np.array([1-cp, cp]), Joint=np.array([0, 0]), Children=[0,0])
			self.add_edge(Parent, self.Count)
		self.Count += 1

# self.Run() runs the tree so that
# for every node N in the tree,
# the joint probability of N and its ancestors is computed.
# If all leaf nodes represent an event A,
# then self.Run() returns P(A=1).

	def Run(self):
		for N in self.nodes()[1:]:
			self.node[N]['Joint'] = self.node[N]['CP']*self.node[self.predecessors(N)[0]]['Joint'][self.node[N]['YesOrNo']]
		return sum([self.node[N]['Joint'] for N in self.nodes() if self.out_degree(N)==0])[1]

# Let string be a string.
# Example: The line
# self.Update('B=0')
# identifies all nodes in the tree labeled B,
# deletes all descendants down the B=0 branch (the "no" branch of B),
# deletes B itself,
# and connects the parent node of B with the child node at B=1.

	def Update(self, string):
		YN = int(string[-1])
		Nodes = [M for M in self.nodes() if self.node[M]['Event'] == string[:-2]]
		for M in Nodes:
			P = self.predecessors(M)[0]
			Keep = self.node[M]['Children'][YN]
			Del = self.node[M]['Children'][YN+1 %2]
			self.add_edge(P, Keep)
			self.node[P]['Children'][self.node[M]['YesOrNo']] = Keep
			self.node[Keep]['YesOrNo'] = self.node[M]['YesOrNo']
			self.remove_nodes_from([M, Del] + list(nx.descendants(self, Del)))

# Let Data be a 2D array of 0's and 1's.
# Every row of Data is an observation.
# Every column of Data is a feature,
# where the last column is the target feature.
# Let Label be a list or 1D array of strings,
# where the ith string is the name of the ith feature,
# and the last string is the name of the target feature.
# Let Threshold in [0.5, 1] such that
# P(TargetFeature=1 | Ancestors) >= Threshold or
# P(TargetFeature=0 | Ancestors) >= Threshold
# implies that a leaf node can be placed after Ancestors.
# self.Fit(Data, Labels, Threshold) produces a decision tree fitted to Data,
# where every node is labeled by a name in Labels,
# and every leaf node has CP[0] >= threshold, CP[1] >= Threshold, or has all features as ancestors.

	def Fit(self, Data, Labels, Threshold):
		Data = np.vstack([np.array([[i for i in range(Data.shape[1])]]), Data])
		Temp = [['root', Data]]
		for count in range(NumObs(Data)):
			SplitData = Temp
			Temp = []
			if SplitData == []:
				break
			for k in range(len(SplitData)):
				Branch = SplitData[k][0]
				S = SplitData[k][1]
				if NumFeat(S)==0 or abs(sum(S[1:,-1])/NumObs(S)-1/2) >= Threshold-1/2:
					self.AddNode(Labels[S[0,-1]], Branch, sum(S[1:,-1])/NumObs(S))
				else:
					Entropies = [CondEntropy(S[1:,-1], S[1:,j]) for j in range(NumFeat(S))]
					j_star = np.argmin(Entropies)
					I_0 = [0]+[i for i in range(1,S.shape[0]) if S[i,j_star]==0]
					I_1 = [0]+[i for i in range(1,S.shape[0]) if i not in I_0]
					S_0 = np.delete(np.array([S[i,:] for i in I_0]), j_star, 1)
					S_1 = np.delete(np.array([S[i,:] for i in I_1]), j_star, 1)
					Temp.append([Labels[S[0,j_star]]+'=0', S_0])
					Temp.append([Labels[S[0,j_star]]+'=1', S_1])
					self.AddNode(Labels[S[0,j_star]], Branch, NumObs(S_1)/NumObs(S))



# Example 1:

G = DecisionTree()
G.AddNode('D', 'Root', 0.7)
G.AddNode('B', 'D=1', 0.5)
G.AddNode('C', 'D=0', 0.8)
G.AddNode('A', 'B=1', 0.2)
G.AddNode('A', 'B=0', 0.9)
G.AddNode('A', 'C=1', 0.4)
G.AddNode('A', 'C=0', 0.6)

G.Run()
G.Update('B=0')


# Example 2:

Labels = ['R', 'F', 'U']
Data = np.array([ \
[1, 1, 1], \
[1, 1, 1], \
[1, 0, 1], \
[1, 0, 1], \
[0, 1, 1], \
[0, 1, 1], \
[0, 0, 0], \
[0, 0, 0] \
])

G = DecisionTree()
G.Fit(Data, Labels, 0.8)
G.Run()