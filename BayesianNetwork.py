# Bayesian Network
# Devanshu Agrawal

from __future__ import division
import numpy as np
import networkx as nx

# Bin(n, L) returns the binary representation of an integer n
# with exactly L digits.
# e.g., Bin(5, 4) = 0101

def Bin(n, L):
	B = bin(n + 2**L)
	return [int(B[i]) for i in range(3, len(B))]

# ListUnion(ListOfLists) returns a list that units all lists in ListOfLists.
# The output is sorted and contains no redundant elements,
# even if an input list has redundant elements.

def ListUnion(ListOfLists):
	L = []
	for M in ListOfLists:
		L += M
	return sorted(set(L))

# Let A be an array,
# and Index = [i_1, i_2, .. , I_N] a list of integers.
# ArrayEntry(A, Index) returns A[i_1, i_2, .. , i_N].

def ArrayEntry(A, Index):
	for i in Index:
		A = A[i]
	return A


# A factor is a function
# F: {0, 1}^N -> R,
# whose arguments are denoted x_1, x_2, .. , x_N
# Thus F(x_1, x_2, .. , x_N) is an output.
# F.Table is the range of F in array form;
# F.Table = [F(x) for x in {0, 1}^N].
# F.Variables = [x_1, x_2, .. , x_N] is the list of variables.
# F.VariableIndex is a dictionary that translates
# a variable into the index for the corresponding dimension of F.Table.

class Factor():

	def __init__(self, T, V):
		self.Table = T
		self.Variables = V
		self.VariableIndex = {V[i]:i for i in range(len(V))}


# Let Factors be a list of factors,
# and v a variable common to all factors in Factors.
# FactorProduct forms the product of all factors in Factors,
# whose range of values is stored in the array "table".
# The variable v is "summed out" of table by tensor contraction.
# FactorProduct returns the new factor not containing v.

def FactorProduct(Factors, v):
	variables = ListUnion([F.Variables for F in Factors])	
	table = []
	for n in range(2**len(variables)):
		State = Bin(n, len(variables))
		StateByVariable = {variables[i]:State[i] for i in range(len(variables))}
		P = 1
		for F in Factors:
			P = P*ArrayEntry(F.Table, [StateByVariable[w] for w in F.Variables])
		table.append(P)
	table = np.asarray([table])
	table = table.reshape([2 for i in range(len(variables))])
	i = variables.index(v)
	table = np.tensordot(table, np.array([1,1]), [i,0])
	variables.remove(v)
	return Factor(table, variables)

# Let Factors be a list of factors,
# and VariablesOut some subset of variables.
# VariableElimination returns a new list of factors,
# where all elements of VariablesOut are eliminated.

def VariableElimination(Factors, VariablesOut):
	for v in VariablesOut:
		IndicesRelevant = [i for i in range(len(Factors)) if v in Factors[i].Variables]
		FactorsRelevant = [Factors[i] for i in IndicesRelevant]
		F = FactorProduct(FactorsRelevant, v)
		Factors = [Factors[i] for i in range(len(Factors)) if i not in IndicesRelevant] + [F]
	return Factors


# BayesNet creates a DiGraph instance with "Order" nodes.
# The nodes are labeled 0 through Order.
# Every node has a CPT attribute initialized to [0, 0].

class BayesNet(nx.DiGraph):

	def __init__(self, order):
		nx.DiGraph.__init__(self)
		self.Order = order
		for M in range(order):
			self.add_node(M, CPT=np.array([0,0]))

# Let M be a node in self.
# Let Neighbors = self.predecessors(M)
# let "table" be nested list with
# dim(table) = |Neighbors|
# and entries
# table[State] = P(M=1 | Neighbors=State),
# where State in {0, 1}^|Neighbors|.
# SetCPT assigns to the CPT attribute of node M an array
# with dimension |Neighbors| + 1
# and entries
# P(M=i | Neighbors=State).
# Thus a first index of 0 means $M=0$.
# and a first index of 1 means M=1.

	def SetCPT(self, M, table):
		self.node[M]['CPT'] = np.array([1-np.array(table), np.array(table)])

# Marginal(Nodes, State) returns the marginal probability
# P(Nodes = State),
# where Nodes is a list of nodes in self,
# State in {0, 1}^|Nodes|,
# and Nodes = State means elementwise equality.

	def Marginal(self, Nodes, State):
		StateByNode = {Nodes[i]:State[i] for i in range(len(Nodes))}
		VariablesOut = [i for i in range(self.Order) if i not in Nodes]
		Factors = []
		for M in range(self.Order):
			F = Factor(self.node[M]['CPT'], [M]+sorted(self.predecessors(M)))
			Factors.append(F)
		Factors = VariableElimination(Factors, VariablesOut)
		P = 1
		for F in Factors:
			P = P*ArrayEntry(F.Table, [StateByNode[M] for M in F.Variables])
		return P

# Conditional(Nodes, State, Given, StateOfGiven) returns the probability
# P(Nodes=State | Given=StateOfGiven),
# where Nodes and Given are lists of nodes in self,
# State in {0, 1}^|Nodes| and StateOfGiven in {0, 1}^|Given|,
# and equality of lists is elementwise.

	def Conditional(self, Nodes, State, Given, StateOfGiven):
		return Marginal(Nodes+Given, State+StateOfGiven)/Marginal(Given, StateOfGiven)


# Example:

# Define a BayesNet G with 3 nodes.
G = BayesNet(3)

# Draw an arc from node 1 to node 0.
G.add_edge(1,0)

# Draw an arc from node 2 to node 0.
G.add_edge(2,0)

# Set the CPT of node 0 such that:
# P(0=1 | [1, 2]=[0, 0]) = 0.1
# P(0=1 | [1, 2]=[0, 1]) = 0.3
# P(0=1 | [1, 2]=[1, 0]) = 0.7
# P(0=1 | [1, 2]=[1, 1]) = 0.5
G.SetCPT(0, [[0.1, 0.3],[0.7, 0.5]])

# Set the CPT of node 1 such that:
# P(1=1) = 0.9
G.SetCPT(1, [0.9])

# Set the CPT of node 2 such that:
# P(2=1) = 0.9
G.SetCPT(2, [0.9])

# Print the probability that node 0 happens;
# i.e., P(0=1)
print G.Marginal([0], [1])