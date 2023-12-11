from viterbi import *
from readFiles import *
from visualizeModel import * 

# Load in trained model
Pi = np.loadtxt('GenesPi.txt',delimiter=' ')
P = np.loadtxt('GenesP.txt',delimiter=' ')
B = np.loadtxt('GenesB.txt',delimiter=' ')

# Load in gene for testing
geneFileName = "HBB" # EDIT HERE TO CHANGE GENE FILE
numBasesToInclude = 400 # EDIT HERE TO CHANGE LENGTH OF BASES WE USE (< 536)
seqstr = fileToGene(geneFileName, False)[0:numBasesToInclude]

# define a code (treats capital & lower case the same)
code = {"A":0, "C":1, "G":2, "T":3, "a":0, "c":1, "g":2, "t":3}
# make a seq array
seq = np.array([code[b] for b in seqstr])


# Predict states
code2 = {0:"E1", 1:"E2", 2:"E3", 3:"IS1", 4:"IS2", 5:"I", 6:"IE1", 7:"IE2"} 
states=viterbi(P, B, Pi, seq, np.shape(P)[0], len(seq))
states=[code2[q] for q in states]

# Print Out the Inferred States
print(" ".join(states))

visualize(states, seqstr, "HBB")