from readFiles import fileToGene
from baum_welchv3 import *
from viterbi import *
import numpy as np

##################################################################################
#                                  MODEL 1                                       #
##################################################################################
# read in the gene file
geneFileName = "INS" # EDIT HERE TO CHANGE GENE FILE
numBasesToInclude = 402 # EDIT HERE TO CHANGE LENGTH OF BASES WE USE (< 536)
seqstr = fileToGene(geneFileName, False)[0:numBasesToInclude]
print(seqstr)

# define a code (treats capital & lower case the same)
code = {"A":0, "C":1, "G":2, "T":3, "a":0, "c":1, "g":2, "t":3}

# make a seq array
seq = np.array([code[b] for b in seqstr])

# EDIT HERE TO TEST NEW MODELS########################################
# define model
P = np.array([
    #E1 E2 E3 I
    [0, 1, 0, 0], # E1
    [0, 0, 1, 0], # E2
    [0.5, 0, 0, 0.5], # E3
    [0.5, 0, 0, 0.5], # I
])

# describes which transition rates should be learned (True)
# and which are fixed (False)
# must have same dimensions as P
free = np.array([
    #E1 E2 E3 I
    [False, False, False, False], # E1
    [False, False, False, False], # E2
    [True, False, False, True], # E3
    [True, False, False, True], # I
])

# For now, saying it will always start in E1
# We might not want to update Pi if this gene should always start with E1?
Pi = np.array([1, 0, 0, 0])

# For now, initialize to uniform probabilities
# maybe random initialization is better? - can play around with it
B = np.array([
    # A      C     G     T
    [0.25, 0.25, 0.25, 0.25], # E1
    [0.25, 0.25, 0.25, 0.25], # E2
    [0.25, 0.25, 0.25, 0.25], # E3
    [0.25, 0.25, 0.25, 0.25], # I
])
#############################################################################

# Initialize Matrices 
# this code should be the same for all models?
nstates = np.shape(P)[0]
Alpha=np.empty((nstates,len(seq)))
Beta=np.empty((nstates, len(seq)))
Cmat=np.empty((len(seq)-1, nstates, nstates)) 
Gamma=np.empty((nstates, len(seq)))

# Need to somehow initialize these - find a better way?
Estep(Alpha, Beta, Cmat, Gamma, Pi, P, B, seq, len(seq))
Mstep(Alpha, Beta, Cmat, Gamma, Pi, P, B, seq, len(seq), free)

# Run the EM Algorithm #######################################################

# Can play with these values depending on run time, performance, etc.
epsilon = 0.01 # convergence criteria (difference between current and next likelihood)
max_iterations = 100 # cut off if it still hasn't converged

EMalgorithm(Alpha, Beta, Cmat, Gamma, Pi, P, B, seq, len(seq), free, epsilon, max_iterations)

# Print Out the Learned Parameters of the Model
print(Pi)
print(P)
print(B)

# Run the Viterbi Algorithm #######################################################

# code for printing the hidden states
code2 = {0:"E1", 1:"E2", 2:"E3", 3:"I"} 

states=viterbi(P, B, Pi, seq, nstates, len(seq))
states=[code2[q] for q in states]

# Print Out the Inferred States
print(states)

##################################################################################
# Methods for Visualizing the Output

"""INSERT CODE HERE FOR ANY VISUALIZATION WE WANT TO DO :)"""


##################################################################################
# Performance Statistics

"""INSERT CODE HERE FOR ANY PERFORMANCE STATISTICS WE WANT TO DO :)"""