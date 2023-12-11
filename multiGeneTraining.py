from readFiles import fileToGene
from mult_seq_baum_welch import *
from viterbi import *
import numpy as np

##################################################################################
#                                  MODEL 2                                       #
##################################################################################
# read in the gene file
# test on HBB so don't include in training
geneFileNames = ["INS", "TUBB", "ACTB", "IGHV3-23", "LEP"] # EDIT HERE TO CHANGE GENE FILE
numBasesToInclude = 400 # EDIT HERE TO CHANGE LENGTH OF BASES WE USE (< 536)
seqstrs = [fileToGene(geneFile, False)[0:numBasesToInclude] for geneFile in geneFileNames]

# define a code (treats capital & lower case the same)
code = {"A":0, "C":1, "G":2, "T":3, "a":0, "c":1, "g":2, "t":3}

# make a seq array
seqs = np.empty((len(seqstrs),numBasesToInclude),dtype=np.int32)
for i in range(len(seqstrs)):
    seqs[i,:] = np.array([code[b] for b in seqstrs[i]])

# EDIT HERE TO TEST NEW MODELS########################################
# define model (Notes pg 249 - the intergenic regions)
p1 = np.random.rand()
p2 = np.random.rand()
P = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0], 
    [0, 0, 1, 0, 0, 0, 0, 0],
    [p1, 0, 0, 1-p1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0 ,0, 1, 0, 0],
    [0, 0, 0, 0, 0, p2, 1-p2, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0]
])

# describes which transition rates should be learned (True)
# and which are fixed (False)
# must have same dimensions as P
free = np.full((8,8), False)
free[2, 0] = True
free[2, 3] = True
free[5, 5] = True
free[5, 6] = True

# For now, saying it will always start in E1
# We might not want to update Pi if this gene should always start with E1?
Pi = np.array([1, 0, 0, 0, 0, 0, 0, 0])

# For now, initialize to uniform probabilities
# maybe random initialization is better? - can play around with it
# B = np.array([
#     # A      C     G     T
#     [0.25, 0.25, 0.25, 0.25], 
#     [0.25, 0.25, 0.25, 0.25], 
#     [0.25, 0.25, 0.25, 0.25], 
#     [0.25, 0.25, 0.25, 0.25], 
#     [0.25, 0.25, 0.25, 0.25],
#     [0.25, 0.25, 0.25, 0.25],
#     [0.25, 0.25, 0.25, 0.25],
#     [0.25, 0.25, 0.25, 0.25]
# ])
B = np.random.rand(np.shape(P)[0],4)
B = B/np.sum(B, axis=1)[:,np.newaxis]
#############################################################################

# Initialize Matrices 
# this code should be the same for all models?
n = np.shape(P)[0]
R, T = np.shape(seqs)
Alpha=np.empty((R, n, T))
Beta=np.empty((R, n, T))
Cmat=np.empty((R, T-1, n, n)) # first dim of Cmat is T-1?
Gamma=np.empty((R, n, T))

# Need to somehow initialize these - find a better way?
Estep(Alpha, Beta, Cmat, Gamma, Pi, P, B, seqs, T, R)
Mstep(Alpha, Beta, Cmat, Gamma, Pi, P, B, seqs, T, R, free)

# Run the EM Algorithm #######################################################

# Can play with these values depending on run time, performance, etc.
epsilon = 0.001 # convergence criteria (difference between current and next likelihood)
max_iterations = 1000 # cut off if it still hasn't converged

print("Training...")
EMalgorithm(Alpha, Beta, Cmat, Gamma, Pi, P, B, seqs, T, R, free, epsilon, max_iterations)

# Print Out the Learned Parameters of the Model
print(Pi)
print(P)
print(B)

# Save the results
np.savetxt("GenesPi.txt", Pi)
np.savetxt("GenesP.txt", P)
np.savetxt("GenesB.txt", B)