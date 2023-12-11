from mult_seq_baum_welch import *
from viterbi import *

# read in simulated sequences
sequences = open('simulatedSequences.txt','r')
seqstrs = sequences.readlines()

# strip new line character from end of each sequence
seqstrs = [seq.strip() for seq in seqstrs]

# convert to a numpy array
code = {"A":0, "C":1, "G":2, "T":3}
seq = [[code[b] for b in seqstrs[i]] for i in range(len(seqstrs))]
seq = np.array(seq)

# use the first 900 sequences for training
training_seq = seq[0:900,:]

# set up the model
# initialize randomly
# set a seed
np.random.seed(178394)

p = np.random.rand()
print("initialized p to: ", np.round(p,3))
P = np.array([
    # N  M1  M2 M3 M4 M5 M6 M7 M8
    [p, 1-p, 0, 0, 0, 0, 0, 0, 0], # N
    [0, 0, 1, 0, 0, 0, 0, 0, 0], # M1
    [0, 0, 0, 1, 0, 0, 0, 0, 0], # M2
    [0, 0, 0, 0, 1, 0, 0, 0, 0], # M3
    [0, 0, 0, 0, 0, 1, 0, 0, 0], # M4
    [0, 0, 0, 0, 0, 0, 1, 0, 0], # M5
    [0, 0, 0, 0, 0, 0, 0, 1, 0], # M6
    [0, 0, 0, 0, 0, 0, 0, 0, 1], # M7
    [1, 0, 0, 0, 0, 0, 0, 0, 0], # M8
])

free = np.full((9,9), False)
free[0, 0] = True
free[0, 1] = True

Pi = np.array([1,0,0,0,0,0,0,0,0]) 

# initialize B 
# need a good enough intial guess for it to converge
# B = np.array([
#     # T     G      C     A 
#     [0.25, 0.25, 0.25, 0.25], # N
#     [0.7, 0.1, 0.1, 0.1], # M1
#     [0.1, 0.7, 0.1, 0.1], # M2
#     [0.1, 0.1, 0.1, 0.7], # M3
#     [0.1, 0.1, 0.7, 0.1], # M4
#     [0.1, 0.7, 0.1, 0.1], # M5
#     [0.7, 0.1, 0.1, 0.1], # M6
#     [0.1, 0.1, 0.7, 0.1], # M7
#     [0.1, 0.1, 0.1, 0.7], # M8
# ])
# initialize randomly
B = np.random.rand(9,4)
B = B/np.sum(B, axis=1)[:,np.newaxis]
print("B initialized to: ", np.round(B, 3))

# Initialize Alpha, Beta, C, and Gamma matricess
R, T = np.shape(training_seq)
n = np.shape(P)[0]
Alpha=np.empty((R, n, T))
Beta=np.empty((R, n, T))
Cmat=np.empty((R, T-1, n, n)) # first dim of Cmat is T-1?
Gamma=np.empty((R, n, T))

# Initalize them
Estep(Alpha, Beta, Cmat, Gamma, Pi, P, B, training_seq, T, R)
Mstep(Alpha, Beta, Cmat, Gamma, Pi, P, B, training_seq, T, R, free)

# Run the EM algorithm
epsilon = 0.0001 # convergence criteria (difference between current and next likelihood)
max_iterations = 1000 # cut off if it still hasn't converged

EMalgorithm(Alpha, Beta, Cmat, Gamma, Pi, P, B, training_seq, T, R, free, epsilon, max_iterations)

# Print Out the Learned Parameters of the Model
print(np.round(Pi, 3))
print(np.round(P, 3))
print(np.round(B, 3))

# Save the learned parameters
np.savetxt("MotifsPi.txt", Pi)
np.savetxt("MotifsP.txt", P)
np.savetxt("MotifsB.txt", B)

# # Test on test sequences
# print(viterbi(P, B, Pi, seq[900,:], n, T))

# hiddenStatesFile = open('simulatedHiddenStates.txt','r')
# hiddenStates = hiddenStatesFile.readlines()

# true = hiddenStates[900]
# pred="""0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 0 0 0 0 0 1 2 3
#  4 5 6 7 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4
#  5 6 7 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 0 0 0 1
#  2 3 4 5 6 7 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"""
# code2={"0":"N", "1":"M1", "2":"M2", "3":"M3", "4":"M4", "5":"M5", "6":"M6", "7":"M7", "8":"M8", " ":" ", "\n":""}
# predout = [code2[s] for s in pred]
# print()
# print("pred -", "".join(predout)[0:100])
# print()
# print("true -", true[0:100])
# print()
# print("pred -","".join(predout)[100:200])
# print()
# print("true -",true[100:200])
# print()
# print("pred -","".join(predout)[200:250])
# print()
# print("true -",true[200:250])
# print()
