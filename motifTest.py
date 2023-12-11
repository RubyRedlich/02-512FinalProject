from viterbi import * 

# Load in parameters
Pi = np.loadtxt('MotifsPi.txt',delimiter=' ')
P = np.loadtxt('MotifsP.txt',delimiter=' ')
B = np.loadtxt('MotifsB.txt',delimiter=' ')

# Load in testing sequences
sequences = open('simulatedSequences.txt','r')
seqstrs = sequences.readlines()

# strip new line character from end of each sequence
seqstrs = [seq.strip() for seq in seqstrs]

# convert to a numpy array
code = {"A":0, "C":1, "G":2, "T":3}
seq = [[code[b] for b in seqstrs[i]] for i in range(len(seqstrs))]
seq = np.array(seq)

# use the last 100 sequences for testing
testing_seq = seq[900:1000,:]

# read in true values 
hiddenStatesFile = open('simulatedHiddenStates.txt','r')
hiddenStates = hiddenStatesFile.readlines()
hiddenStates = [seq.strip() for seq in hiddenStates]

# test predictions vs true values and measure accuracy
n = np.shape(P)[0]
T = np.shape(testing_seq)[1]
stateToInt1={"N":0, "M1":1, "M2":2, "M3":3, "M4":4, "M5":5, "M6":6, "M7":7, "M8":8}
# 1st column - exact match
# 2nd column - # motifs / total motifs
# 3rd column - non-motif vs. motif
accuracy = np.zeros((100,3))
for i in range(np.shape(testing_seq)[0]):
    pred = viterbi(P, B, Pi, testing_seq[i,:], n, T)
    true = np.array([stateToInt1[q] for q in hiddenStates[900+i].split()])  # because we're testing on the LAST 900 sequences
    accuracy[i,0] = np.sum(true==pred)/len(pred)
    accuracy[i,1] = np.sum(np.logical_and(true==1,pred==1))/np.sum(true==1)
    accuracy[i,2] = np.sum((pred==0)==(true==0))/len(pred)

print(np.mean(accuracy,axis=0))
