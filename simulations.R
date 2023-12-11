library(seqHMM)

# Parameters for the HMM
emission_probs <- t(matrix(c(0.25, 0.25, 0.25, 0.25,
                           1, 0, 0, 0,
                           0.034, 0.966, 0, 0,
                           0, 0, 0, 1,
                           0, 0, 1, 0,
                           0, 0.966, 0.034, 0,
                           0.931, 0.035, 0.034, 0,
                           0, 0.034, 0.552, 0.414,
                           0.241, 0.138, 0.035, 0.586), 4, 9))
transition_probs <- matrix(data=0,9,9)
transition_probs[1,1]=0.95
transition_probs[1,2]=0.05
trans=matrix(c(2:8,9,3:9,1),8,2)
transition_probs[trans]=1
initial_probs <- c(1,0,0,0,0,0,0,0,0)

# Setting the seed for simulation
set.seed(1)

# Simulating sequences
sim <- simulate_hmm(
  n_sequences = 1000, initial_probs = initial_probs,
  transition_probs = transition_probs,
  emission_probs = emission_probs,
  sequence_length = 250
)

# turn observations into DNA sequences
m = matrix(data=0,nrow=1000,ncol=250)
for(i in 1:250){
  m[,i] = sim$observations[[i]]
}
m

code <- c("T","G","C","A")
m <- apply(m,c(1,2),function(i){code[i]})
seqs <- apply(m,1,paste,collapse="")
write(seqs, file = "simulatedSequences.txt")

# turn states into txt file of sequences
m = matrix(data=0,nrow=1000,ncol=250)
for(i in 1:250){
  m[,i] = sim$states[[i]]
}
m

code <- c("N ","M1 ","M2 ","M3 ", "M4 ", "M5 ", "M6 ", "M7 ", "M8 ")
m <- apply(m,c(1,2),function(i){code[i]})
seqs <- apply(m,1,paste,collapse="")
write(seqs, file = "simulatedHiddenStates.txt")

# Plot the motifs
par(mfrow=c(1,2))
library(motifStack)
trueMotif <- matrix(c(1, 0, 0, 0,
                        0.034, 0.966, 0, 0,
                        0, 0, 0, 1,
                        0, 0, 1, 0,
                        0, 0.966, 0.034, 0,
                        0.931, 0.035, 0.034, 0,
                        0, 0.034, 0.552, 0.414,
                        0.241, 0.138, 0.035, 0.586), 4, 8)
rownames(trueMotif) <- c("T","G","C","A")
motif <- new("pcm", mat=trueMotif, name="True CRE Motif")
plot(motif, ic.scale=FALSE, ylab="probability")

inferredMotif <- matrix(c(0.266, 0.253, 0.238, 0.243,
                      0.277, 0.246, 0.242, 0.236,
                      0.001, 0.001, 0.001, 0.991,
                      0, 0, 0.953, 0.047,
                      0.991, 0.002, 0.007, 0,
                      0, 0.997, 0.003, 0,
                      0, 0.03, 0.97, 0,
                      0, 0.044, 0.039, 0.917), 4, 8)
rownames(inferredMotif) <- c("T","G","C","A")
motif <- new("pcm", mat=inferredMotif, name="Inferred CRE Motif")
plot(motif, ic.scale=FALSE, ylab="probability")



