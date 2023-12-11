import numpy as np

def viterbi(P, B, Pi, seq, n, T):
    M = np.empty((n,T))
    A = np.empty((n,T))
    # initialize first column of M
    M[:,0] = B[:,seq[0]] * Pi
    # fill in the matrices storing probs and states
    for j in range(1,T):
        for i in range(n):
            v = (M[:,j-1] * P[:,i]) * B[i,seq[j]]
            M[i,j] = np.max(v)
            A[i,j] = np.argmax(v)
    # find max final state and walk backwards through A
    Q = np.empty((T,), dtype=np.int8)
    Q[T-1] = int(np.argmax(M[:,T-1]))
    for j in range(T-2, -1, -1):
        Q[j] = int(A[Q[j+1],j+1])
    return Q