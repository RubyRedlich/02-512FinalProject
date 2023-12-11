import numpy as np

# updates Alpha in place (doesn't return anything)
def forward_algorithm(Alpha, Pi, P, B, seq, T):
    Alpha[:,0] = B[:,seq[0]] * Pi
    for j in range(1,T): 
        Alpha[:,j] = B[:,seq[j]] * (Alpha[:,j-1] @ P)

# updates Beta in place (doesn't return anything)
def backward_algorithm(Beta, P, B, seq, T):
    Beta[:,T-1] = 1
    for j in range(T-2, -1, -1):
        Beta[:,j] = P @ (B[:,seq[j+1]] * Beta[:,j+1])

# convergence criteria
def prob_obs(Alpha, T):
    # Pr{x|HMM}, for convergence
    return np.log(np.sum(Alpha[:,T-1])) # use log likelihood otherwise numbers will get really small?

# updates Cmat in place
def update_C_matrix(Alpha, Beta, Cmat, P, B, seq, T):
    for t in range(T-1): 
        denom = np.sum(Alpha[:,t] * np.sum(P * (Beta[:,t+1] * B[:,seq[t+1]]),axis=1)) # top pg 256
        Cmat[t,:,:] = P * Alpha[:,t][:,np.newaxis] # multiply every column of P by col t of Alpha
        Cmat[t,:,:] = Cmat[t,:,:] * (Beta[:,t+1] * B[:,seq[t+1]])# multiply every row of P by col (t+1) of Beta &  multiply every row of P by col seq[t+1] of B
        Cmat[t,:,:] = Cmat[t,:,:]/denom

# updates Gamma in place
def update_Gamma_matrix(Cmat, Gamma, T):
    # mid page 256
    Gamma[:,1:T] = np.sum(Cmat,axis=1).T # eq for t > 0
    Gamma[:,0] = np.sum(Cmat[0,:,:],axis=1) # eq for t < T

# updates Pi in place
def update_Pi(Pi, Gamma):
    Pi[:] = Gamma[:,0]

# updates P in place
# free describes which transitions in P should be upated (the rest remain fixed)
def update_P_matrix(P, Cmat, Gamma, free, T):
    num = np.sum(Cmat,axis=0)
    d = np.sum(Gamma[:,0:(T-1)], axis=1)[:,np.newaxis]
    # for rows where division by 0 would occur, use the old values of P
    tmp = num/d
    rows = np.where(d==0)
    tmp[rows,:] = P[rows,:]
    P[free] = tmp[free]

def update_B_matrix(Gamma, B, seq):
    d = np.sum(Gamma, axis=1)
    for k in range(np.shape(B)[1]):
        B[:,k] = np.sum(Gamma[:,seq==k],axis=1) / d

def Estep(Alpha, Beta, Cmat, Gamma, Pi, P, B, seq, T):
    forward_algorithm(Alpha, Pi, P, B, seq, T)
    backward_algorithm(Beta, P, B, seq, T)
    update_C_matrix(Alpha, Beta, Cmat, P, B, seq, T)
    update_Gamma_matrix(Cmat, Gamma, T)

def Mstep(Alpha, Beta, Cmat, Gamma, Pi, P, B, seq, T, free):
    update_Pi(Pi, Gamma)
    update_P_matrix(P, Cmat, Gamma, free, T)
    update_B_matrix(Gamma, B, seq)

def EMalgorithm(Alpha, Beta, Cmat, Gamma, Pi, P, B, seq, T, free, epsilon, maxits):
    curr_lik = prob_obs(Alpha, T)    
    Estep(Alpha, Beta, Cmat, Gamma, Pi, P, B, seq, T)
    Mstep(Alpha, Beta, Cmat, Gamma, Pi, P, B, seq, T, free)
    new_lik = prob_obs(Alpha, T)
    cnt = 0
    while abs(curr_lik - new_lik) > epsilon:
        if cnt >= maxits:
            print("maximum iterations reached without converging")
            return
        curr_lik = new_lik
        Estep(Alpha, Beta, Cmat, Gamma, Pi, P, B, seq, T)
        Mstep(Alpha, Beta, Cmat, Gamma, Pi, P, B, seq, T, free)
        new_lik = prob_obs(Alpha, T)
        cnt += 1


