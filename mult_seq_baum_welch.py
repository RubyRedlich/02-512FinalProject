import numpy as np

# updates Alpha in place (doesn't return anything)
def forward_algorithm(Alpha, Pi, P, B, seq, T):
    Alpha[:,:,0]=(B[:,seq[:,0]]*Pi[:,np.newaxis]).T
    for j in range(1,T): 
        Alpha[:,:,j] = B[:,seq[:,j]].T * (Alpha[:,:,j-1] @ P)

# updates Beta in place (doesn't return anything)
def backward_algorithm(Beta, P, B, seq, T):
    Beta[:,:,T-1] = 1
    for j in range(T-2, -1, -1):
        Beta[:,:,j] = (P @ (B[:,seq[:,j+1]] * Beta[:,:,j+1].T)).T

# convergence criteria
def prob_obs(Alpha, T):
    # Pr{x|HMM}, for convergence
    # Alpha R x n
    return np.sum(np.log(np.sum(Alpha[:,:,T-1], axis=1))) # use log likelihood otherwise numbers will get really small?


# updates Cmat in place
def update_C_matrix(Alpha, Beta, Cmat, P, B, seq, T, R):
    for r in range(R):
        for t in range(T-1): 
            denom = np.sum(Alpha[r,:,t] * np.sum(P * (Beta[r,:,t+1] * B[:,seq[r,t+1]]),axis=1)) # top pg 256
            Cmat[r,t,:,:] = P * Alpha[r,:,t][:,np.newaxis] # multiply every column of P by col t of Alpha
            Cmat[r,t,:,:] = Cmat[r,t,:,:] * (Beta[r,:,t+1] * B[:,seq[r,t+1]])# multiply every row of P by col (t+1) of Beta &  multiply every row of P by col seq[t+1] of B
            Cmat[r,t,:,:] = Cmat[r,t,:,:]/denom

# updates Gamma in place
def update_Gamma_matrix(Cmat, Gamma, T, R):
    # mid page 256
    for r in range(R):
        Gamma[r,:,1:T] = np.sum(Cmat[r,:,:,:],axis=1).T # eq for t > 0
        Gamma[r,:,0] = np.sum(Cmat[r,0,:,:],axis=1) # eq for t < T

# updates Pi in place
def update_Pi(Pi, Gamma, R):
    Pi[:] = np.sum(Gamma[:,:,0],axis=0) / R

# updates P in place
# free describes which transitions in P should be upated (the rest remain fixed)
def update_P_matrix(P, Cmat, Gamma, free, T):
    num = np.sum(np.sum(Cmat,axis=1),axis=0)
    d = np.sum(np.sum(Gamma[:,:,0:(T-1)], axis=2),axis=0)[:,np.newaxis]
    # for rows where division by 0 would occur, use the old values of P
    tmp = num/d
    rows = np.where(d==0)
    tmp[rows,:] = P[rows,:]
    P[free] = tmp[free]

def update_B_matrix(Gamma, B, seq):
    d = np.sum(np.sum(Gamma, axis=2), axis=0) 
    for k in range(np.shape(B)[1]):
        B[:,k] = np.sum(np.sum(Gamma*(seq==k)[:,np.newaxis,:],axis=2),axis=0) / d

def Estep(Alpha, Beta, Cmat, Gamma, Pi, P, B, seq, T, R):
    forward_algorithm(Alpha, Pi, P, B, seq, T)
    backward_algorithm(Beta, P, B, seq, T)
    update_C_matrix(Alpha, Beta, Cmat, P, B, seq, T, R)
    update_Gamma_matrix(Cmat, Gamma, T, R)

def Mstep(Alpha, Beta, Cmat, Gamma, Pi, P, B, seq, T, R, free):
    update_Pi(Pi, Gamma, R)
    update_P_matrix(P, Cmat, Gamma, free, T)
    update_B_matrix(Gamma, B, seq)

def EMalgorithm(Alpha, Beta, Cmat, Gamma, Pi, P, B, seq, T, R, free, epsilon, maxits):
    curr_lik = prob_obs(Alpha, T)    
    Estep(Alpha, Beta, Cmat, Gamma, Pi, P, B, seq, T, R)
    Mstep(Alpha, Beta, Cmat, Gamma, Pi, P, B, seq, T, R, free)
    new_lik = prob_obs(Alpha, T)
    cnt = 0
    while abs(curr_lik - new_lik) > epsilon:
        if cnt >= maxits:
            print("maximum iterations reached without converging")
            return
        curr_lik = new_lik
        Estep(Alpha, Beta, Cmat, Gamma, Pi, P, B, seq, T, R)
        Mstep(Alpha, Beta, Cmat, Gamma, Pi, P, B, seq, T, R, free)
        new_lik = prob_obs(Alpha, T)
        cnt += 1


