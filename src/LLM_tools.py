import numpy as np
import random
import math
import itertools

def error_perturb(stg,error):
    return stg*(1-error) + (1-stg)*error

import numpy as np

def kosaraju_scc(adj):
    """Return list of SCCs (each as sorted list of nodes) using Kosaraju's algorithm.
    adj is adjacency list or adjacency matrix (we accept numpy array)."""
    n = adj.shape[0]
    # build adjacency lists
    graph = [list(np.nonzero(adj[i] > 0)[0]) for i in range(n)]
    rev_graph = [list(np.nonzero(adj[:, i] > 0)[0]) for i in range(n)]

    visited = [False]*n
    order = []

    def dfs1(u):
        visited[u] = True
        for v in graph[u]:
            if not visited[v]:
                dfs1(v)
        order.append(u)

    for i in range(n):
        if not visited[i]:
            dfs1(i)

    visited = [False]*n
    components = []

    def dfs2(u, comp):
        visited[u] = True
        comp.append(u)
        for v in rev_graph[u]:
            if not visited[v]:
                dfs2(v, comp)

    for u in reversed(order):
        if not visited[u]:
            comp = []
            dfs2(u, comp)
            components.append(sorted(comp))

    return components


def find_closed_sccs(P, tol=1e-12):
    """Return closed SCCs (absorbing subsets) as lists of indices."""
    sccs = kosaraju_scc(P)
    closed = []
    for scc in sccs:
        scc_set = set(scc)
        outgoing = False
        for i in scc:
            # any outgoing probability to a state not in scc?
            outs = np.where(P[i] > tol)[0]
            if any(j not in scc_set for j in outs):
                outgoing = True
                break
        if not outgoing:
            closed.append(scc)
    return closed


def stationary_distribution(P_sub, tol=1e-12):
    """Compute stationary distribution for the (closed) submatrix P_sub."""
    # Solve pi = pi P_sub  subject to sum(pi)=1
    n = P_sub.shape[0]
    # Transpose equation: (P_sub^T - I^T) pi^T = 0  with sum constraint
    A = P_sub.T - np.eye(n)
    # Replace last row with ones for normalization
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    # Solve least squares (handles degenerate numerical cases)
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    # clip small negative noise and renormalize
    pi = np.clip(pi, 0, None)
    if pi.sum() == 0:
        # fallback to uniform if numerical issue
        pi = np.ones(n) / n
    else:
        pi = pi / pi.sum()
    return pi


def absorption_probs_and_stationaries(P, p0, tol=1e-12):
    """Return (closed_subsets, subset_absorption_probs, subset_stationaries)."""
    n = P.shape[0]
    closed = find_closed_sccs(P, tol=tol)

    # Flatten absorbing states
    absorbing_states = sorted([s for comp in closed for s in comp])
    transient_states = [i for i in range(n) if i not in absorbing_states]

    # If no transient states: initial mass already in absorbing classes
    if len(transient_states) == 0:
        subset_probs = [np.sum(p0[comp]) for comp in closed]
        stationaries = [stationary_distribution(P[np.ix_(comp, comp)]) for comp in closed]
        return closed, subset_probs, stationaries

    # Partition
    Q = P[np.ix_(transient_states, transient_states)]
    R = P[np.ix_(transient_states, absorbing_states)]

    # Fundamental matrix N = (I - Q)^{-1}
    I = np.eye(Q.shape[0])
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        # fallback to pseudo-inverse if singular numerically
        N = np.linalg.pinv(I - Q)

    B = N @ R  # absorption probabilities from transient -> absorbing states

    p_trans = p0[transient_states]
    p_abs = p0[absorbing_states]

    absorption_state_probs = p_trans @ B + p_abs  # final prob of being in each absorbing state

    # Map absorption_state_probs back to subsets
    # absorption_state_probs index corresponds to absorbing_states list order
    # Build a dict from state -> index
    state_to_idx = {s: idx for idx, s in enumerate(absorbing_states)}
    subset_probs = []
    for comp in closed:
        idxs = [state_to_idx[s] for s in comp]
        subset_probs.append(np.sum(absorption_state_probs[idxs]))

    stationaries = [stationary_distribution(P[np.ix_(comp, comp)]) for comp in closed]

    return closed, subset_probs, stationaries


def memory_one_payoff(firststg,secondstg,delta,error,R,S,T,P):

    stg1 = error_perturb(firststg,error); stg2 = error_perturb(secondstg,error);
    v0 =np.array([stg1[0]*stg2[0],stg1[0]*(1-stg2[0]),(1-stg1[0])*stg2[0],(1-stg1[0])*(1-stg2[0])])
    q = stg1[1:]; p = stg2[1:]
    M = np.zeros((4,4))
    M[0] = [q[0]*p[0], q[0]*(1-p[0]), (1-q[0])*p[0], (1-q[0])*(1-p[0])]
    M[1] = [q[1]*p[2], q[1]*(1-p[2]), (1-q[1])*p[2], (1-q[1])*(1-p[2])]
    M[2] = [q[2]*p[1], q[2]*(1-p[1]), (1-q[2])*p[1], (1-q[2])*(1-p[1])]
    M[3] = [q[3]*p[3], q[3]*(1-p[3]), (1-q[3])*p[3], (1-q[3])*(1-p[3])]


    if delta < 1:
        
        vect = np.eye(M.shape[0])
        inv_term = np.linalg.inv(vect - delta * M)
        w = (1 - delta) * v0 @ inv_term
        return w[0]*R + w[1]*S + w[2]*T + w[3]*P, w[0]*R + w[1]*T + w[2]*S + w[3]*P
    
    
    elif delta==1.0:

        absorbing_subsets, probs, stationaries = absorption_probs_and_stationaries(M, v0)

        if (len(absorbing_subsets)==1) and (absorbing_subsets[0]==[0,1,2,3]):
            q1=stg2[1]; q2=stg2[2]; q3=stg2[3]; q4=stg2[4]
            p1=stg1[1]; p2=stg1[2]; p3=stg1[3]; p4=stg1[4]
            pay1 = (P*(-1 + q2 - p3*q2 + p3*q1*q2 + p3*q3 - p3*q2*q3 + p1*p3*q2*q3 -p1*q1*(-1 + q2 + p3*q3) +       p2*(1 + p1*(-1 + q2)*(q1 - q3) - q3 - p3*q3 + p3*q1*q3 +          q2*(-1 + p3 - p3*q1 + q3))) - p3*p4*q2*q3*R - p4*q4*R - p3*q2*q4*R +    p4*q2*q4*R + p3*p4*q3*q4*R - p4*S + p1*p4*q1*S + p4*q2*S - p3*p4*q2*S -    p1*p4*q1*q2*S + p3*p4*q1*q2*S - p3*q4*S + p4*q4*S - p1*p4*q4*S + p3*p4*q4*S +    p1*p3*q1*q4*S - p3*p4*q1*q4*S + p3*q2*q4*S - p1*p3*q2*q4*S - p4*q2*q4*S +    p1*p4*q2*q4*S - p4*q3*T + p1*p4*q1*q3*T - q4*T + p4*q4*T + p1*q1*q4*T -    p4*q1*q4*T + p4*q3*q4*T - p1*p4*q3*q4*T +    p2*p4*(q3 - q4)*((-1 + q2)*R + T - q1*T) +    p2*q4*(p3*(q2 - q3)*R - (-1 + p1*(q1 - q3) + q3)*T))/ (-1 + p1*q1 + q2 - p3*q2 - p1*q1*q2 + p3*q1*q2 + p3*q3 - p1*p3*q1*q3 -    p3*q2*q3 + p1*p3*q2*q3 - q4 - p3*q4 + p1*q1*q4 + p1*p3*q1*q4 - p1*p3*q2*q4 +    p4*(-1 + q2 + p3*q2*(-1 + q1 - q3) - q3 - p1*(-1 + q2 - q3)*(q1 - q4) + q4 +       p3*q4 - q1*q4 - p3*q1*q4 + q3*q4 + p3*q3*q4) +    p2*(1 - q3 - p3*q3 + p3*q1*q3 - p4*q1*q3 + p1*(q1 - q3)*(-1 + q2 - q4) + q4 +       p4*q1*q4 - q3*q4 - p3*q3*q4 +       q2*(-1 + p3 - p3*q1 + q3 + p4*q3 + p3*q4 - p4*q4)))
            q1=stg1[1]; q2=stg1[2]; q3=stg1[3]; q4=stg1[4]
            p1=stg2[1]; p2=stg2[2]; p3=stg2[3]; p4=stg2[4]
            pay2 = (P*(-1 + q2 - p3*q2 + p3*q1*q2 + p3*q3 - p3*q2*q3 + p1*p3*q2*q3 -p1*q1*(-1 + q2 + p3*q3) +       p2*(1 + p1*(-1 + q2)*(q1 - q3) - q3 - p3*q3 + p3*q1*q3 +          q2*(-1 + p3 - p3*q1 + q3))) - p3*p4*q2*q3*R - p4*q4*R - p3*q2*q4*R +    p4*q2*q4*R + p3*p4*q3*q4*R - p4*S + p1*p4*q1*S + p4*q2*S - p3*p4*q2*S -    p1*p4*q1*q2*S + p3*p4*q1*q2*S - p3*q4*S + p4*q4*S - p1*p4*q4*S + p3*p4*q4*S +    p1*p3*q1*q4*S - p3*p4*q1*q4*S + p3*q2*q4*S - p1*p3*q2*q4*S - p4*q2*q4*S +    p1*p4*q2*q4*S - p4*q3*T + p1*p4*q1*q3*T - q4*T + p4*q4*T + p1*q1*q4*T -    p4*q1*q4*T + p4*q3*q4*T - p1*p4*q3*q4*T +    p2*p4*(q3 - q4)*((-1 + q2)*R + T - q1*T) +    p2*q4*(p3*(q2 - q3)*R - (-1 + p1*(q1 - q3) + q3)*T))/ (-1 + p1*q1 + q2 - p3*q2 - p1*q1*q2 + p3*q1*q2 + p3*q3 - p1*p3*q1*q3 -    p3*q2*q3 + p1*p3*q2*q3 - q4 - p3*q4 + p1*q1*q4 + p1*p3*q1*q4 - p1*p3*q2*q4 +    p4*(-1 + q2 + p3*q2*(-1 + q1 - q3) - q3 - p1*(-1 + q2 - q3)*(q1 - q4) + q4 +       p3*q4 - q1*q4 - p3*q1*q4 + q3*q4 + p3*q3*q4) +    p2*(1 - q3 - p3*q3 + p3*q1*q3 - p4*q1*q3 + p1*(q1 - q3)*(-1 + q2 - q4) + q4 +       p4*q1*q4 - q3*q4 - p3*q3*q4 +       q2*(-1 + p3 - p3*q1 + q3 + p4*q3 + p3*q4 - p4*q4)))
            return pay1,pay2
        
        else: 
            total_val_p1 = 0; total_val_p2 = 0
            for setcount in range(0,len(absorbing_subsets)):
                set = absorbing_subsets[setcount]
                prob_set = probs[setcount]
                stationary = stationaries[setcount]
                for statecount in range(0,len(set)):
                    total_val_p1 += prob_set*stationary[statecount]*[R,S,T,P][set[statecount]]
                    total_val_p2 += prob_set*stationary[statecount]*[R,T,S,P][set[statecount]]
            return total_val_p1,total_val_p2
        

def make_q_tilde(q):
    """Transforms q to match player 2's perspective."""
    qt = q.copy()
    qt[1], qt[2] = q[2], q[1]
    qt[4], qt[5], qt[6], qt[7] = q[8], q[10], q[9], q[11]
    qt[8], qt[9], qt[10], qt[11] = q[4], q[6], q[5], q[7]
    qt[13], qt[14] = q[14], q[13]
    return qt

def memory_two_payoff(firststg,secondstg,delta,error,R,S,T,P):
    """
    Calculates the payoff of a p-player against a q-player when both use memory-2 strategies
    for a finitely repeated prisoner's dilemma.
    
    Parameters:
    p, q : arrays of length 16
        Long-run strategies of the players for rounds 3,4,5,...
    p0, q0 : arrays of length 5
        Short-run strategies for rounds 1+2
    RR, SS, TT, PP : float
        Payoff parameters
    delta : float
        Discount factor
    
    Returns:
    pip, piq : float
        Expected payoffs for player p and q
    v : array
        16-dim vector of state probabilities
    coop : float
        Expected cooperation rate
    """

    p = firststg[5:]
    q = secondstg[5:]
    p0 = firststg[0:5]; 
    q0 = secondstg[0:5];

    ###quick fixes

    if (p[0]==1) and (p0[0]==1) and (p0[1]==1) and (q[0]==1) and (q0[0]==1) and (q0[1]==1):
        return R,R

    if (p[-1]==0) and (p0[0]==0) and (p0[-1]==0) and (q[-1]==0) and (q0[0]==0) and (q0[-1]==0):
        return P,P

    ep = error

    # Preprocess strategies
    p = (1 - ep) * p + ep * (1 - p)
    q = (1 - ep) * q + ep * (1 - q)
    p0 = (1 - ep) * p0 + ep * (1 - p0)
    q0 = (1 - ep) * q0 + ep * (1 - q0)
    q = make_q_tilde(q)

    # Transition matrix B
    B = np.zeros((16, 16))
    for i in range(4):
        for j in range(4):
            pij = p[4 * i + j]
            qij = q[4 * i + j]
            B[4 * i + j, i] = pij * qij
            B[4 * i + j, 4 + i] = pij * (1 - qij)
            B[4 * i + j, 8 + i] = (1 - pij) * qij
            B[4 * i + j, 12 + i] = (1 - pij) * (1 - qij)

    # Initial state vector x after first 2 rounds
    x1 = np.array([
        p0[0]*q0[0], p0[0]*(1-q0[0]), (1-p0[0])*q0[0], (1-p0[0])*(1-q0[0])
    ])
    x1 = np.tile(x1, 4)

    x2 = np.array([
        p0[1]*q0[1], p0[2]*q0[3], p0[3]*q0[2], p0[4]*q0[4],
        p0[1]*(1-q0[1]), p0[2]*(1-q0[3]), p0[3]*(1-q0[2]), p0[4]*(1-q0[4]),
        (1-p0[1])*q0[1], (1-p0[2])*q0[3], (1-p0[3])*q0[2], (1-p0[4])*q0[4],
        (1-p0[1])*(1-q0[1]), (1-p0[2])*(1-q0[3]), (1-p0[3])*(1-q0[2]), (1-p0[4])*(1-q0[4])
    ])
    x = x1 * x2

    # Payoff vectors
    piv1 = np.array([R, S, T, P] * 4)
    piv2 = np.array([R, T, S, P] * 4)

    # Solve for v
    v = ((1 - delta) * x)@ np.linalg.inv(np.eye(16) - delta * B)

    pip = np.dot(v, piv1)
    piq = np.dot(v, piv2)

    #coop = sum(v[0:16:4]) + sum(v[1:16:4])/2 + sum(v[2:16:4])/2

    return pip, piq


def check_Nash_M1(stg,delta,error,R,S,T,P):
    """This returns a triple: (True/False for Nash, True/False for Partner and True/False for Rival)"""
    
    selfpayoff = memory_one_payoff(stg,stg,delta,error,R,S,T,P)[0]
    checkers = [np.array(p) for p in itertools.product([0, 1], repeat=5)]
    
    NASHFLAG = True
    RIVALFLAG = True 

    for stg2 in checkers:
        deviationpayoff, focalpayoff = memory_one_payoff(stg2,stg,delta,error,R,S,T,P)
        
        if deviationpayoff-selfpayoff > 10**-10:
            NASHFLAG = False

        if deviationpayoff-focalpayoff > 10**-10:
            RIVALFLAG = False

    if NASHFLAG and abs(selfpayoff-R)<10**-10:
        y = True 
    else:
        y = False

    z = True if RIVALFLAG else False 

    return (NASHFLAG,y,z)


def check_Nash_M2(stg,delta,error,R,S,T,P):
    selfpayoff = memory_two_payoff(stg,stg,delta,error,R,S,T,P)[0]
    checkers = [np.array(p) for p in itertools.product([0, 1], repeat=21)]
    for stg2 in checkers:
        deviationpayoff = memory_two_payoff(stg2,stg,delta,error,R,S,T,P)[0]
        if deviationpayoff-selfpayoff > 10**-10:
            return False
        else:
            continue

    return True

def invasion_process_percentage_M1(stg,delta,error,R,S,T,P,mut_number=10**6):
    a_ = memory_one_payoff(stg,stg,delta,error,R,S,T,P)[0]
    count = 0
    for _ in range(0,mut_number):
        opponent = np.random.rand(5)
        c_ = memory_one_payoff(opponent,stg,delta,error,R,S,T,P)[0]
        if a_ > c_:
            count += 1
    return count/mut_number

def invasion_process_percentage_M2(stg,delta,error,R,S,T,P,mut_number=10**6):
    a_ = memory_two_payoff(stg,stg,delta,error,R,S,T,P)[0]
    count = 0
    for _ in range(0,mut_number):
        opponent = np.random.rand(21)
        c_ = memory_two_payoff(opponent,stg,delta,error,R,S,T,P)[0]
        if a_ > c_:
            count += 1
    return count/mut_number



#TURNS OUT WE NEVER NEEDED THE FOLLOWING FUNCTION
def reorderingstrategy(stg):
    """having to write this function because LLM strategies need to be reordered in intuitive ordering"""
    newstrategy = np.zeros(21)
    newstrategy[0:5] = stg[0:5]
    for i in range(5,21):
        j=i-5
        newstrategy[5+((j%4)*4+(j//4))] = stg[i]
    return newstrategy




    



        
        