
import numpy as np
import matplotlib.pyplot as plt

# Value function iteration

'''
------------------------------------------------------------------------
Value Function Iteration    
------------------------------------------------------------------------
VFtol     = scalar, tolerance required for value function to converge
VFdist    = scalar, distance between last two value functions
VFmaxiter = integer, maximum number of iterations for value function
V         = vector, the value functions at each iteration
Vmat      = matrix, the value for each possible combination of e and e'
Vstore    = matrix, stores V at each iteration 
VFiter    = integer, current iteration number
TV        = vector, the value function after applying the Bellman operator
PF        = vector, indicies of choices of e' for all e 
VF        = vector, the "true" value function
------------------------------------------------------------------------
'''

def VFI(params, e_grid, u, c):
    beta, sigma,m,n, k = params
    VFiter = 1
    VFmaxiter = 1000
    VFtol = 0.0001
    VFdist = 7 
    V = np.zeros_like(e_grid)
    Vmat = np.zeros_like(u)
    while VFdist > VFtol and VFiter < VFmaxiter:
        for i, e in enumerate(e_grid):
            for j, e_prime in enumerate(e_grid):
                c = (m+ n*e)-(k*e_prime)
                #u[i, j] = (c ** (1 - sigma)) / (1 - sigma)
                u[i, j] = np.log(c)
                Vmat[i, j]= u[i, j] + beta * V[j]
        TV = Vmat.max(axis=1)
        PF = np.argmax(Vmat, axis=1)
        VFdist = (np.abs(V - TV)).max()
        V = TV
        VFiter += 1
        print('iteration = ', iter)
     
    if VFiter < VFmaxiter:
        print('value function converged')
        print('difference = ', VFdist)
    else:
        print('value function did not converge')
    return(V, PF)


#opte = e_grid[PF] # tomorrow's optimal education size 
#optc = e_grid - opte # optimal consumption - get consumption through the transition eqn
 

