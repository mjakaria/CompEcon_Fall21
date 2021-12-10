import SS as ss
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


# Household Parameters
yrs_live = 80
S = 80
beta_annual = .90
beta = beta_annual ** (yrs_live / S)
sigma = 1.5
l_tilda = 1.0
b = 0.501
upsilon = 1.554

# Firms Parameters
alpha = 0.35
A = 1.0
delta_annual = 0.05
delta = 1.0 - ((1.0 - delta_annual) ** (yrs_live / S))


# Find Steady State
max_iter = 500
tol = 1e-9
xi = 0.1
abs_ss = 1
ss_iter = 0
r_old = 0.06
while abs_ss > tol and ss_iter < max_iter:
    ss_iter += 1
    r_old = r_old * np.ones(S)
    w_old = ss.get_w(r_old, (A, alpha, delta)) * np.ones(S)
    
  # Calculate household decisions that make last-period savings zero
    c1_guess = 0.1
    c1_args = (r_old, w_old, beta, sigma, l_tilda, b, upsilon, S)
    result_c1 = opt.root(ss.get_b_last, c1_guess, args = (c1_args))
    if result_c1.success:
        c1 = result_c1.x
    else:
        raise ValueError("failed to find an appropriate initial consumption")
 
    # Calculate aggregate supplies for capital and labor
    
    cvec = ss.get_c(c1, (r_old, beta, sigma, S))
    nvec = ss.get_n((cvec, sigma, l_tilda, b, upsilon, w_old))
    bvec = ss.get_b(cvec, nvec, r_old, w_old, S)
    K = ss.get_K(bvec)
    L = ss.get_L(nvec)
    r_new = ss.get_r(K, L, (A, alpha, delta))
   
    # Check market clearing
    
    abs_ss = ((r_new - r_old) ** 2).max()
   
    # Update guess
    r_old = xi * r_new + (1 - xi) * r_old
    print('iteration:', ss_iter, ' squared distance: ', abs_ss)

r_ss = r_old * np.ones(S)
w_ss = ss.get_w(r_ss, (A, alpha, delta)) * np.ones(S)


'''
Now Plot the saving, labor supply, and consumption for each age S
'''

# I run the codes below separately to get three separate figure#

plot = True
if plot:
    plt.plot ( np.arange(80), bvec, 'go--', color = 'blue', label = 'savings')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Steady-State distribution of Savings', fontsize=20)
    plt.xlabel('age')
    plt.ylabel('units of saving')
    plt.legend()
    
plot = True
if plot:
    plt.plot ( np.arange(80), nvec, 'go--', label = 'labor supply')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Steady-State distribution of Labor Supply', fontsize=20)
    plt.xlabel('age')
    plt.ylabel('labor supply')
    plt.legend()   
    

plot = True
if plot:  
    
    plt.plot (np.arange(80), cvec, 'go--', color = 'red', label = 'consumption')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Steady-State distribution of Consumption', fontsize=20)
    plt.xlabel('age')
    plt.ylabel('units of consumption')
    plt.legend()


    
