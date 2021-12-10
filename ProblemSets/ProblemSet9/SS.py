import numpy as np
import scipy.optimize as opt

def get_L(nvec): 
    
    '''
    Function to compute aggregate
    labor supplied
    '''
    L = nvec.sum()
    return L

def get_K(bvec): 
    
    '''
    Function to compute aggregate
    capital supplied
    '''
    K = bvec.sum()
    return K

#Firms
def get_r(K, L, params): 
    '''
    Compute the interest rate from
    the firm's FOC
    '''
    A, alpha, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta
    return r

def get_w(r, params): 
    
    '''
    Solve for the w that is consistent
    with r from the firm's FOC
    '''
    
    A, alpha, delta = params
    w = (1 - alpha) * A * (((alpha * A) / (r + delta)) ** (alpha / (1 - alpha)))
    return w


# Household
def get_c(c1, params): 
    '''
    function for implied steady-state consumption, given initial guess c1
    '''
    r, beta, sigma, p= params
    cvec = np.zeros(p)
    cvec[0] = c1
    cs = c1
    s = 0
    while s < p - 1:
        cvec[s + 1] = cs * (beta * (1 + r[s + 1])) ** (1 / sigma)
        cs = cvec[s + 1]
        s += 1
    return cvec

def MU_c_func(cvec, sigma):
    '''
    Marginal utility of consumption
    '''
    mu_c = cvec ** (-sigma)
    epsilon = 1e-4
    m1 = (-sigma) * epsilon ** (-sigma - 1)
    m2 = epsilon ** (-sigma) - m1 * epsilon
    c_cnstr = cvec < epsilon
    mu_c[c_cnstr] = m1 * cvec[c_cnstr] + m2
    return mu_c

def MU_n_func(nvec, params):
    l_tilda, b, upsilon = params
    epsilon_lb = 1e-6
    epsilon_ub = l_tilda - epsilon_lb
    
    '''
    Marginal utility of labor
    '''
    mu_n = ((b / l_tilda) * ((nvec / l_tilda) ** (upsilon - 1)) * (1 - ((nvec / l_tilda) ** upsilon)) **\
           ((1 - upsilon) / upsilon))

    m1 = (b * (l_tilda ** (-upsilon)) * (upsilon - 1) * (epsilon_lb ** (upsilon - 2)) * \
         ((1 - ((epsilon_lb / l_tilda) ** upsilon)) ** ((1 - upsilon) / upsilon)) * \
         (1 + ((epsilon_lb / l_tilda) ** upsilon) * ((1 - ((epsilon_lb / l_tilda) ** upsilon)) ** (-1))))
    m2 = ((b / l_tilda) * ((epsilon_lb / l_tilda) ** (upsilon - 1)) * \
         ((1 - ((epsilon_lb / l_tilda) ** upsilon)) ** ((1 - upsilon) / upsilon)) - (m1 * epsilon_lb))

    q1 = (b * (l_tilda ** (-upsilon)) * (upsilon - 1) * (epsilon_ub ** (upsilon - 2)) * \
         ((1 - ((epsilon_ub / l_tilda) ** upsilon)) ** ((1 - upsilon) / upsilon)) * \
         (1 + ((epsilon_ub / l_tilda) ** upsilon) * ((1 - ((epsilon_ub / l_tilda) ** upsilon)) ** (-1))))

    q2 = ((b / l_tilda) * ((epsilon_ub / l_tilda) ** (upsilon - 1)) * \
         ((1 - ((epsilon_ub / l_tilda) ** upsilon)) ** ((1 - upsilon) / upsilon)) - (q1 * epsilon_ub))

    nl_cstr = nvec < epsilon_lb
    nu_cstr = nvec > epsilon_ub

    mu_n[nl_cstr] = m1 * nvec[nl_cstr] + m2
    mu_n[nu_cstr] = q1 * nvec[nu_cstr] + q2
    return mu_n

def get_b(cvec, nvec, r, w, p, bs = 0.0): 
    '''
    function for calculating lifetime savings, given consumption and labor decisions
    '''
    bvec = np.zeros(p)
    s = 0
    bvec[0] = bs
    while s < p - 1:
        bvec[s + 1] = (1 + r[s]) * bs + w[s] * nvec[s] - cvec[s]
        bs = bvec[s + 1]
        s += 1
    return bvec

def get_n_errors(nvec, *args):
    cvec, sigma, l_tilda, b, upsilon, w = args
    mu_c = MU_c_func(cvec, sigma)
    mu_n = MU_n_func(nvec, (l_tilda, b, upsilon))
    n_errors = w * mu_c - mu_n
    return n_errors

def get_n(params): 
    cvec, sigma, l_tilda, b, upsilon, w=params
    n_args = params
    S = 80
    n_guess = 0.5 * l_tilda * np.ones(S)
    result = opt.root(get_n_errors, n_guess, args = (n_args), method = 'lm')
    if result.success:
        nvec = result.x
    else:
        raise ValueError("failed to find an appropriate labor decision")
    return nvec

def get_b_last(c1, *args): 
    '''
    function for last-period savings, given intial guess c1
    '''
    r, w, beta, sigma, l_tilda, b, upsilon, p = args
    cvec = get_c(c1, (r, beta, sigma, p))
    nvec = get_n((cvec, sigma, l_tilda, b, upsilon, w))
    bvec = get_b(cvec, nvec, r, w, p, bs = 0.0)
    b_last = (1 + r[-1]) * bvec[-1] + w[-1] * nvec[-1] - cvec[-1]
    return b_last





