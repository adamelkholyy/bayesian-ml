#
# hmc_Lab for CM50268 Bayesian Machine Learning
# April 2022
# Python port of Radford Neal "one_step" code - original comments below
#
# SIMPLE IMPLEMENTATION OF HAMILTONIAN MONTE CARLO.
#
# Radford M. Neal, 2010.
#
# This program appears in Figure 2 of "MCMC using Hamiltonian dynamics",
# to appear in the Handbook of Markov Chain Monte Carlo.
#
# The arguments to the HMC function are as follows:
#
#   U          A function to evaluate minus the log of the density of the
#              distribution to be sampled, plus any constant - ie, the
#              "potential energy".
#
#   grad_U     A function to evaluate the gradient of U.
#
#   epsilon    The stepsize to use for the leapfrog steps.
#
#   L          The number of leapfrog steps to do to propose a new state.
#
#   current_q  The current state (position variables only).
#
# Momentum variables are sampled from independent standard normal
# distributions within this function.  The value return is the vector
# of new position variables (equal to current_q if the endpoint of the
# trajectory was rejected).
#
# This function was written for illustrative purposes.  More elaborate
# implementations of this basic HMC method and various variants of HMC
# are available from my web page, http://www.cs.utoronto.ca/~radford/
#
import numpy as np
import time


def gradient_check(x0, fn_error, fn_grad, *args):
    """
    Check that the error and gradient functions are consistent.
    
    :param x0: Array specifying point at which to test
    :param fn_error: Function object: the "error" or "energy"
    :param fn_grad: Function object, the gradient of the error wrt x
    :param args: Any optional arguments to be passed to both functions
    """
    #
    delt = 1e-6
    alert_rel_red = 4
    alert_rel_yellow = 6
    #
    N = len(x0)
    #
    x = x0.copy()
    grad = fn_grad(x, *args)
    gnum = np.empty(N)
    #
    print('{0:13s} {1:13s} {2:13s} Acc.'.format('Calc.', 'Numeric', 'Delta'))
    for n in range(N):
        x[n] += delt
        fplus = fn_error(x, *args)
        x[n] -= 2*delt
        fminus = fn_error(x, *args)
        x[n] += delt
        #
        gnum[n] = (fplus-fminus)/(2*delt)
        differ = gnum[n]-grad[n]
        if differ == 0:
            one_part_in = 16
        else:
            one_part_in = int(np.log10(np.abs(gnum[n] / differ)))+1
        #
        if one_part_in <= alert_rel_red:
            trm_col = '91'
        elif one_part_in <= alert_rel_yellow:
            trm_col = '93'
        else:
            trm_col = '37'
        #
        coldiff = '\x1b[{0}m{1: 10e}\x1b[0m'.format(trm_col, differ)
        print('{0:> 12g}  {1:> 12g}  {2:12} {3:3d}'.
              format(grad[n], gnum[n], coldiff, one_part_in))


#
# Radford's original single-trajectory function,
# updated to return some extra stuff
#
def one_step(U, grad_U, epsilon, L, current_q, *args):
    #
    q = current_q.copy()
    M = q.size
    p = np.random.normal(size=M)  # independent standard normal variates
    current_p = p.copy()

    # Make a half step for momentum at the beginning
    p -= epsilon * grad_U(q, *args) / 2
    
    for i in range(L):
        # Make a full step for the position
        q += epsilon * p
        
        # Make a full step for the momentum, except at end of trajectory
        if i != L-1:
            p -= epsilon * grad_U(q, *args)

    # Make a half step for momentum at the end.
    p -= epsilon * grad_U(q, *args) / 2
    
    # Evaluate potential and kinetic energies at start and end of trajectory
    current_U = U(current_q, *args)
    current_K = np.sum(current_p**2) / 2
    proposed_U = U(q, *args)
    proposed_K = np.sum(p**2) / 2
    
    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position

    if np.random.uniform() < np.exp(current_U-proposed_U+current_K-proposed_K):
        reject = 0
    else:
        q = current_q  # reject
        reject = 1
        
    return q, p, reject 


#
# Full sampling function
#
def sample(x, energy_func, energy_grad, R, L, epsilon0, burn=0,
           checkgrad=False,  args=()):
    """
    Run HMC sampler
    :param x: initial state
    :param energy_func: the energy function
    :param energy_grad: gradient of the energy function
    :param R: number of samples to return
    :param L: number of steps to simulate dynamics for each sample
    :param epsilon0: step-length of simulation
    :param burn: number of initial samples to discard
    :param checkgrad: True/False to check consistency of _func and _grad
    :param args: iterable containing all additional arguments that must be
    passed to func and grad
    
    :return: 2D array of samples, one per row
    """
    #
    if checkgrad:
        gradient_check(x, energy_func, energy_grad,  *args)
    #
    t_zero = time.time()
    D = len(x)
    Samples = np.empty((R, D))
    num_rejects = 0
    ten_percent_points = (np.arange(10)*R/10).astype(int)
    #
    for n in range(-burn, R):
        #
        if n in ten_percent_points or n == R - 1:
            p10 = int(10 * n / (R - 1))
            progress = "#" * p10 + "-" * (10 - p10)
            t_taken = time.time() - t_zero
            t_togo = (t_taken / (n + burn)) * (R - n)
            txt_togo = "{0:.0f} secs".format(t_togo)
            print("|{0}| {1:3.0%} accepted [ {2} to go ]".
                  format(progress, (n - num_rejects) / (n + 0.01), txt_togo), flush=True)
    
        # Small perturbation of step length to avoid cycles
        epsilon = epsilon0*(1.0 + 0.1*np.random.normal())
        x, p, rej = one_step(energy_func, energy_grad, epsilon, L, x, *args)
        if n >= 0:
            num_rejects += rej
            Samples[n] = x
    #
    accept = (R-num_rejects) / R
    print("HMC: R={0} / L={1} / eps={2:g} / Accept={3:.1%}".
          format(R, L, epsilon0, accept))

    return Samples, num_rejects
