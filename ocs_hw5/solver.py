
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import solve
from scipy.sparse.linalg import spsolve
from scipy.sparse import issparse, spdiags

import matplotlib.pyplot as plt

import scipy as sp
import pdb

import sys, select

def solve_standardLP(A,b,c):
    x = linprog(c,A,b).x
    return x

def solve_affine_scaling(A, b, c, maxiter=50):

    x = np.zeros_like(c)
    energy = []

    x_tilde = np.ones_like(x)

    height, width = A.shape
    I = sp.sparse.identity(width, dtype=float)
    I_height = sp.sparse.identity(height, dtype=float)
    prev_x = x = x_tilde-A.T * ( spsolve(A*A.T, I_height).dot(A*x_tilde-b) )
    print np.min(x), np.max(x)

    alpha = 0.7
    epsilon = 0.01
    maxiter = 8
    for k in range(maxiter):
        H = sp.sparse.csc_matrix(np.diag(x)**2)
        d = (H * (I-A.T*spsolve(A*H*A.T + I_height*10**(-14), I_height)*A*H ))*c
        pdb.set_trace()
        alpha = 1e5
        tmp = x - alpha * d
        while True:
            error = c.dot(tmp)
            print 'alpha ', alpha, ' error ', error
            if np.sum(tmp <= 0.0) == 0:
            # if np.all(tmp) > 0:
                break
            alpha /= 1.01
            tmp = x - alpha * d
        x = x - alpha*0.99 * d

        if abs(c.T.dot(prev_x-x)) <= epsilon:
            print 'stopping criteria'
            break

        prev_x = x.copy()

        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            break
        error = c.dot(x)
        print "k {}, error {}, \t alpha {}".format(k, error, alpha)
        energy.append(error)
        #plot_energy(energy)

    print " leave function, x {} ".format(x)
    return x, energy

def plot_energy(energy):
    plt.clf()
    plt.plot(energy)
    plt.xlabel('Epochs')
    plt.ylabel('Energy')
    plt.grid(True)
    plt.title("Energy Plot")
    plt.pause(0.001)
