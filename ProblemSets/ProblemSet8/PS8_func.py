# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 08:42:57 2017

@author: mmic
"""
import seaborn as sns
import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
import time
import numba


# Function to calculate aggregate value
def Aggregate(matx1, matx2):
    Aggr = (np.multiply(matx1, matx2)).sum()
    return Aggr

# Value Function
@numba.jit
def Vfunc(V, e, betafirm, sizez, sizek, Vmat, pi):
    Vp = np.dot(pi, V)
    for i in range(sizez):  # loop over z
        for j in range(sizek):  # loop over k
            for k in range(sizek): # loop over k'
                Vmat[i, j, k] = e[i, j, k] + betafirm * Vp[i, k]
    return Vmat

# Stationary Distribution Function
@numba.jit
def Sdist(PF, pi, Gamma, sizez, sizek):
    HGamma = np.zeros((sizez, sizek))
    for i in range(sizez):  # z
        for j in range(sizek):  # k
            for m in range(sizez):  # z'
                HGamma[m, PF[i, j]] = \
                    HGamma[m, PF[i, j]] + pi[i, m] * Gamma[i, j]
    return HGamma

# Function for integration
def integrand(x, sigma_z, sigma_eps, rho, mu, z_j, z_jp1):
    val = (np.exp((-1 * ((x - mu) ** 2)) / (2 * (sigma_z ** 2)))
            * (norm.cdf((z_jp1 - (mu * (1 - rho)) - (rho * x)) / sigma_eps)
               - norm.cdf((z_j - (mu * (1 - rho)) - (rho * x)) / sigma_eps)))

    return val