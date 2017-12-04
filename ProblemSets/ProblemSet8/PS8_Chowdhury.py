# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 06:46:39 2017

@author: mmic
"""



# import packages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numba
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import PS8_func as fn


# Values for the parameters
rho = 0.7605
mu = 0.0
sigma_eps = 0.213
alpha_k = 0.297
alpha_l = 0.650
delta = 0.154
psi = 1.08
r = 0.04
betafirm = (1 / (1 + r))
h = 6.616
w = 0.62

# Generating grid points of productivity shocks (z)
num_draws = 90000 # number of shocks to draw
eps = np.random.normal(0.0, sigma_eps, size=(num_draws))
z = np.empty(num_draws)
z[0] = 0.0 + eps[0]
for i in range(1, num_draws):
    z[i] = rho * z[i - 1] + (1 - rho) * mu + eps[i]

sigma_z = sigma_eps / ((1 - rho ** 2) ** (1 / 2))

# Computation of cut-off values
N = 9  # number of grid points
z_cutoffs = (sigma_z * norm.ppf(np.arange(N + 1) / N)) + mu

# compute grid points for z considering log (z) in the function
z_grid = np.exp((((N * sigma_z * (norm.pdf((z_cutoffs[:-1] - mu) / sigma_z)
                              - norm.pdf((z_cutoffs[1:] - mu) / sigma_z)))
              + mu)))

# Computing transition probabilities
pi = np.empty((N, N))
for i in range(N):
    for j in range(N):
        results = integrate.quad(fn.integrand, z_cutoffs[i], z_cutoffs[i + 1],
                                 args = (sigma_z, sigma_eps, rho, mu,
                                         z_cutoffs[j], z_cutoffs[j + 1]))
        pi[i,j] = (N / np.sqrt(2 * np.pi * sigma_z ** 2)) * results[0]



## Define the labor market clearing function
def labor_mkt(w):

    # Get grid points for k (capital stock)
    dens = 7
    # setting bounds for the capital stock space (Guess)
    kbar = 1
    lb_k = 0.001
    ub_k = kbar
    krat = np.log(lb_k / ub_k)
    numb = np.ceil(krat / np.log(1 - delta))
    K = np.zeros(int(numb * dens))
    # we'll create in a way where we pin down the upper bound - since
    # the distance will be small near the lower bound, we'll miss that by little
    for j in range(int(numb * dens)):
        K[j] = ub_k * (1 - delta) ** (j / dens)
    kgrid = K[::-1]
    sizek = kgrid.shape[0]


    # Value function iteration
    # operating profits, op
    sizez = z_grid.shape[0]
    op = np.zeros((sizez, sizek))
    for i in range(sizez):
        for j in range(sizek):
            op[i,j] = ((1 - alpha_l) * ((alpha_l / w) ** (alpha_l / (1 - alpha_l))) *
          ((kgrid[j] ** alpha_k) ** (1 / (1 - alpha_l))) * (z_grid[i] ** (1/(1 - alpha_l))))

    # firm cash flow, e
    e = np.zeros((sizez, sizek, sizek))
    for i in range(sizez):
        for j in range(sizek):
            for k in range(sizek):
                e[i, j, k] = (op[i,j] - kgrid[k] + ((1 - delta) * kgrid[j]) -
                           ((psi / 2) * ((kgrid[k] - ((1 - delta) * kgrid[j])) ** 2)
                            / kgrid[j]))

    # Value funtion iteration
    VFtol = 1e-5
    VFdist = 7.0
    VFmaxiter = 3000
    V = np.zeros((sizez, sizek))  # initial guess at value function
    Vmat = np.zeros((sizez, sizek, sizek))  # initialize Vmat matrix
    Vstore = np.zeros((sizez, sizek, VFmaxiter))  # initialize Vstore array
    VFiter = 1

    start_time = time.clock()
    while VFdist > VFtol and VFiter < VFmaxiter:
        TV = V
        Vmat = fn.Vfunc(V, e, betafirm, sizez, sizek, Vmat, pi)
        Vstore[:, :, VFiter] = V.reshape(sizez, sizek,)  # store value function at each
        # iteration for graphing later
        V = Vmat.max(axis=2)  # apply max operator to Vmat (to get V(k))
        PF = np.argmax(Vmat, axis=2)  # find the index of the optimal k'
        Vstore[:,:, i] = V  # store V at each iteration of VFI
        VFdist = (np.absolute(V - TV)).max()  # check distance between value
        # function for this iteration and value function from past iteration
        VFiter += 1

    VFI_time = time.clock() - start_time
    if VFiter < VFmaxiter:
        print('Value function converged after this many iterations:', VFiter)
    else:
        print('Value function did not converge')
    print('VFI took ', VFI_time, ' seconds to solve')

    VF = V  # solution to the functional equation


    ### Collect optimal values(functions)
    # Optimal capital stock k'
    optK = kgrid[PF]

    # optimal investment I
    optI = optK - (1 - delta) * kgrid

    # optimal labor demand
    LD = np.zeros((sizez, sizek))
    for i in range(sizez):
        for j in range(sizek):
            LD[i,j] = (((alpha_l / w) ** (1 / (1 - alpha_l)))*
                 (z_grid[i] ** (1/(1 - alpha_l)))*((kgrid[j] ** alpha_k) ** (1 / (1 - alpha_l))))


    ### Find Stationary Distribution
    Gamma = np.ones((sizez, sizek)) * (1 / (sizek * sizez))
    SDtol = 1e-12
    SDdist = 7
    SDiter = 0
    SDmaxiter = 1000
    while SDdist > SDtol and SDmaxiter > SDiter:
        HGamma = fn.Sdist(PF, pi, Gamma, sizez, sizek)
        SDdist = (np.absolute(HGamma - Gamma)).max()
        Gamma = HGamma
        SDiter += 1

    if SDiter < SDmaxiter:
        print('Stationary distribution converged after this many iterations: ',
              SDiter)
    else:
        print('Stationary distribution did not converge')


    # Aggregate labor demand
    Aggr_LD = fn.Aggregate(LD, Gamma)

    # Aggregate Investment
    Aggr_I = fn.Aggregate(optI, Gamma)

    # Aggregate Adjustment costs
    Adj_C = psi/2 * np.multiply((optI)**2, 1/kgrid)
    Aggr_Adj_C = fn.Aggregate(Adj_C, Gamma)

    # Aggregate Output
    Y = np.multiply(np.multiply((LD) ** alpha_l, kgrid ** alpha_k),np.transpose([z_grid]))
    Aggr_Y = fn.Aggregate(Y, Gamma)

    # Aggregate Consumption
    C = Aggr_Y - Aggr_Adj_C - Aggr_I


    ## labor supply
    Aggr_LS = w/(h * C)

    MK_equilm = abs(Aggr_LD - Aggr_LS)

    # Three Dimensional Plot of Stationary distribution
    zmat, kmat = np.meshgrid(kgrid, np.log(z_grid))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_ylim(8, 12)
    ax.plot_surface(kmat, zmat, Gamma, rstride=1, cstride=1, cmap=cm.Blues,
                    linewidth=0, antialiased=False)
    ax.view_init(elev=20, azim=20)  # to rotate plot for better view
    ax.set_title(r'Fig 1: Stationary Distribution')
    ax.set_xlabel(r'log(productivity)')
    ax.set_ylabel(r'Capital Stock')
    ax.set_zlabel(r'Density')
    fig.savefig('Graph-A.png', transparent=False, dpi=80, bbox_inches="tight")

    # Plot policy function for k'
    zmat, kmat = np.meshgrid(kgrid, np.log(z_grid))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(kmat, zmat, optK, rstride=1, cstride=1, cmap=cm.Blues,
                    linewidth=0, antialiased=False)
    ax.view_init(elev=20, azim=20)  # to rotate plot for better view
    ax.set_title(r'Fig 2: Policy Function')
    ax.set_xlabel(r'log(productivity)')
    ax.set_ylabel(r'Capital Stock')
    ax.set_zlabel(r'Optimal Capital Stock')
    fig.savefig('Graph-B.png', transparent=False, dpi=80, bbox_inches="tight")


    return MK_equilm

# Call the minimizer
# Minimize with (truncated) Newton's method (called Newton Conjugate Gradient method)
wage_guess = w
GE_results = opt.minimize(labor_mkt, wage_guess, method='Nelder-Mead', tol = 1e-10, options={'maxiter': 5000})



### Output results
Optimal_w = GE_results['x']
print('The equilibrium wage rate is', Optimal_w)
