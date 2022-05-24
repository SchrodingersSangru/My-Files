#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:17:42 2017

@author: jingang
"""

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("data.CSV")
mom=pd.read_csv("momentum.CSV")

name=list(data)
name.append('WML')
name.pop(0)
print len(name)
print name

Mkt=map(float,list(data[data.columns[1]][327:650]))
SMB=map(float,list(data[data.columns[2]][327:650]))
HML=map(float,list(data[data.columns[3]][327:650]))
RMW=map(float,list(data[data.columns[4]][327:650]))
CMA=map(float,list(data[data.columns[5]][327:650]))
RF=map(float,list(data[data.columns[6]][327:650]))
WML=map(float,list(mom[mom.columns[1]][765:1088]))
all_seven=np.matrix([Mkt,SMB,HML,RMW,CMA,RF,WML])
first_six=np.matrix([Mkt,SMB,HML,RMW,CMA,WML])

mean=np.zeros(7)
var=np.zeros(7)
for i in range(7):
    mean[i]=np.mean(all_seven[i,:])
    var[i]=np.std(all_seven[i,:])
    
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(var,mean) 

for i in range(7):                                       
    ax.annotate(name[i],xy=(var[i],mean[i]),xytext=(var[i],mean[i]))
   
plt.xlabel("standard deviation")
plt.ylabel("average return")
plt.show()


import cvxopt as opt
from cvxopt import blas, solvers
np.random.seed(88)

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    #sign=np.random.randint(2,size=n)*2-1
    k = np.random.rand(n)-0.5
    
    return k / sum(k)

print rand_weights(6)


def random_portfolio(returns,lim):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
    if sigma > lim:
        return random_portfolio(returns,lim)
    return mu, sigma

n_portfolios = 1000
means, stds = np.column_stack([
    random_portfolio(returns=np.array([Mkt,SMB]),lim=4) 
    for _ in xrange(n_portfolios)
])
plt.plot(stds, means, 'o', markersize=2)
plt.xlabel('std')
plt.ylabel('mean')

plt.title('Mean and standard deviation of Mkt and SMB')


n_portfolios = 1000
means, stds = np.column_stack([
    random_portfolio(np.array([HML,RMW]),lim=3) 
    for _ in xrange(n_portfolios)
])
plt.plot(stds, means, 'o', markersize=2)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of HML and RMW')



n_portfolios = 1000
means, stds = np.column_stack([
    random_portfolio(np.array([CMA,WML]),lim=4) 
    for _ in xrange(n_portfolios)
])
plt.plot(stds, means, 'o', markersize=2)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of CMA and WML')



n_portfolios = 5000
means, stds = np.column_stack([
    random_portfolio(returns=np.array(first_six),lim=1.6) 
    for _ in xrange(n_portfolios)
])
plt.plot(stds, means, 'o', markersize=2)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of returns of randomly generated portfolios')



def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    
    mus = [10**(5 * t/N - 1.0) for t in range(N)]
   
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.matrix(np.array(np.mean(returns, axis=1))))
  
    # Create constraint matrices
    #G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    #h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    C = opt.matrix(-1.0, (1, n))
    d = opt.matrix(-1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar,A,b,C,d)['x'] 
                  for mu in mus]

    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    #print portfolios[99]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, A,b,C,d)['x']
    return np.asarray(wt), returns, risks


weights, returns, risks = optimal_portfolio(first_six)

plt.plot(stds, means, 'o')
plt.ylabel('mean')
plt.xlabel('std')
plt.plot(risks, returns, 'y-o')


